#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2019 Arkadiusz Netczuk <dev.arnet@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import time
import atexit
import pylab
from scipy import where
from random import choice
import numpy as np
from collections import deque

import gym

from pybraingym.environment import Transformation
from pybraingym.task import GymTask
from pybraingym.parallelexperiment import ProcessExperiment, createExperiment, executeExperiments
from pybraingym.experiment import doEpisode, processLastReward, demonstrate

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import SARSA, Q, QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment


## observation: current field, begins always in 0

## Gym action:
##   0 -- left
##   1 -- down
##   2 -- right
##   3 -- up

## reward: 1 when reaching goal, otherwise 0


## =============================================================================


class EnvTransformation(Transformation):

    def observation(self, observationValue):
        ## Gym environment returns number, but PyBrain agent expects array with single integer
        return [observationValue]

    def action(self, actionValue):
        ## Gym environment expects one integer value, but PyBrain returns array with single float
        return int(actionValue[0])
    
#     def reward(self, rewardValue):
#         """Transform reward value received from OpenAi Gym and pass result to PyBrain."""
#         if self.env.done and rewardValue == 0:
#             return -1
#         return rewardValue


## =============================================================================


def getBestValue(table):
    values = table.params.reshape(16, 4)
    bestActions = []
    for i in range(len(values)):
        stateValues = values[i, :].flatten()
        val = max(stateValues)
        bestActions.append(val)
    return np.array( bestActions ).reshape(4, 4)

def getBestActions(table):
    values = table.params.reshape(16, 4)
    bestActions = []
    for i in range(len(values)):
        stateValues = values[i, :].flatten()
        action = where(stateValues == max(stateValues))[0]
        action = choice(action)
        bestActions.append(action)
    return np.array( bestActions ).reshape(4, 4)


def getBestReadableActions(table):
    values = table.params.reshape(16, 4)
    bestActions = []
    for i in range(len(values)):
        stateValues = values[i, :].flatten()
        action = where(stateValues == max(stateValues))[0]
        action = choice(action)
        if action == 0:
            bestActions.append("L")
        elif action == 1:
            bestActions.append("D")
        elif action == 2:
            bestActions.append("R")
        elif action == 3:
            bestActions.append("U")
        else:
            bestActions.append("?")
    return np.array( bestActions ).reshape(4, 4)


def plotData(table):
    field = getBestActions(table)
    pylab.pcolor(field)
    pylab.draw()
    pylab.pause(0.000001)


## =============================================================================


def createAgent(module):
    ### create agent with controller and learner - use SARSA(), Q() or QLambda() here
    ## alpha -- learning rate (preference of new information -- update value factor)
    ## gamma -- discount factor (importance of future reward -- next value factor)
    
    learner = SARSA(alpha=0.3, gamma=0.99)
    # learner = Q(alpha=0.3, gamma=0.99)
    explorer = learner.explorer
    explorer.epsilon = 0.4
    explorer.decay = 0.9999
    
    agent = LearningAgent(module, learner)
    return agent


def copyAgentState(fromAgent, currAgent):
    newModule = fromAgent.module.copy()
    newAgent = createAgent( newModule )
    return newAgent


class ExperimentIteration:

    def __init__(self):
        pass

    def __call__(self, experiment, iteration, render_steps=False):
        experiment.doEpisode(render_steps)
        if render_steps is False:
            experiment.processLastReward()              ## store final reward for learner
            experiment.learn()    
        if iteration % 100 == 0:
            expId = experiment.getId()
            reward = experiment.getReward()
            totalReward = experiment.getCumulativeReward() + reward
            quality = experiment.getQuality()
            rate = quality.getRate()
            bestRate = quality.getBestPeriodReward()
            print( "Experiment %s episode ended: %i total reward: %d rate: %d highest period reward: %d" % (expId, iteration, totalReward, rate, bestRate) )


class QualityFunctor:

    def __init__(self):
        self.period_rewards = deque( maxlen=100 )
        self.best_period_reward = float('-inf')
        self.rate = 0

    def getBestPeriodReward(self):
        return self.best_period_reward

    def getRate(self):
        return self.rate

    def getSuccessRate(self):
        success_rate = self.best_period_reward / self.period_rewards.maxlen * 100
        return success_rate

    def __call__(self, iteration, reward):
        self.period_rewards.append(reward)
        self.rate = np.sum(self.period_rewards)
        if self.rate > self.best_period_reward:
            self.best_period_reward = self.rate
        return self.rate


def createExperimentInstance():
    gymRawEnv = gym.make('FrozenLake-v0')
    
    transformation = EnvTransformation()
     
    task = GymTask.createTask(gymRawEnv)
    env = task.env
    env.setTransformation( transformation )
    ## env.setCumulativeRewardMode()
     
    # create value table and initialize with ones
    table = ActionValueTable(gymRawEnv.observation_space.n, gymRawEnv.action_space.n)
    table.initialize(0.0)
    # table.initialize( np.random.rand( table.paramdim ) )
    agent = createAgent(table)
    
    experiment = Experiment(task, agent)
    iterator = ExperimentIteration()
    quality = QualityFunctor()
    experiment = ProcessExperiment( experiment, iterator, quality )
    return experiment


## =============================================================================


render_steps = False
parallel_exps = 8
round_epochs = 3000
rounds_num = 2


experiment = createExperiment( parallel_exps, createExperimentInstance, copyAgentState )


## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( experiment.close )


print("")
print("Parallel experiments:", parallel_exps)
print("Epochs per round:", round_epochs)
print("Rounds:", rounds_num)

print("Starting")


# total_reward = 0
# period_rewards = deque( maxlen=100 )
# best_period_reward = float('-inf')

procStartTime = time.time()

executeExperiments(experiment, rounds_num, round_epochs)

procEndTime = time.time()
print("Duration:", (procEndTime - procStartTime), "sec")


reward = experiment.demonstrate()
print("\nFinal demonstration, reward: %d" % ( reward ) )

success_rate = float('-inf')
for procExp in experiment.getExperiments():
    qual = procExp.getQuality()
    rate = qual.getSuccessRate()
    if rate > success_rate:
        success_rate = rate
print("Success rate: %d%%" % ( success_rate ) ) 

bestExp = experiment.getBestExperiment()
bestAgent = bestExp.getAgent()
table = bestAgent.module
 
print( "\nValues:\n", table.params.reshape(16, 4), sep='' )
 
print( "\nBest actions:\n", getBestReadableActions(table), sep='' )
 
# plotData(table)
# pylab.ioff()
# pylab.show()

experiment.close()

print("\nDone\n")
