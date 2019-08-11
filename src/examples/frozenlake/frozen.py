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
import pylab
from scipy import where
from random import choice
import numpy as np
from collections import deque

import gym

from pybraingym.environment import Transformation
from pybraingym.task import GymTask
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


def getBestValue():
    values = table.params.reshape(16, 4)
    bestActions = []
    for i in range(len(values)):
        stateValues = values[i, :].flatten()
        val = max(stateValues)
        bestActions.append(val)
    return np.array( bestActions ).reshape(4, 4)

def getBestActions():
    values = table.params.reshape(16, 4)
    bestActions = []
    for i in range(len(values)):
        stateValues = values[i, :].flatten()
        action = where(stateValues == max(stateValues))[0]
        action = choice(action)
        bestActions.append(action)
    return np.array( bestActions ).reshape(4, 4)


def getBestReadableActions():
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


def plotData():
    field = getBestActions()
    pylab.pcolor(field)
    pylab.draw()
    pylab.pause(0.000001)


## =============================================================================


gymRawEnv = gym.make('FrozenLake-v0')

task = GymTask.createTask( gymRawEnv )
task.env.setTransformation( EnvTransformation() )

# create value table and initialize with ones
table = ActionValueTable(gymRawEnv.observation_space.n, gymRawEnv.action_space.n)
table.initialize(0.0)
# table.initialize( np.random.rand( table.paramdim ) )

### create agent with controller and learner - use SARSA(), Q() or QLambda() here
## alpha -- learning rate (preference of new information -- update value factor)
## gamma -- discount factor (importance of future reward -- next value factor)

learner = SARSA(alpha=0.3, gamma=0.99)
# learner = Q(alpha=0.3, gamma=0.99)
explorer = learner.explorer
explorer.epsilon = 0.4
explorer.decay = 0.9999

agent = LearningAgent(table, learner)

experiment = Experiment(task, agent)


render_steps = False
imax = 5000


# prepare plotting
if render_steps:
    pylab.gray()
    pylab.gca().invert_yaxis()
    pylab.ion()


print("Starting")


total_reward = 0
period_rewards = deque( maxlen=100 )
best_period_reward = float('-inf')

procStartTime = time.time()

i=0
for i in range(1, imax + 1):
# while(True):
# while(explorer.epsilon > 0.00000001):
    i += 1
    doEpisode( experiment, render_steps )
    processLastReward(task, agent)              ## store final reward for learner
    agent.learn()

    reward = task.getCumulativeReward()
    total_reward += reward
    period_rewards.append(reward)
    sum_reward = np.sum(period_rewards)
    if sum_reward > best_period_reward:
        best_period_reward = sum_reward
    if i >= period_rewards.maxlen and sum_reward == period_rewards.maxlen:
        print("Problem solved -- reached goal in %d consecutive trials" % period_rewards.maxlen)
        break

    if i % 100 == 0:
        print("Episode ended: %i/%i period reward: %d best period reward: %d epsilon: %f" % (i, imax, sum_reward, best_period_reward, explorer.epsilon) )

    # draw the table
    if render_steps and (i % 10 == 0):
        plotData()

procEndTime = time.time()
print("Duration:", (procEndTime - procStartTime), "sec")


demonstrate( experiment, 0.2 )
reward = task.getCumulativeReward()
print("\nFinal demonstration, reward: %d" % ( reward ) )

success_rate = best_period_reward / period_rewards.maxlen * 100
print("Success rate: %d%%" % ( success_rate ) )

task.reset()
task.render()

print( "\nValues:\n", table.params.reshape(16, 4), sep='' )

print( "\nBest actions:\n", getBestReadableActions(), sep='' )

task.close()

if render_steps:
    plotData()
    pylab.ioff()
    pylab.show()

print("\n\nDone")
