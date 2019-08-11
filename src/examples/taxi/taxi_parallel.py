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
import numpy as np
from collections import deque

import gym

from pybraingym.environment import Transformation
from pybraingym.task import GymTask
from pybraingym.parallelexperiment import ProcessExperiment, createExperiment, executeExperiments
from pybraingym.interface import ActionValueTableWrapper
from pybraingym.experiment import doEpisode, processLastReward, evaluate, demonstrate

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import Q, SARSA
# from pybrain.rl.learners import QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment


## =============================================================================


class EnvTransformation(Transformation):

    def __init__(self):
        pass

    def observation(self, observationValue):
        ## gets single int
        return [observationValue]

    def action(self, actionValue):
        ## Gym environment expects one integer value, but PyBrain returns array with single float
        state = actionValue[0]
        return int(state)


## =============================================================================


## Gym expected action:
##     Actions: 
##     There are 6 discrete deterministic actions:
##     - 0: move south
##     - 1: move north
##     - 2: move east 
##     - 3: move west 
##     - 4: pickup passenger
##     - 5: dropoff passenger

## Gym observation:
##    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations. 
##    state space is represented by: (taxi_row, taxi_col, passenger_location, destination)

## Rewards:
##    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.


def createAgent(module):
    ### create agent with controller and learner - use SARSA(), Q() or QLambda() here
    ## alpha -- learning rate (preference of new information -- update value factor)
    ## gamma -- discount factor (importance of future reward -- next value factor)
    
    learner = Q(0.2, 0.99)
    # learner = SARSA(0.2, 0.99)
    ## learner = QLambda(0.5, 0.99, 0.9)
    explorer = learner.explorer
    explorer.decay = 1.0
     
    agent = LearningAgent(module, learner)
    return agent


def copyAgentState(fromAgent, currAgent):
    newModule = fromAgent.module.copy()
    newAgent = createAgent( newModule )
    return newAgent


def experimentIteration( experiment, iteration, render_steps=False ):
    experiment.doEpisode(render_steps)
    if render_steps is False:
        experiment.processLastReward()              ## store final reward for learner
        experiment.learn()

    if iteration % 100 == 0:
        reward = experiment.getReward()
        totalReward = experiment.getCumulativeReward() + reward
        print("Episode ended: %i total reward: %d rate: %f" % (iteration, totalReward, totalReward / iteration ))


def createExperimentInstance():
    gymRawEnv = gym.make('Taxi-v2')
    
    transformation = EnvTransformation()
     
    task = GymTask.createTask(gymRawEnv)
    env = task.env
    env.setTransformation( transformation )
    ## env.setCumulativeRewardMode()
     
    ## create value table and initialize with ones
    table = ActionValueTable(env.numStates, env.numActions)
#     table = ActionValueTableWrapper(table)
    table.initialize(0.0)
    # table.initialize( np.random.rand( table.paramdim ) )
    agent = createAgent(table)
     
    experiment = Experiment(task, agent)
    experiment = ProcessExperiment( experiment, experimentIteration )
    return experiment


## =============================================================================


render_demo = False
render_steps = False
parallel_exps = 8
round_epochs = 3000
rounds_num = 1
# imax = 7000
period_print = 100
eval_periods = 100
# render_demo = False
# imax = 5000
# period_print = 100
# eval_periods = 100


experiment = createExperiment( parallel_exps, createExperimentInstance, copyAgentState )


## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( experiment.close )


print("")
print("Parallel experiments:", parallel_exps)
print("Epochs per round:", round_epochs)
print("Rounds:", rounds_num)

print("\nStarting")


procStartTime = time.time()

executeExperiments(experiment, rounds_num, round_epochs)
 
procEndTime = time.time()
print("Duration:", (procEndTime - procStartTime), "sec")


evaluation = experiment.evaluate( eval_periods )
print( "Evaluation:", evaluation )

 
reward = experiment.demonstrate()
print("\nFinal demonstration, reward: %d" % ( reward ) )

experiment.close()

print("\nDone")
