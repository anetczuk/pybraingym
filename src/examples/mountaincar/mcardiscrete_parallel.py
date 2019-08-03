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


import gym

from pybraingym.environment import Transformation
from pybraingym.task import GymTask
from pybraingym.parallelexperiment import ProcessExperiment, MultiExperiment
from pybraingym.digitizer import Digitizer, ArrayDigitizer

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import SARSA, Q, QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment

import time
import atexit
import numpy as np
import copy


## =============================================================================


class EnvTransformation(Transformation):

    def __init__(self, observationDigitizer):
        self.observationDigitizer = observationDigitizer

    def observation(self, observationValue):
        state = self.observationDigitizer.state( observationValue )
        return [ state ]

    def action(self, actionValue):
        ## Gym environment expects one integer value, but PyBrain returns array with single float
        state = actionValue[0]
        return int(state)

    def reward(self, rewardValue):
        return rewardValue


## =============================================================================


## Gym expected action:
##   0 -- left
##   1 -- neutral
##   2 -- right

## Gym observation: [position, velocity]
##        position: (-1.2, 0.6)
##        velocity (-0.07, 0.07)

## Gym reward:
##        -1 for each time step, until the goal position of 0.5 is reached. As with MountainCarContinuous v0, there is no penalty for climbing the left hill, which upon reached acts as a wall.


def createAgent(module):
    # create agent with controller and learner - use SARSA(), Q() or QLambda() here
    ## alpha -- learning rate (preference of new information)
    ## gamma -- discount factor (importance of future reward)

    # learner = Q(0.5, 0.99)
    learner = SARSA(0.5, 0.99)
    # learner = QLambda(0.5, 0.99, 0.9)

    agent = LearningAgent(module, learner)
    return agent


def createExperimentInstance():
    gymRawEnv = gym.make('MountainCar-v0')

    cartPositionGroup = Digitizer.buildBins(-1.2, 0.6, 16)
    cartVelocityGroup = Digitizer.buildBins(-0.07, 0.07, 16)

#     print("Cart position bins:", cartPositionGroup)
#     print("Cart velocity bins:", cartVelocityGroup)

    observationDigitizer = ArrayDigitizer( [ cartPositionGroup, cartVelocityGroup ] )
    transformation = EnvTransformation(observationDigitizer)

    task = GymTask.createTask(gymRawEnv)
    env = task.env
    env.setTransformation( transformation )
    # env.setCumulativeRewardMode()

    # create value table and initialize with ones
    table = ActionValueTable(observationDigitizer.states, env.numActions)
    table.initialize(0.0)
    # table.initialize( np.random.rand( table.paramdim ) )
    agent = createAgent( table )

    experiment = Experiment(task, agent)
    experiment = ProcessExperiment( experiment, ExperimentIteration() )
    return experiment


class ExperimentIteration:

    def __init__(self):
        pass

    def __call__(self, experiment, iteration, render_steps=False):
        experiment.doEpisode(render_steps)
        if render_steps is False:
            experiment.processLastReward()              ## store final reward for learner
            experiment.learn()

        if iteration % 100 == 0:
            reward = experiment.getReward()
            totalReward = experiment.getCumulativeReward() + reward
            print("Episode ended: %i reward: %d total reward: %d rate: %f" % (iteration, reward, totalReward, totalReward / iteration) )


def copyAgentState(fromAgent, currAgent):
    newModule = copy.deepcopy(fromAgent.module)
    newAgent = createAgent( newModule )
    return newAgent


## =============================================================================


# render_steps = False
# imax = 18000

# render_steps = False
# render_demo = False
# parallel_exps = 4
# round_epochs = 20
# rounds_num = int(1000 / round_epochs)

render_steps = False
render_demo = False
parallel_exps = 8
round_epochs = 3000
# rounds_num = int(1000 / round_epochs)
# rounds_num = 10
rounds_num = 1


experiment = MultiExperiment( parallel_exps, createExperimentInstance, copyAgentState )
# experiment = createExperimentInstance()


## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( experiment.close )


print("")
print("Parallel experiments:", parallel_exps)
print("Epochs per round:", round_epochs)
print("Rounds:", rounds_num)

print("\nStarting")

procStartTime = time.time()

for i in range(1, rounds_num + 1):
    experiment.doExperiment(round_epochs, render_steps)
    reward = experiment.getCumulativeReward()
    print("Round ended: %i/%i best reward: %d per epoch: %f" % (i, rounds_num, reward, reward / round_epochs) )

    if render_demo and i % 10 == 0:
        reward = experiment.demoBest()
        print("Demonstration ended, reward: %d" % ( reward ) )

# reward = experiment.demonstrate()
# print("Final demonstration, reward: %d" % ( reward ) )

procEndTime = time.time()
print("Duration:", (procEndTime - procStartTime), "sec")

reward = experiment.demonstrate()
print("\nFinal demonstration, reward: %d" % ( reward ) )
print("\nDone")

experiment.close()
