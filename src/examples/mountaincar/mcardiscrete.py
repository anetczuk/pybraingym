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

from pybraingym import GymTask, Transformation
from pybraingym.experiment import doEpisode, processLastReward
from pybraingym.digitizer import Digitizer, ArrayDigitizer

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import SARSA, Q, QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment

import atexit
import numpy as np


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

gymRawEnv = gym.make('MountainCar-v0')


cartPositionGroup = Digitizer.buildBins(-1.2, 0.6, 16)
cartVelocityGroup = Digitizer.buildBins(-0.07, 0.07, 4)

print("Cart position bins:", cartPositionGroup)
print("Cart velocity bins:", cartVelocityGroup)

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

# create agent with controller and learner - use SARSA(), Q() or QLambda() here
## alpha -- learning rate (preference of new information)
## gamma -- discount factor (importance of future reward)

# learner = Q(0.5, 0.99)
learner = SARSA(0.5, 0.99)
# learner = QLambda(0.5, 0.99, 0.9)

agent = LearningAgent(table, learner)

experiment = Experiment(task, agent)


render_steps = False
imax = 18000


print("\nStarting")


## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( task.close )


total_reward = 0

for i in range(1, imax+1):
    agent.reset()
    doEpisode( experiment, render_steps )

    total_reward += task.getCumulativeReward()
    processLastReward(task, agent)              ## store final reward for learner

    agent.learn()

    if i % 100 == 0:
        print("Episode ended: %i/%i reward: %d total reward: %d rate: %f" % (i, imax, task.getCumulativeReward(), total_reward, total_reward / i) )

    if i % 1000 == 0:
        doEpisode( experiment, True )
#         print("vals:", table.params.reshape(16 * 4, 3))


print("\nDone")

task.close()

