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
from collections import deque
import numpy as np

import gym

from pybraingym.environment import Transformation
from pybraingym.task import GymTask
from pybraingym.experiment import doEpisode, processLastReward, evaluate
from pybraingym.digitizer import Digitizer, ArrayDigitizer

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import SARSA, Q, QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment


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


## =============================================================================


## Gym expected action:
##   0 -- left
##   1 -- right

## Gym observation: [position, velocity, pole angle, pole velocity]
##        position: (-2.5, 2.5)
##        velocity (-inf, inf)
##        pole angle (-41.8, 41.8)
##        pole vel: (-inf, inf)

## Reward:
##        A reward of +1 is provided for every timestep that the pole remains upright. 


gymRawEnv = gym.make('CartPole-v1')
# gymRawEnv = gym.make('CartPole-v0')

## env.tags['wrapper_config.TimeLimit.max_episode_steps'] = 500


cartPositionGroup = Digitizer.buildBins(-2.4, 2.4, 6)       ## terminates when outside range (-2.4, 2.4)
cartVelocityGroup = Digitizer.buildBins(-1.0, 1.0, 3)
poleAngleGroup = Digitizer.buildBins(-12.0, 12.0, 2)        ## terminates when outside range (-12, 12)
poleVelocityGroup = Digitizer.buildBins(-4.0, 4.0, 4)

print("Cart position bins:", cartPositionGroup)
print("Cart velocity bins:", cartVelocityGroup)
print("Pole angle bins:", poleAngleGroup)
print("Pole velocity bins:", poleVelocityGroup)

observationDigitizer = ArrayDigitizer( [ cartPositionGroup, cartVelocityGroup, poleAngleGroup, poleVelocityGroup ] )
transformation = EnvTransformation(observationDigitizer)

task = GymTask.createTask(gymRawEnv)
env = task.env
env.setTransformation( transformation )
env.setCumulativeRewardMode()

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
explorer = learner.explorer
explorer.decay = 0.999992

agent = LearningAgent(table, learner)

experiment = Experiment(task, agent)

## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( task.close )


render_demo = False
render_steps = False
imax = 7000
period_print = 100
eval_periods = 100


print("\nStarting")


total_reward = 0
period_reward = 0

procStartTime = time.time()

for i in range(1, imax + 1):
    doEpisode( experiment, render_steps )

    reward = task.getCumulativeReward()
    total_reward += reward
    period_reward += reward
    processLastReward(task, agent)              ## store final reward for learner

    agent.learn()

    if i % period_print == 0:
        epsil = explorer.epsilon
        print("Episode ended: %i/%i period reward: %f total reward: %d rate: %f epsilon: %f" % (i, imax, period_reward / period_print, total_reward, total_reward / i, epsil) )
        period_reward = 0

    if render_demo and i % 1000 == 0:
        doEpisode( experiment, True )

procEndTime = time.time()
print("Duration:", (procEndTime - procStartTime), "sec")


evaluation = evaluate( experiment, eval_periods )
print( "Evaluation:", evaluation )


print("\nDone")

task.close()
