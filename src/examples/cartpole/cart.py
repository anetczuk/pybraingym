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


# import time
import gym

from pybraingym import OpenAiEnvironment, OpenAiTask, Transformation
from pybraingym.experiment import SampleExperiment, doEpisode, processLastReward
from pybraingym.digitizer import Digitizer, ArrayDigitizer

from pybrain.rl.learners.valuebased import ActionValueTable, ActionValueNetwork
from pybrain.rl.learners import SARSA, Q, QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment
from pybrain.tools.shortcuts import buildNetwork

import atexit
from scipy import where
from random import choice
import numpy as np


## =============================================================================


class CartTransformation(Transformation):

    def __init__(self, observationDigitizer):
        self.observationDigitizer = observationDigitizer

    def observation(self, observationValue):
        state = self.observationDigitizer.state( observationValue )
        return [ state ]
     
    def action(self, actionValue):
        ## OpenAi environment expects one integer value, but PyBrain returns array with single float
        state = actionValue[0]
        return int(state)


## =============================================================================


## OpenAi expected action:
##   0 -- left
##   1 -- right

## openai observation: [position, velocity, pole angle, pole velocity]
##        position: (-2.5, 2.5)
##        velocity (-inf, inf)
##        pole angle (-41.8, 41.8)
##        pole vel: (-inf, inf)

openai_env = gym.make('CartPole-v1')
# openai_env = gym.make('CartPole-v0')

## env.tags['wrapper_config.TimeLimit.max_episode_steps'] = 500


cartPositionGroup = Digitizer.buildBins(-2.4, 2.4, 4)       ## terminates when outside range (-2.4, 2.4)
cartVelocityGroup = Digitizer.buildBins(-1.0, 1.0, 3)
poleAngleGroup = Digitizer.buildBins(-12.0, 12.0, 2)        ## terminates when outside range (-12, 12)
poleVelocityGroup = Digitizer.buildBins(-4.0, 4.0, 8)

print("Cart position bins:", cartPositionGroup)
print("Cart velocity bins:", cartVelocityGroup)
print("Pole angle bins:", poleAngleGroup)
print("Pole velocity bins:", poleVelocityGroup)

observationDigitizer = ArrayDigitizer( [ cartPositionGroup, cartVelocityGroup, poleAngleGroup, poleVelocityGroup ] )

env = OpenAiEnvironment(openai_env, CartTransformation(observationDigitizer), True)

task = OpenAiTask(env)
 
# create value table and initialize with ones
table = ActionValueTable(observationDigitizer.states, 2)
table.initialize(0.0)
# table.initialize( np.random.rand( table.paramdim ) )
 
# create agent with controller and learner - use SARSA(), Q() or QLambda() here
## alpha -- learning rate
## gamma -- discount factor

# learner = Q(0.5, 0.99)
learner = SARSA(0.5, 0.99)
# learner = QLambda(0.5, 0.99, 0.9)

agent = LearningAgent(table, learner)
 
experiment = Experiment(task, agent)


render_steps = False
imax = 3000


print("\nStarting")


## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( env.close )


total_reward = 0

for i in range(1, imax+1):
    agent.reset()
    doEpisode( experiment, render_steps )
    
    total_reward += env.cumReward
    processLastReward(task, agent)              ## store final reward for learner
        
    agent.learn()
    
    if i % 100 == 0:
        print("Episode ended: %i/%i reward: %d total reward: %d rate: %f" % (i, imax, env.cumReward, total_reward, total_reward / i) )
        
    if i % 1000 == 0:
        doEpisode( experiment, True )


print("\nDone")

env.close()

