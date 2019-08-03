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
from pybraingym.experiment import doEpisode, processLastReward, evaluate
from pybraingym.digitizer import Digitizer, ArrayDigitizer

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import SARSA, Q, QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment

import time
import atexit
import numpy as np
import argparse
import os.path


## =============================================================================


parser = argparse.ArgumentParser(description='Mountain Car solution')
parser.add_argument('-l', '--load', action='store', default=None, help='File with agent state to load from' )
parser.add_argument('-s', '--save', action='store', default=None, help='File with agent state to save to' )
parser.add_argument('-x', '--state', action='store', default=None, help='File with agent state to load and save, same as -l -s' )
args = parser.parse_args()


state_load_file = None
state_save_file = None

if args.state is not None:
    if os.path.isfile( args.state ):
        state_load_file = args.state
        print("Loading state from file:", state_load_file)
    
    state_save_file = args.state
    save_basedir = os.path.dirname(state_save_file)
    if not os.path.exists(save_basedir):
        os.makedirs(save_basedir)
else:
    if args.load is not None:
        if os.path.isfile( args.load ):
            state_load_file = args.load
            print("Loading state from file:", state_load_file)

    if args.save is not None:
        state_save_file = args.save
        save_basedir = os.path.dirname(state_save_file)
        if not os.path.exists(save_basedir):
            os.makedirs(save_basedir)    


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
cartVelocityGroup = Digitizer.buildBins(-0.07, 0.07, 16)

# print("Cart position bins:", cartPositionGroup)
# print("Cart velocity bins:", cartVelocityGroup)

observationDigitizer = ArrayDigitizer( [ cartPositionGroup, cartVelocityGroup ] )
transformation = EnvTransformation(observationDigitizer)

task = GymTask.createTask(gymRawEnv)
env = task.env
env.setTransformation( transformation )
## env.setCumulativeRewardMode()

# create value table and initialize with ones
table = ActionValueTable(observationDigitizer.states, env.numActions)

if state_load_file is not None:
    loadedParams = np.fromfile( state_load_file )
    table._setParameters( loadedParams )
    # print( "raw data:", loadedParams )
else:
    # table.initialize(0.0)
    rand_arr = np.random.rand( table.paramdim )
    table.initialize( rand_arr )

# create agent with controller and learner - use SARSA(), Q() or QLambda() here
## alpha -- learning rate (preference of new information)
## gamma -- discount factor (importance of future reward)

# learner = Q(0.5, 0.99)
learner = SARSA(0.5, 0.99)
# learner = QLambda(0.5, 0.99, 0.9)
explorer = learner.explorer

agent = LearningAgent(table, learner)

experiment = Experiment(task, agent)

## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( task.close )


render_demo = False
render_steps = False
imax = 800
period_print = 10
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
#         print("vals:", table.params.reshape(16 * 4, 3))

procEndTime = time.time()
print("Duration:", (procEndTime - procStartTime), "sec")


if state_save_file is not None:
    print("Storing state to file:", state_save_file)
    ## save learned data
    learnedData = table.params
    learnedData.tofile( state_save_file )
    # print( "raw data:", learnedData )


evaluation = evaluate( experiment, eval_periods )
print( "Evaluation:", evaluation )


# doEpisode( experiment, True )
# reward = task.getCumulativeReward()
# print("\nFinal demonstration, reward: %d" % ( reward ) )
# print("\nDone")


task.close()
