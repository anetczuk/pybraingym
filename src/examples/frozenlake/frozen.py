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

from pybraingym import GymEnvironment, GymTask, Transformation
from pybraingym.experiment import doEpisode, processLastReward

from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import SARSA, Q, QLambda
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment

import pylab
from scipy import where
from random import choice
import numpy as np


## observation: current field, begins always in 0

## Gym action:
##   0 -- left
##   1 -- down
##   2 -- right
##   3 -- up

## reward: 1 when reaching goal, otherwise 0


## =============================================================================


class FrozenTransformation(Transformation):

    def observation(self, observationValue):
        ## Gym environment returns number, but PyBrain agent expects array with single integer
        return [observationValue]

    def action(self, actionValue):
        ## Gym environment expects one integer value, but PyBrain returns array with single float
        return int(actionValue[0])


## =============================================================================


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


openai_env = gym.make('FrozenLake-v0')

env = GymEnvironment( openai_env, FrozenTransformation() )
task = GymTask( env )

# create value table and initialize with ones
table = ActionValueTable(openai_env.observation_space.n, openai_env.action_space.n)
table.initialize(0.)
# table.initialize( np.random.rand( table.paramdim ) )

# create agent with controller and learner - use SARSA(), Q() or QLambda() here
learner = SARSA()
# learner = Q()
# learner.batchMode = False        # uses only one sample (interaction)

agent = LearningAgent(table, learner)

experiment = Experiment(task, agent)

render_steps = False
imax = 2000


print("Starting")


# prepare plotting
if render_steps:
    pylab.gray()
    pylab.gca().invert_yaxis()
    pylab.ion()

total_reward = 0
for i in range(1, imax+1):
    agent.reset()
    doEpisode( experiment, render_steps )

    total_reward += env.cumReward
    processLastReward(task, agent)              ## store final reward for learner

    agent.learn()

    if i % 100 == 0:
        print("Episode ended: %i/%i total reward: %d rate: %f" % (i, imax, total_reward, total_reward / i) )

    # draw the table
    if render_steps and (i % 10 == 0):
        plotData()


print("\n\nDone")

env.reset()
env.render()

env.close()


print( "\nValues:\n", table.params.reshape(16, 4), sep='' )

print( "\nBest actions:\n", getBestReadableActions(), sep='' )


if render_steps:
    plotData()
    pylab.ioff()
    pylab.show()

