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


gymRawEnv = gym.make('Taxi-v2')

transformation = EnvTransformation()
 
task = GymTask.createTask(gymRawEnv)
env = task.env
env.setTransformation( transformation )
## env.setCumulativeRewardMode()
 
## create value table and initialize with ones
table = ActionValueTable(env.numStates, env.numActions)
table = ActionValueTableWrapper(table)
table.initialize(0.0)
# table.initialize( np.random.rand( table.paramdim ) )
 
### create agent with controller and learner - use SARSA(), Q() or QLambda() here
## alpha -- learning rate (preference of new information)
## gamma -- discount factor (importance of future reward)

learner = Q(0.2, 0.99)
# learner = SARSA(0.2, 0.99)
## learner = QLambda(0.5, 0.99, 0.9)
explorer = learner.explorer
explorer.decay = 1.0
 
agent = LearningAgent(table, learner)
 
experiment = Experiment(task, agent)
 
## prevents "ImportError: sys.meta_path is None, Python is likely shutting down"
atexit.register( task.close )
 
 
render_demo = False
imax = 5000
period_print = 100
eval_periods = 100
 
 
print("\nStarting")
 
 
total_reward = 0
period_rewards = deque( maxlen=2 * period_print )
best_avg_reward = float('-inf')
 
procStartTime = time.time()
 
for i in range(1, imax + 1):
    doEpisode( experiment )
    processLastReward(task, agent)              ## store final reward for learner
    agent.learn()
 
    reward = task.getCumulativeReward()
    total_reward += reward
    period_rewards.append(reward)
    avg_reward = np.mean(period_rewards)
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward 
 
    if i % period_print == 0:
        print("Episode ended: %i/%i period reward: %f total reward: %d best avg reward: %f rate: %f" % (i, imax, avg_reward, total_reward, best_avg_reward, total_reward / i) )
 
    if render_demo and i % 1000 == 0:
        doEpisode( experiment, True )
 
procEndTime = time.time()
print("Duration:", (procEndTime - procStartTime), "sec")


evaluation = evaluate( experiment, eval_periods )
print( "Evaluation:", evaluation )

 
demonstrate( experiment, 0.2 )
reward = task.getCumulativeReward()
print("\nFinal demonstration, reward: %d" % ( reward ) )

 
print("Showing data")
data = table.stateArray
data = data.reshape(20, 25)
## print("Data:\n", data)
pylab.pcolor(data)
pylab.colorbar()
pylab.draw()
pylab.show()
 
 
print("\nDone")
 
task.close()
