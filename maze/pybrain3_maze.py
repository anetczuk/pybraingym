#!/usr/bin/env python
__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

""" This example demonstrates how to use the discrete Temporal Difference
Reinforcement Learning algorithms (SARSA, Q, Q(lambda)) in a classical
fully observable MDP maze task. The goal point is the top right free
field. """

from scipy import * #@UnusedWildImport
import pylab

from pybrain3.rl.environments.mazes import Maze, MDPMazeTask
from pybrain3.rl.learners.valuebased import ActionValueTable
from pybrain3.rl.agents import LearningAgent
from pybrain3.rl.learners import Q, QLambda, SARSA #@UnusedImport
from pybrain3.rl.explorers import BoltzmannExplorer #@UnusedImport
from pybrain3.rl.experiments import Experiment


# create the maze with walls (1)
envmatrix = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],     #  0 -  8
                   [1, 0, 0, 1, 0, 0, 0, 0, 1],     #  9 - 17
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],     # 18 - 26
                   [1, 0, 0, 1, 0, 0, 1, 0, 1],     # 27 - 35 
                   [1, 0, 0, 1, 0, 1, 1, 0, 1],     # 36 - 44
                   [1, 0, 0, 0, 0, 0, 1, 0, 1],     # 45 - 53
                   [1, 1, 1, 1, 1, 1, 1, 0, 1],     # 54 - 62
                   [1, 0, 0, 0, 0, 0, 0, 0, 1],     # 63 - 71
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]])    # 72 - 81

env = Maze(envmatrix, (7, 7))

# create task
task = MDPMazeTask(env)

# create value table and initialize with ones
table = ActionValueTable(81, 4)
table.initialize(1.)

# create agent with controller and learner - use SARSA(), Q() or QLambda() here
learner = SARSA()

# standard exploration is e-greedy, but a different type can be chosen as well
# learner.explorer = BoltzmannExplorer()

# create agent
agent = LearningAgent(table, learner)

# create experiment
experiment = Experiment(task, agent)
 
# prepare plotting
# pylab.gray()
pylab.ion()

imax = 1000
for i in range(1, imax+1):
     
    ## interact with the environment (here in batch mode)
    # experiment.doInteractions(100)
    for j in range(100):
        experiment.doInteractions(1)
        # print( env )
    agent.learn()
    agent.reset()
 
    # and draw the table
    if i % 10 == 0:
        field = table.params.reshape(81,4).max(1).reshape(9,9)
        pylab.pcolor(field)
        pylab.draw()
        pylab.pause(0.000001)
        
    if i % 100 == 0:
        print(i, "/", imax)


print("Done")

pylab.ioff()    
pylab.show()
