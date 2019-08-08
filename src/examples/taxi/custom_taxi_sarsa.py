#!/usr/bin/env python3

##
## Code taken from book: "Hands-On Reinforcement Learning with Python" by Sudharsan Ravichandiran
##

#Like we did in Q learning, we import necessary libraries and initialize environment
import gym
import random
import time
from collections import deque
import numpy as np


env = gym.make('Taxi-v2')

alpha = 0.1
gamma = 0.90
epsilon = 0.2

#Then we initialize Q table as dictionary for storing the state-action values
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0

#Now, we define a function called epsilon_greedy for performing action
#according epsilon greedy policy
def epsilon_greedy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])


def epoch(render=False):
    #We store cumulative reward of each episodes in
    r = 0
    step = 0
    epochEpsilon = epsilon
    #Then for every iterations, we initialize the state,
    state = env.reset()
    if render:
        print("Step:", step)
        epochEpsilon = 0.0
        env.render()
        time.sleep(0.2)
    #then we pick up the action using epsilon greedy policy
    action = epsilon_greedy(state, epochEpsilon)
    while True:
        #Then we perform the action in the state and move the next state
        nextstate, reward, done, _ = env.step(action)
        if render:
            print("Step:", step)
            env.render()
            time.sleep(0.2)
        
        #Then we pick up the next action using epsilon greedy policy
        nextaction = epsilon_greedy(nextstate, epochEpsilon)
        
        if render==False:
            #we calculate Q value of the previous state using our update rule
            currValue = Q[(state,action)]
            nextValue = Q[(nextstate,nextaction)]
            updateValue = alpha * (reward + gamma * nextValue - currValue)
            Q[(state,action)] += updateValue
            
        #finally we update our state and action with next action
        #and next state
        action = nextaction
        state = nextstate
        r += reward
        step += 1
        #we will break the loop, if we are at the terminal
        #state of the episode
        if done:
            break

    return r


period_rewards = deque(maxlen=100)
best_reward = float('-inf')

for i in range(4000):
    reward = epoch()
    if reward > best_reward:
        best_reward = reward
    period_rewards.append(reward)
    avg_reward = np.mean(period_rewards)
    print( "Epoch: %5d  reward: %3d  avg: %f  best reward: %3d" % (i, reward, avg_reward, best_reward) )


reward = epoch(True)
print( "\nDemonstration reward: %d\n" % ( reward ) )

env.close()
