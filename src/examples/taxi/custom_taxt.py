#!/usr/bin/env python3

##
## Code taken from book: "Hands-On Reinforcement Learning with Python" by Sudharsan Ravichandiran
##

#Like we did in Q learning, we import necessary libraries and initialize environment
import gym
import random
env = gym.make('Taxi-v2')

alpha = 0.85
gamma = 0.90
epsilon = 0.8

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

for i in range(4000):
    #We store cumulative reward of each episodes in
    r = 0
    #Then for every iterations, we initialize the state,
    state = env.reset()
    #then we pick up the action using epsilon greedy policy
    action = epsilon_greedy(state,epsilon)
    while True:
        #Then we perform the action in the state and move the next state
        nextstate, reward, done, _ = env.step(action)
        #Then we pick up the next action using epsilon greedy policy
        nextaction = epsilon_greedy(nextstate,epsilon)
        #we calculate Q value of the previous state using our update rule
        Q[(state,action)] += alpha * (reward + gamma * Q[(nextstate,nextaction)]-Q[(state,action)])
        
        #finally we update our state and action with next action
        #and next state
        action = nextaction
        state = nextstate
        r += reward
        #we will break the loop, if we are at the terminal
        #state of the episode
        if done:
            break

env.close()
