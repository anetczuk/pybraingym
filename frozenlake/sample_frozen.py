#!/usr/bin/python3

import gym
import time

env = gym.make('FrozenLake-v0')

env.reset()
   
print("\nStarting")

imax = 1000
for i in range(1, imax+1):
    env.render()
    # time.sleep(0.01)
    ## action:
    ##   0 -- left
    ##   1 -- down
    ##   2 -- right
    ##   3 -- up
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("epoch done")
        env.reset()
        
    if i % 100 == 0:
        print(i, "/", imax)    
     
env.close()
   
print("Done")
