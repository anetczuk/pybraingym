#!/usr/bin/python3

import time
import gym


env = gym.make('CartPole-v0')
env.reset()
  
print("\nStarting")
 
for _ in range(10000):
    env.render()
    time.sleep(0.01)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()
 
env.close()
  
print("Done")
 
 
# ## fix import error (sys.meta_path)
# env.env.close()
