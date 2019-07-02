#!/usr/bin/python


## Adapted from the official example
## See http://terokarvinen.com/2016/install-openai-universe-on-ubuntu-16-04
import sys
assert sys.version_info[0]==3 # python3 required
import gym
import universe  # register the universe environments


env = gym.make('flashgames.DuskDrive-v0')

env.configure(remotes=1)  # automatically creates a local docker container
# observation_n = env.reset()
# while True:
#  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
#  observation_n, reward_n, done_n, info = env.step(action_n)
#  env.render()

