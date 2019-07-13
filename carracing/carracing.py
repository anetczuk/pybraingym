#!/usr/bin/env python3

import gym


env = gym.make('CarRacing-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

## fix import error (sys.meta_path)
env.env.close()
