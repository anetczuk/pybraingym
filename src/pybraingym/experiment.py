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


def doEpisode(experiment, demonstrate=False, delay=0.02):
    if demonstrate:
        doEpisode2(experiment, demonstrate, False, delay )
    else:
        doEpisode2(experiment, demonstrate, True, delay )


def doEpisode2(experiment, render, learn, delay=0.02):
    task = experiment.task
    env = task.env
    agent = experiment.agent

    agent.reset()

    if env.done:
        env.reset()

    agent.newEpisode()

    if learn is True:
        _doEpisodeIterations(experiment, render, delay)
    else:            
        prevlearning = agent.learning
        agent.learning = False
        _doEpisodeIterations(experiment, render, delay)
        agent.learning = prevlearning


def _doEpisodeIterations(experiment, render=False, delay=0.02):
    task = experiment.task
    env = task.env
    
    if render is False:
        while(env.done is False):
            experiment.doInteractions(1)
        return

    ## with rendering
    while(env.done is False):
        env.render()
        if delay > 0:
            time.sleep( delay )
        experiment.doInteractions(1)
    env.render()


def evaluate( experiment, episodes ):
    task = experiment.task
    
    total_reward = 0
    min_reward = float('inf')
    max_reward = -float('inf')
    for _ in range(1, episodes + 1):
        doEpisode2( experiment, False, False )
        reward = task.getCumulativeReward()
        total_reward += reward
        if reward > max_reward:
            max_reward = reward
        if reward < min_reward:
            min_reward = reward
    return (min_reward, max_reward, total_reward / episodes)


def demonstrate( experiment, delay=0.02 ):
    doEpisode2(experiment, True, False, delay )


def processLastReward(task, agent):
    """Store last reward when episode is done.

       Step is important in edge case of Q/SARSA learning
       when reward is only received in last step.
    """
    observation = task.getObservation()
    agent.integrateObservation(observation)
    agent.getAction()
    reward = task.getReward()                   ## repeat last reward
    agent.giveReward(reward)


class SampleExperiment(object):

    def __init__(self, env):
        self.env = env
        self.stepid = 0

    def doInteractions(self, number=1):
        for _ in range(number):
            self._oneInteraction()
        return self.stepid

    def _oneInteraction(self):
        self.stepid += 1
        action = self.env.sampleAction()
        self.env.performAction(action)
        return self.env.reward
