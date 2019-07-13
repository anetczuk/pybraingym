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


def doEpisode(experiment, render = False):
    task = experiment.task
    env = task.env

    if env.done:
        env.reset()

    if render:
        env.render()
        time.sleep(0.02)

    while(env.done == False):
        experiment.doInteractions(1)
        if render:
            env.render()
            time.sleep(0.02)


def processLastReward(task, agent):
    """ Store last reward when episode is done.
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

    def doInteractions(self, number = 1):
        for _ in range(number):
            self._oneInteraction()
        return self.stepid

    def _oneInteraction(self):
        self.stepid += 1
        action = self.env.sampleAction()
        self.env.performAction(action)
        return self.env.reward
