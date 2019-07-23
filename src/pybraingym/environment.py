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


from pybrain.rl.environments.environment import Environment
from gym.spaces.discrete import Discrete


class GymEnvironment(Environment):

    def __init__(self, gymRawEnv):
        Environment.__init__(self)

        actionSpace = gymRawEnv.action_space
        if type(actionSpace) == Discrete:
            self.indim = 1
            self.discreteActions = True
            self.numActions = actionSpace.n

        self.env = gymRawEnv
        self.observation = None
        self.reward = 0
        self.cumReward = 0
        self.done = True
        self.info = None
        self.transform = None
        self.doCumulative = False
        self.doRender = False

    def setRendering(self, render = True):
        self.doRender = render

    def getCumulativeRewardMode(self):
        return self.doCumulative

    def setCumulativeRewardMode(self, cumulativeReward = True):
        self.doCumulative = cumulativeReward

    def setTransformation(self, transformation):
        self.transform = transformation

    # ==========================================================================

    def getSensors(self):
        return self.observation

    def performAction(self, action):
        if self.transform is not None:
            action = self.transform.action(action)
        self.observation, self.reward, self.done, self.info = self.env.step(action)
        if self.transform is not None:
            self.observation = self.transform.observation(self.observation)
            self.reward = self.transform.reward(self.reward)
        self.cumReward += self.reward

    def reset(self):
        self.done = False
        self.reward = 0
        self.cumReward = 0
        self.info = None
        self.observation = self.env.reset()
        if self.transform is not None:
            self.observation = self.transform.observation(self.observation)

    # ==========================================================================

    def getReward(self):
        if self.doCumulative:
            return self.cumReward
        else:
            return self.reward

    def sampleAction(self):
        return self.env.action_space.sample()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()



class Transformation:

    def observation(self, observationValue):
        """ Transform observation value received from OpenAi Gym. Transformed value is passed to PyBrain. """
        return observationValue

    def action(self, actionValue):
        """ Transform action value received from PyBrain and pass result to OpenAi Gym. """
        return actionValue

    def reward(self, rewardValue):
        """ Transform reward value received from OpenAi Gym and pass result to PyBrain. """
        return rewardValue

