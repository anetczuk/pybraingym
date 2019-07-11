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


class OpenAiEnvironment(Environment):
    
    def __init__(self, openAiEnv, transformation = None):
        Environment.__init__(self)
        self.env = openAiEnv
        self.observation = None
        self.reward = None
        self.done = True
        self.info = None
        self.transform = transformation
    
    def getSensors(self):
        return self.observation
        
    def performAction(self, action):
        if self.transform is not None:
            action = self.transform.action(action)
        self.observation, self.reward, self.done, self.info = self.env.step(action)
        if self.transform is not None:
            self.observation = self.transform.observation(self.observation)

    def reset(self):
        self.done = False
        self.observation = self.env.reset()
        if self.transform is not None:
            self.observation = self.transform.observation(self.observation)
        
    def sampleAction(self):
        return self.env.action_space.sample()
        
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()


    
class Transformation:
    
    def observation(self, observation):
        """ Transform observation value received from OpenAi. Transformed value is passed to PyBrain. """
        return observation

    def action(self, action):
        """ Transform action value received from PyBrain and pass result to OpenAi. """
        return action
