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


from pybrain.rl.environments.task import Task
from pybraingym.environment import GymEnvironment


class GymTask(Task):
    """Extends Task class by implementing reward function."""

    def __init__(self, gymEnvironment):
        """Class constructor.

        Arguments:
        gymEnvironment -- object compatible with interface of GymgymEnvironment class
        """
        Task.__init__(self, gymEnvironment)

    def getReward(self):
        return self.env.getReward()

    def getCumulativeReward(self):
        return self.env.cumReward

    def reset(self):
        self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @staticmethod
    def createTask(gymRawEnvironment):
        env = GymEnvironment( gymRawEnvironment )
        task = GymTask( env )
        return task


