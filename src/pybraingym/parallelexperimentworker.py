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


from pybraingym.experiment import doEpisode, processLastReward, evaluate
from multiprocessing.managers import BaseManager
from multiprocessing import Value

import abc


class AbstractExperimentWorker(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def getAgent(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def setAgent(self, newAgent):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def doExperiment(self, number=1, render_steps=False):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def getReward(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def getCumulativeReward(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def getQualityRate(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def evaluate(self, number=1):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def demonstrate(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError('You need to define this method in derived class!')


class ProcessExperimentWorker(AbstractExperimentWorker):
    """Wrapper for PyBrain's Experiment class."""

    _ids = Value('i', 0)                ## variable shared between all processes
    

    def __init__(self, experiment, doSingleExperiment, qualityFunctor=None):
        AbstractExperimentWorker.__init__( self )
        #self.objId = procId
        self.objId = self._ids.value
        self._ids.value += 1
        self.exp = experiment
        self.cumulativeReward = 0
        self.experimentExecutor = doSingleExperiment
        self.qualityFunctor = qualityFunctor
        self.qualityRate = 0

    def getId(self):
        return self.objId

    def getAgent(self):
        return self.exp.agent

    def setAgent(self, newAgent):
        self.exp.agent = newAgent

    def getExecutor(self):
        return self.experimentExecutor

    def getQuality(self):
        return self.qualityFunctor

    def doExperiment( self, number=1, render_steps=False ):
        self.cumulativeReward = 0
        for i in range(1, number+1):
            self.experimentExecutor( self, i, render_steps )
            task = self.exp.task
            reward = task.getCumulativeReward()
            self.cumulativeReward += reward
            if self.qualityFunctor is not None:
                self.qualityRate = self.qualityFunctor( i, reward )
            else:
                self.qualityRate = self.cumulativeReward 

    def getReward(self):
        task = self.exp.task
        return task.getCumulativeReward()

    def getCumulativeReward(self):
        return self.cumulativeReward
    
    def getQualityRate(self):
        return self.qualityRate

    def evaluate(self, number=1):
        return evaluate( self.exp, number )

    def demonstrate(self):
        self.doExperiment(1, True)
        return self.getCumulativeReward()

    def close(self):
        self.exp.task.close()

    ## =================================================

    def doEpisode(self, demonstrate=False):
        doEpisode( self.exp, demonstrate )

    def processLastReward(self):
        task = self.exp.task
        agent = self.exp.agent
        processLastReward( task, agent )

    def learn(self):
        agent = self.exp.agent
        agent.learn()


## ===========================================================================


class ManagedExperimentWorker(AbstractExperimentWorker):

    def __init__(self, createExperimentInstance):
        AbstractExperimentWorker.__init__( self )
        self.manager = BaseManager()
        self.manager.register( 'createExperiment', createExperimentInstance )
        self.manager.start()
        self.exp = self.manager.createExperiment()

    def getAgent(self):
        return self.exp.getAgent()

    def setAgent(self, newAgent):
        self.exp.setAgent( newAgent )

    def doExperiment(self, number=1, render_steps=False):
        self.exp.doExperiment( number, render_steps )

    def getReward(self):
        return self.exp.getReward()

    def getCumulativeReward(self):
        return self.exp.getCumulativeReward()

    def getQualityRate(self):
        return self.exp.getQualityRate()

    def evaluate(self, number=1):
        return self.exp.evaluate( number )

    def demonstrate(self):
        return self.exp.demonstrate()

    def close(self):
        self.exp.close()
