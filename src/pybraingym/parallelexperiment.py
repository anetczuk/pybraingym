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


from pybraingym.parallelexperimentworker import ProcessExperimentWorker as ProcessExperiment    ## backward compatibility
from pybraingym.parallelexperimentworker import ManagedExperimentWorker
from pybraingym.experiment import doEpisode, processLastReward, evaluate
from concurrent.futures import ThreadPoolExecutor

import abc


class ParallelExperiment(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def getExperiments(self):
        raise NotImplementedError('You need to define this method in derived class!')
 
    @abc.abstractmethod
    def doExperiment(self, number=1, render_steps=False):
        raise NotImplementedError('You need to define this method in derived class!')
 
    @abc.abstractmethod
    def getCumulativeReward(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def demonstrate(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def getBestExperiment(self):
        raise NotImplementedError('You need to define this method in derived class!')

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError('You need to define this method in derived class!')
 

class SingleExperiment(ParallelExperiment):
    
    def __init__(self, createExperimentInstance):
        ParallelExperiment.__init__( self )
        self.experiment = createExperimentInstance()
    
    def getExperiments(self):
        return [self.experiment]
    
    def doExperiment(self, number=1, render_steps=False):
        self.experiment.doExperiment( number, render_steps )
    
    def getCumulativeReward(self):
        return self.experiment.getCumulativeReward()
    
    def demonstrate(self):
        return self.experiment.demonstrate()
    
    def getBestExperiment(self):
        return self.experiment
    
    def close(self):
        self.experiment.close()


class MultiExperiment(ParallelExperiment):

    def __init__(self, experimentsNumber, createExperimentInstance, copyAgentState=None):
        ParallelExperiment.__init__( self )
        self.stepid = 0
        self.bestExperiment = None
        self.expNum = experimentsNumber
        self.copyAgentState = copyAgentState
        self.experiments = []
        for _ in range(0, self.expNum):
            procExp = ManagedExperimentWorker( createExperimentInstance )
            self.experiments.append( procExp )

    def getExperiments(self):
        ret = []
        for exp in self.experiments:
            ret.append( exp.exp )
        return ret

    def doExperiment(self, number=1, render_steps=False):
        self.bestExperiment = None
        ## execute experiments
        paramsList = []
        for exp in self.experiments:
            paramsList.append( (exp, number, render_steps) )
        with ThreadPoolExecutor( max_workers=self.expNum ) as pool:
            pool.map(MultiExperiment.processExperiment, paramsList)

        ## find best agent
        self.bestExperiment = self.getBestExperimentIndex()
        if self.copyAgentState is not None:
            self._propagateBestResult()

    @staticmethod
    def processExperiment(params):
        experiment = params[0]
        number = params[1]
        demonstrate = params[2]
        experiment.doExperiment(number, demonstrate)

    def getCumulativeReward(self):
        assert self.bestExperiment >= 0
        exp = self.experiments[ self.bestExperiment ]
        return exp.getCumulativeReward()

    def getBestExperimentIndex(self):
        if self.expNum < 1:
            return -1
        if self.expNum == 1:
            return 0
        retIndex = 0
        maxRew = self.experiments[0].getQualityRate()
        for i in range(1, self.expNum):
            exp = self.experiments[i]
            rew = exp.getQualityRate()
            if rew > maxRew:
                maxRew = rew
                retIndex = i
        return retIndex

    def getBestExperiment(self):
        assert self.bestExperiment >= 0
        bestExp = self.experiments[ self.bestExperiment ]
        return bestExp.exp
    
    def evaluate(self, number=1):
        ## evaluate experiments
        paramsList = []
        for exp in self.experiments:
            paramsList.append( (exp, number) )

        with ThreadPoolExecutor( max_workers=self.expNum ) as pool:
            ret = pool.map(MultiExperiment.processEvaluation, paramsList)
        
        retData = []
        while True:
            try:
                item = next(ret) # lst_iterator.__next__()
                retData.append( item )
            except StopIteration:
                break
        return retData

    @staticmethod
    def processEvaluation(params):
        experiment = params[0]
        number = params[1]
        return experiment.evaluate(number)
    
    def demonstrate(self):
        assert self.bestExperiment >= 0
        bestExp = self.experiments[ self.bestExperiment ]
        return bestExp.demonstrate()

    def close(self):
        for exp in self.experiments:
            exp.close()

    def _propagateBestResult(self):
        assert self.bestExperiment >= 0
        bestExp = self.experiments[ self.bestExperiment ]
        bestAgent = bestExp.getAgent()
        for i in range(0, self.expNum):
            if i == self.bestExperiment:
                continue
            exp = self.experiments[i]
            agent = exp.getAgent()
            newAgent = self.copyAgentState(bestAgent, agent)
            exp.setAgent( newAgent )
            

def createExperiment(experimentsNumber, createExperimentInstance, copyAgentState=None):
    if experimentsNumber == 1:
        return SingleExperiment( createExperimentInstance )
    else:
        return MultiExperiment( experimentsNumber, createExperimentInstance, copyAgentState )


def executeExperiments(multiExperiment, rounds, epochs_per_round):
    for i in range(1, rounds + 1):
        multiExperiment.doExperiment(epochs_per_round, False)
        bestExp = multiExperiment.getBestExperiment()
        rate = bestExp.getQualityRate()
        print("Round ended: %i/%i best rate: %f" % (i, rounds, rate) )
