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


from pybraingym.experiment import doEpisode, processLastReward
from multiprocessing.dummy import Pool as ThreadPool
# from concurrent.futures import ThreadPoolExecutor as ThreadPool
from multiprocessing.managers import BaseManager


class SingleExperiment:
    
    def __init__(self, experiment):
        self.exp = experiment
    
    def getAgent(self):
        return self.exp.agent
    
    def setAgent(self, newAgent):
        self.exp.agent = newAgent
    
    def doEpisode(self, demonstrate = False):
        doEpisode( self.exp, demonstrate )
 
    def processLastReward(self):
        task = self.exp.task
        agent = self.exp.agent
        processLastReward( task, agent )
     
    def learn(self):
        agent = self.exp.agent
        agent.learn()
        
    def getCumulativeReward(self):
        task = self.exp.task
        return task.getCumulativeReward()
        
    def close(self):
        self.exp.task.close()
        

## ===========================================================================


class ProcessExperiment:
    
    def __init__(self, createExperimentInstance, doSingleExperiment):
        self.doSingleExperiment = doSingleExperiment
        self.manager = BaseManager()
        self.manager.register( 'createExperiment', createExperimentInstance ) 
        self.manager.start()
        self.exp = self.manager.createExperiment()
    
    def getAgent(self):
        return self.exp.getAgent()
    
    def setAgent(self, newAgent):
        self.exp.setAgent( newAgent )
    
    def doExperiment(self, number = 1, render_steps = False):
        for _ in range(0, number):
            self.doSingleExperiment( self.exp, render_steps )
#             print("Epoch reward:", self.exp.getCumulativeReward())

    def demonstrate(self):
        self.doExperiment(1, True)

    def getCumulativeReward(self):
        return self.exp.getCumulativeReward()
    
    def close(self):
        self.exp.close()



class MultiExperiment(object):
    
    def __init__(self, experimentsNumber, createExperimentInstance, doSingleExperiment, copyAgentState):
        self.stepid = 0
        self.bestExperiment = None
        self.expNum = experimentsNumber
        self.copyAgentState = copyAgentState
        self.experiments = []
        for _ in range(0, self.expNum):
            procExp = ProcessExperiment( createExperimentInstance, doSingleExperiment )
            self.experiments.append( procExp )
    
    def doExperiment(self, number = 1, render_steps = False):
        self.bestExperiment = None
        ## execute experiments
        paramsList = []
        for exp in self.experiments:
            paramsList.append( (exp, number, render_steps) )
        with ThreadPool( processes=self.expNum ) as pool:
#         with ThreadPool( max_workers=self.expNum ) as pool:
            pool.map(MultiExperiment.processExperiment, paramsList)
        
        ## calculate best actor
        self.bestExperiment = self.getMaxRewardExperimentIndex()
        self._propagateBestResult()
    
    @staticmethod
    def processExperiment(params):
        experiment = params[0]
        number = params[1]
        demonstrate = params[2]
        experiment.doExperiment(number, demonstrate)
    
    def demoBest(self):
        assert self.bestExperiment >= 0
        bestExp = self.experiments[ self.bestExperiment ]
        bestExp.demonstrate()
        return bestExp.getCumulativeReward()
    
    def getCumulativeReward(self):
        assert self.bestExperiment >= 0
        exp = self.experiments[ self.bestExperiment ]
        return exp.getCumulativeReward()
    
    def getMaxRewardExperimentIndex(self):
        if self.expNum < 1:
            return -1
        retIndex = 0
        maxRew = self.experiments[0].getCumulativeReward()
        for i in range(1, self.expNum):
            exp = self.experiments[i]
            rew = exp.getCumulativeReward()
            if rew > maxRew:
                maxRew = rew
                retIndex = i
        return retIndex
    
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
        