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


import numpy as np


class Digitizer:
    """Performs quantization/discretization of numbers."""

    def __init__(self, points):
        self.bins = points
        self.states = len(self.bins) + 1

    def numstates(self):
        return self.states

    def digitize(self, data):
        return np.digitize(data, self.bins)

    def index(self, digitized):
        return digitized

    def state(self, data):
        return self.digitize(data)

    def value(self, state):
        if state < 0:
            raise ValueError("invalid state:", state)
        if state >= self.states:
            raise ValueError("invalid state:", state)
        if state == 0:
            return self.bins[ 0 ]
        if state == self.states -1:
            return self.bins[ state-1 ]
        ## normal case
        return (self.bins[ state ] + self.bins[ state-1 ]) / 2

    @staticmethod
    def buildBins(fromValue, toValue, numBins, includeEdges = False):
        if includeEdges == False:
            if numBins < 2:
                raise AssertionError("invalid parameter: bin - it has to be greater than 1")
            diff = toValue - fromValue
            step = diff/numBins
            bins = []
            currBinValue = fromValue
            for _ in range(0, numBins-1):
                currBinValue += step
                bins.append( currBinValue )
            return bins
        else:
            if numBins < 3:
                raise AssertionError("invalid parameter: bin - it has to be greater than 2")
            bins = [fromValue]
            innerNum = numBins - 2
            if innerNum > 1:
                innerBins = Digitizer.buildBins(fromValue, toValue, innerNum, False)
                bins += innerBins
            bins.append( toValue )
            return bins



class ArrayDigitizer:
    """Performs quantization/discretization of 1-D arrays."""

    def __init__(self, pointsLists):
        self.bins = pointsLists
        self.binsSize = len(self.bins)
        self.states = 1
        for i in range(0, self.binsSize):
            self.states *= (len(self.bins[i]) + 1)

    def numstates(self):
        return self.states

    def digitize(self, data):
        assert len(data) == self.binsSize, "data size have to be the same as number of bin groups"
        ret = []
        for i in range(0, self.binsSize):
            index = np.digitize(data[i], self.bins[i])
            ret.append( index )
        return ret

    def index(self, digitized):
        assert len(digitized) == self.binsSize, "data size have to be the same as number of bin groups"
        state = 0
        multiplier = 1
        for i in range(0, self.binsSize):
            state += multiplier * digitized[i]
            groupStates = (len(self.bins[i]) + 1)
            multiplier *= groupStates
        return state

    def state(self, data):
        digitized = self.digitize(data)
        indexed = self.index(digitized)
        return indexed
