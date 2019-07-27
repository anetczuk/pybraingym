#!/usr/bin/env python3

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

from multiprocessing import Pool, Process
from multiprocessing.sharedctypes import Value
from fib import fib_classic, Fib
import ctypes


def processWorker(obj, num):
    objWorker = ComplexType(obj)
    objWorker.calc(num)


class ComplexData(ctypes.Structure):
    ## here goes definition of fields to serialize -- names and types of data
    _fields_ = [('fib', ctypes.c_int64)]


class ComplexType:
    
    def __init__(self, shared=None):
        if shared != None:
            self.shared = shared
        else:
            self.shared = Value( ComplexData )
        self.worker = Fib()
    
    def calc(self, number):
        self.shared.fib = self.worker.calc(number)
        
    def result(self):
        return self.shared.fib


## ===========================================================================


if __name__ == '__main__':
    """ Spawns 4 subprocesses, no additional threads """

    procnum = 4
    fib_arg = 37

    procStartTime = time.time()

    proclist = []
    for _ in range(0, procnum):
        val = ComplexType()
        proc = Process( target=processWorker, args=[val.shared, fib_arg] )
        proclist.append( (proc, val) )
        proc.start()
 
    results = []
    for (proc, val) in proclist:
        proc.join()
        results.append( val.result() )


    procEndTime = time.time()

    fib_classic(fib_arg)

    refEndTime = time.time()

    procDur = procEndTime - procStartTime
    refDur = refEndTime - procEndTime

    print("Results:", results)
    print("Reference duration:", refDur, "sec")
    print("Processes duration:", procDur, "sec", (procDur / refDur * 100), "%")
