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

from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from fib import Fib



def processWorker( args ):
    fib = args[0]
    num = args[1]
    return fib.calc(num)


## ===========================================================================


if __name__ == '__main__':

    procnum = 4
    fib_arg = 37


    BaseManager.register('Fib', Fib)
    manager = BaseManager() 
    manager.start()
        

    procStartTime = time.time()

    calcList = []
    for _ in range(0, procnum):
        calc = manager.Fib()
        calcList.append( (calc, fib_arg) )
    
    with Pool( processes=procnum ) as pool:
        pool.map(processWorker, calcList)

    procEndTime = time.time()

    fib = Fib()
    fib.calc(fib_arg)

    refEndTime = time.time()
    
    procDur = procEndTime - procStartTime
    refDur = refEndTime - procEndTime

    results = []
    for calcPar in calcList:
        fib = calcPar[0]
        val = fib.result()
        results.append( val )

    print("Results:", results)    
    print("Reference duration:", refDur, "sec")
    print("Processes duration:", procDur, "sec", (procDur/refDur * 100), "%")
    