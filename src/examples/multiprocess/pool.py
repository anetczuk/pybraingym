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
from fib import fib_classic


def processWorker(num):
    return fib_classic(num)


## ===========================================================================


if __name__ == '__main__':
    """ Spawns 4 subprocesses and 3 threads """

    procnum = 4
    fib_arg = 37

    procStartTime = time.time()

    numbers = []
    for _ in range(0, procnum):
        numbers.append( fib_arg )

    results = None
    with Pool( processes=procnum ) as pool:
        results = pool.map(processWorker, numbers)

    procEndTime = time.time()

    fib_classic(fib_arg)

    refEndTime = time.time()

    procDur = procEndTime - procStartTime
    refDur = refEndTime - procEndTime

    print("Results:", results)
    print("Reference duration:", refDur, "sec")
    print("Processes duration:", procDur, "sec", (procDur / refDur * 100), "%")
