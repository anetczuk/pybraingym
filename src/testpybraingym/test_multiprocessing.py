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


import unittest
import numpy.testing as npt

import numpy as np
from multiprocessing import Pool as ThreadPool                      ## NOT affected by GIL
from multiprocessing import Process
from multiprocessing.managers import BaseManager


## ====================================================


def doubler(number):
    return number * 2


class Custom:

    def __init__(self):
        self.data = 2

    def calc(self, num):
        self.data *= num


class Calc:

    def __init__(self):
        self.num = -1
        self.arr = np.array( [1, 2] )
        self.cust = Custom()

    def get(self):
        return self.num

    def getArr(self):
        return self.arr

    def getCust(self):
        return self.cust.data

    def calc(self, num):
        self.num = num * 10
        self.arr = self.arr * num
        self.cust.calc(num)


def processWorker(calc):
    calc.calc(3)


## ====================================================


class MultiprocessingTest(unittest.TestCase):

    def test_pool_return(self):
        numbers = [5, 10, 20]
        with ThreadPool( processes=3 ) as pool:
            doubled = pool.map(doubler, numbers)
        self.assertEqual(doubled, [10, 20, 40])

    def test_Process_obj(self):
        calc = Calc()
        calc.num = 11

        proc = Process( target=processWorker, args=(calc,))
        proc.start()
        proc.join()

        self.assertEqual(calc.num, 11)

    def test_Pool_obj(self):
        obj = Calc()
        numbers = [3]
        with ThreadPool( processes=3 ) as pool:
            pool.map(obj.calc, numbers)

        self.assertEqual(obj.num, -1)

    def test_proxy(self):
        BaseManager.register('CalcClass', Calc)
        manager = BaseManager()
        manager.start()
        inst = manager.CalcClass()

        p = Process(target=processWorker, args=[inst])
        p.start()
        p.join()

        self.assertEqual(inst.get(), 30)
        npt.assert_equal(inst.getArr(), [3, 6])
        self.assertEqual(inst.getCust(), 6)

