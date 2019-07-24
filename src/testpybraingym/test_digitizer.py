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

from pybraingym.digitizer import Digitizer, ArrayDigitizer


class DigitizerTest(unittest.TestCase):
    def setUp(self):
        ## Called before testfunction is executed
        pass

    def tearDown(self):
        ## Called after testfunction was executed
        pass

    def test_numstates(self):
        digitizer = Digitizer([0.0])
        states = digitizer.numstates()
        self.assertEqual(states, 2)

    def test_digitize(self):
        digitizer = Digitizer( [0.0] )
        indexes = digitizer.digitize( [-1.0, 1.0] )
        npt.assert_equal(indexes, [0, 1])

    def test_state(self):
        digitizer = Digitizer( [0.0] )
        indexes = digitizer.digitize( [-1.0, 1.0] )
        npt.assert_equal(indexes, [0, 1])

    def test_value_badInput_negative(self):
        digitizer = Digitizer( [0.0, 2.0] )
        self.assertRaises( ValueError, digitizer.value, -1 )

    def test_value_badInput_positive(self):
        digitizer = Digitizer( [0.0, 2.0] )
        self.assertRaises( ValueError, digitizer.value, 3 )

    def test_value_0(self):
        digitizer = Digitizer( [0.0, 2.0] )
        value = digitizer.value(0)
        npt.assert_array_almost_equal(value, 0.0, 2)

    def test_value_1(self):
        digitizer = Digitizer( [0.0, 2.0] )
        value = digitizer.value(1)
        npt.assert_array_almost_equal(value, 1.0, 2)

    def test_value_2(self):
        digitizer = Digitizer( [0.0, 2.0] )
        value = digitizer.value(2)
        npt.assert_array_almost_equal(value, 2.0, 2)

    def test_value_states(self):
        digitizer = Digitizer( [0.0, 2.0] )
        value = digitizer.value( digitizer.states - 1 )
        npt.assert_array_almost_equal(value, 2.0, 2)

    def test_values_edges_3(self):
        digitizer = Digitizer.build(-1.0, 1.0, 3, True)
        npt.assert_array_almost_equal(digitizer.values, [-1.0, 0.0, 1.0], 3)

    def test_values_edges_5(self):
        digitizer = Digitizer.build(-1.0, 1.0, 5, True)
        npt.assert_array_almost_equal(digitizer.values, [-1.0, -0.5, 0.0, 0.5, 1.0], 3)

    def test_values_noEdges_3(self):
        digitizer = Digitizer.build(-1.0, 1.0, 3, False)
        npt.assert_array_almost_equal(digitizer.values, [-0.333, 0.0, 0.333], 3)

    def test_values_noEdges_5(self):
        digitizer = Digitizer.build(-1.0, 1.0, 5, False)
        npt.assert_array_almost_equal(digitizer.values, [-0.6, -0.3, 0.0, 0.3, 0.6], 3)

    def test_buildBins_bins_bad(self):
        self.assertRaises( AssertionError, Digitizer.buildBins, -1.0, 1.0, 1 )

    def test_buildBins_bins_bad_edges(self):
        self.assertRaises( AssertionError, Digitizer.buildBins, -1.0, 1.0, 2, True )

    def test_buildBins_bins02(self):
        bins = Digitizer.buildBins(0.0, 12.0, 2)
        npt.assert_equal(bins, [6.0])

    def test_buildBins_bins03(self):
        bins = Digitizer.buildBins(0.0, 12.0, 3)
        npt.assert_equal(bins, [4.0, 8.0])

    def test_buildBins_bins03_edges(self):
        bins = Digitizer.buildBins(0.0, 12.0, 3, True)
        npt.assert_equal(bins, [0.0, 12.0])

    def test_buildBins_bins04_edges(self):
        bins = Digitizer.buildBins(0.0, 12.0, 4, True)
        npt.assert_equal(bins, [0.0, 6.0, 12.0])

    def test_build_noEdge_5(self):
        digitizer = Digitizer.build(0.0, 10.0, 5, False)
        self.assertEqual(digitizer.states, 5)
        npt.assert_array_almost_equal(digitizer.bins, [2., 4., 6., 8.], 3)
        npt.assert_array_almost_equal(digitizer.values, [2., 3.5, 5., 6.5, 8.], 3)

    def test_build_edge_5(self):
        digitizer = Digitizer.build(0.0, 10.0, 5, True)
        self.assertEqual(digitizer.states, 5)
        npt.assert_array_almost_equal(digitizer.bins, [0., 3.333, 6.667, 10.], 3)
        npt.assert_array_almost_equal(digitizer.values, [0., 2.5, 5., 7.5, 10.], 3)


class ArrayDigitizerTest(unittest.TestCase):
    def setUp(self):
        ## Called before testfunction is executed
        pass

    def tearDown(self):
        ## Called after testfunction was executed
        pass

    def test_numstates(self):
        digitizer = ArrayDigitizer( [[0.0], [0.0, 10.0]] )
        states = digitizer.numstates()
        self.assertEqual(states, 6)

    def test_digitize(self):
        digitizer = ArrayDigitizer( [[0.0], [0.0, 10.0]] )
        indexes = digitizer.digitize( [-1.0, 1.0] )
        npt.assert_equal(indexes, [0, 1])

    def test_digitize_badSize(self):
        digitizer = ArrayDigitizer( [[0.0], [0.0, 10.0]] )
        self.assertRaises( AssertionError, digitizer.digitize, [-1.0, 1.0, 5.0] )

    def test_index_0(self):
        digitizer = ArrayDigitizer( [[3.0, 6.0], [5.0]] )
        state = digitizer.state( [0.0, 0.0] )
        npt.assert_equal(state, 0)

    def test_index_1(self):
        digitizer = ArrayDigitizer( [[3.0, 6.0], [5.0]] )
        state = digitizer.state( [5.0, 0.0] )
        npt.assert_equal(state, 1)

    def test_index_4(self):
        digitizer = ArrayDigitizer( [[3.0, 6.0], [5.0]] )
        state = digitizer.state( [5.0, 10.0] )
        npt.assert_equal(state, 4)

    def test_index_5(self):
        digitizer = ArrayDigitizer( [[3.0, 6.0], [5.0]] )
        state = digitizer.state( [10.0, 10.0] )
        npt.assert_equal(state, 5)

    def test_state(self):
        digitizer = ArrayDigitizer( [[5.0], [3.0, 6.0]] )
        state = digitizer.state( [0.0, 5.0] )
        npt.assert_equal(state, 2)
