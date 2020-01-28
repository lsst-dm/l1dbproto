#!/bin/env python

# This file is part of l1dbproto.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Unit tests for l1dbproto.geom module.
"""

import math
import numpy as np
import unittest

from lsst.l1dbproto import geom


class TestGeom(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_rot_matrix(self):
        """ Testing rotation matrix method """

        a = np.array([0., 0., 1.])
        b = np.array([0., 0., 1.])
        R = geom.rotation_matrix(a, b)
        self.assertTrue(np.array_equal(R, np.identity(3)))

        a = np.array([1., 0., 0.])
        b = np.array([0., 0., 1.])
        R = geom.rotation_matrix(a, b)
        expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
        self.assertTrue(np.array_equal(R, expected))

        a = np.array([0., -1., 0.])
        b = np.array([0., 0., 1.])
        R = geom.rotation_matrix(a, b)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        self.assertTrue(np.array_equal(R, expected))

    def test_make_square_tiles(self):

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 2, 2)
        self.assertEqual(len(tiles), 4)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 16, 2, np.array([1, 0, 0]))
        self.assertEqual(len(tiles), 32)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 8, 8)
        self.assertEqual(len(tiles), 60)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 8, 8, exclude_disjoint=False)
        self.assertEqual(len(tiles), 64)


#
#  run unit tests when imported as a main module
#
if __name__ == "__main__":
    unittest.main()
