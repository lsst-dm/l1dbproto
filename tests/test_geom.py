#!/bin/env python

"""
Unit tests for l1dbproto.geom module.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import math
import numpy as np
import unittest

#-----------------------------
# Imports for other modules --
#-----------------------------
from lsst.l1dbproto import geom

#---------------------
# Local definitions --
#---------------------

#-------------------------------
#  Unit test class definition --
#-------------------------------


class test_geom(unittest.TestCase):

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
        expected = np.matrix("0,0,-1;0,1,0;1,0,0", dtype=float)
        self.assertTrue(np.array_equal(R, expected))

        a = np.array([0., -1., 0.])
        b = np.array([0., 0., 1.])
        R = geom.rotation_matrix(a, b)
        expected = np.matrix("1,0,0;0,0,1;0,-1,0", dtype=float)
        self.assertTrue(np.array_equal(R, expected))

    def test_make_square_tiles(self):

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 2, 2)
        self.assertEqual(len(tiles), 4)
        
        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 16, 2, np.array([1, 0, 0]))
        self.assertEqual(len(tiles), 32)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 8, 8)
        self.assertEqual(len(tiles), 60)

#
#  run unit tests when imported as a main module
#
if __name__ == "__main__":
    unittest.main()
