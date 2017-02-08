#!/bin/env python

"""
Unit tests for l1dbproto.generators module.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import math
import numpy
import unittest

#-----------------------------
# Imports for other modules --
#-----------------------------
from lsst.l1dbproto import generators

#---------------------
# Local definitions --
#---------------------

#-------------------------------
#  Unit test class definition --
#-------------------------------


class test_generators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sphere_xyz(self):
        """ Test for points-on-sphere generator """

        points = generators.rand_sphere_xyz(1)
        self.assertEqual(points.shape, (1, 3))

        points = generators.rand_sphere_xyz(100)
        for norm in numpy.linalg.norm(points, axis=1):
            self.assertAlmostEqual(norm, 1.)

        # covers whole sphere, z in range -1 to 1
        self.assertLess(min(points[:, 2]), 0)
        self.assertGreater(max(points[:, 2]), 0)

        # covers nothern hemisphere, z in range 0 to 1
        points = generators.rand_sphere_xyz(100, hemi=1)
        self.assertGreater(min(points[:, 2]), 0)

        # covers southern hemisphere, z in range -1 to 0
        points = generators.rand_sphere_xyz(100, hemi=-1)
        self.assertLess(max(points[:, 2]), 0)

    def test_cone_xyz(self):
        """ Test for vector-in-cone generator """

        points = generators.rand_cone_xyz(numpy.array([0., 0., 1.]), 0., 1)
        self.assertEqual(points.shape, (1, 3))

        points = generators.rand_cone_xyz(numpy.array([0., 0., 1.]), 0., 100)
        for norm in numpy.linalg.norm(points, axis=1):
            self.assertAlmostEqual(norm, 1.)
        for point in points:
            self.assertTrue(numpy.array_equal(point, numpy.array([0., 0., 1.])))

        points = generators.rand_cone_xyz(numpy.array([0., 0., 1.]), math.pi / 2, 100)
        self.assertGreater(min(points[:, 2]), math.cos(math.pi / 4))
        self.assertGreater(min(points[:, 0]), -math.cos(math.pi / 4))
        self.assertLess(max(points[:, 0]), math.cos(math.pi / 4))
        self.assertGreater(min(points[:, 1]), -math.cos(math.pi / 4))
        self.assertLess(max(points[:, 1]), math.cos(math.pi / 4))


#
#  run unit tests when imported as a main module
#
if __name__ == "__main__":
    unittest.main()
