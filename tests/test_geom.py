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
import unittest

import lsst.sphgeom as sph
import numpy as np
from lsst.l1dbproto import geom


class TestGeom(unittest.TestCase):
    """Unit tests for geom module."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_rot_matrix(self) -> None:
        """Testing rotation matrix method"""
        a = np.array([0.0, 0.0, 1.0])
        b = np.array([0.0, 0.0, 1.0])
        R = geom.rotation_matrix(a, b)
        self.assertTrue(np.array_equal(R, np.identity(3)))

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        R = geom.rotation_matrix(a, b)
        expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
        self.assertTrue(np.array_equal(R, expected))

        a = np.array([0.0, -1.0, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        R = geom.rotation_matrix(a, b)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        self.assertTrue(np.array_equal(R, expected))

    def test_make_square_tiles(self) -> None:

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 2, 2)
        self.assertEqual(len(tiles), 4)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 16, 2, np.array([1, 0, 0]))
        self.assertEqual(len(tiles), 32)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 8, 8)
        self.assertEqual(len(tiles), 60)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 8, 8, exclude_disjoint=False)
        self.assertEqual(len(tiles), 64)

        tiles = geom.make_square_tiles(3.5 * math.pi / 180, 15, 15, exclude_disjoint=True)
        self.assertEqual(len(tiles), 15 * 15 - 4 * 6)

    def test_make_camera_tiles(self) -> None:

        tiles = geom.make_camera_tiles(3.5 * math.pi / 180, 2)
        self.assertEqual(len(tiles), 84)

        tiles = geom.make_camera_tiles(3.5 * math.pi / 180, 3)
        self.assertEqual(len(tiles), 189)
        # check that corners are not there
        for ix, iy, tile in tiles:
            self.assertFalse(ix < 3 and iy < 3)
            self.assertFalse(ix < 3 and iy >= 12)
            self.assertFalse(ix >= 12 and iy < 3)
            self.assertFalse(ix >= 12 and iy >= 12)

    def test_make_tiles(self) -> None:

        tiles = geom.make_tiles(3.5 * math.pi / 180, 2)
        self.assertEqual(len(tiles), 4)

        tiles = geom.make_tiles(3.5 * math.pi / 180, -2)
        self.assertEqual(len(tiles), 84)

    def _tri_test_one(
        self, v0: sph.UnitVector3d, v1: sph.UnitVector3d, v2: sph.UnitVector3d, area: float
    ) -> None:
        """Test for triangle area with all permutations of vertices."""
        triangles = [
            (v0, v1, v2),
            (v0, v2, v1),
            (v1, v0, v2),
            (v1, v2, v0),
            (v2, v0, v1),
            (v2, v1, v0),
        ]
        for triangle in triangles:
            a = geom.triangle_area(*triangle)
            self.assertAlmostEqual(a, area)

    def test_area_tri(self) -> None:

        sphere_area = 4 * math.pi

        # one quarter of hemisphere
        v0 = sph.UnitVector3d(0, 0, 1)
        v1 = sph.UnitVector3d(1, 0, 0)
        v2 = sph.UnitVector3d(0, 1, 0)
        self._tri_test_one(v0, v1, v2, sphere_area / 8)

        # another quarter of hemisphere
        v0 = sph.UnitVector3d(0, 0, -1)
        v1 = sph.UnitVector3d(1, 0, 0)
        v2 = sph.UnitVector3d(0, -1, 0)
        self._tri_test_one(v0, v1, v2, sphere_area / 8)

        # small triangle near equator
        dz = 1e-6
        dy = 1e-6
        v0 = sph.UnitVector3d(1, -dy, 0)
        v1 = sph.UnitVector3d(1, dy, 0)
        v2 = sph.UnitVector3d(1, 0, dz)
        area = dz * dy
        self._tri_test_one(v0, v1, v2, area)

        # def test_area_poly(self):

        dz = 1e-6
        dy = 1e-6

        # one quarter of hemisphere
        v0 = sph.UnitVector3d(1, 0, -dz / 2)
        v1 = sph.UnitVector3d(1, 0, dz / 2)
        v2 = sph.UnitVector3d(1, dy, dz / 2)
        v3 = sph.UnitVector3d(1, dy, -dz / 2)
        poly = sph.ConvexPolygon([v0, v1, v2, v3])
        area = geom.poly_area(poly)
        self.assertAlmostEqual(area, dz * dy)


#
#  run unit tests when imported as a main module
#
if __name__ == "__main__":
    unittest.main()
