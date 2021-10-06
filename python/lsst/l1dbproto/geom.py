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

"""Module defining methods for handling geometry.
"""
from __future__ import annotations

import logging
import math
import numpy as np
from typing import List, Tuple

import lsst.sphgeom as sph


_LOG = logging.getLogger('ap_proto')


def rotation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Create rotation matrix to rotate vector a into b.

    After http://math.stackexchange.com/a/476311

    Parameters
    ----------
    a,b
        xyz-vectors
    """

    v = np.cross(a, b)
    sin = np.linalg.norm(v)
    if sin == 0:
        return np.identity(3)
    cos = np.vdot(a, b)
    vx = np.mat([[0, -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

    R = np.identity(3) + vx + vx * vx * (1 - cos) / (sin ** 2)

    return R


def make_square_tiles(open_angle: float, nx: int, ny: int,
                      direction: np.ndarray = np.array([0., 0., 1.]),
                      exclude_disjoint: bool = True
                      ) -> List[Tuple[int, int, sph.ConvexPolygon]]:
    """Generate mosaic of square tiles covering round patch of sky.

    Returns the list of tiles, each tile is represented by a tuple containg
    three elements: division index for x axis, division index for y axis, and
    `sphgeom.ConvexPolygon` object.

    Parameters
    ----------
    open_angle: `float`
        opening angle (full) of a cone, radians
    nx: `int`
        number of divisions in "x" axis
    ny: `int`
        number of divisions in "y" axis
    direction:
        XYZ vector of the cone axis, must be numpy array
    exclude_disjoint : `bool`, optional
        If `True` (default) then do not include tiles which have no overlap
        with FOV.

    Returns
    -------
    `list` of 3-tuples
    """

    half_open_angle = open_angle/2

    # make rotation matrix
    R = rotation_matrix(np.array([0., 0., 1.]), direction)
    cone = sph.Circle(sph.UnitVector3d(float(direction[0]), float(direction[1]), float(direction[2])),
                      sph.Angle.fromRadians(half_open_angle))

    # build mosaic at a plane perpendicular to z axis at z=+1
    half_height = math.tan(half_open_angle)
    x_delta = 2*half_height / nx
    y_delta = 2*half_height / ny
    tiles = []
    for ix in range(nx):
        x0 = - half_height + ix*x_delta
        x1 = - half_height + (ix+1)*x_delta
        for iy in range(ny):
            y0 = - half_height + iy*y_delta
            y1 = - half_height + (iy+1)*y_delta

            # move it all into numpy array to do rotation
            points = np.array([[x0, y0, 1.],
                               [x0, y1, 1.],
                               [x1, y0, 1.],
                               [x1, y1, 1.]])
            points = np.asarray(np.inner(points, R))

            # make vectors, those normaliza internally
            c00 = sph.UnitVector3d(points[0][0], points[0][1], points[0][2])
            c01 = sph.UnitVector3d(points[1][0], points[1][1], points[1][2])
            c10 = sph.UnitVector3d(points[2][0], points[2][1], points[2][2])
            c11 = sph.UnitVector3d(points[3][0], points[3][1], points[3][2])

            # make a tile and check that it overlaps with cone
            tile = sph.ConvexPolygon([c00, c01, c11, c10])
            relation = cone.relate(tile)
            if not relation & sph.DISJOINT or not exclude_disjoint:
                tiles.append((ix, iy, tile))

    return tiles
