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

"""Module defining methods for generating random things.
"""

import math
import numpy

from .geom import rotation_matrix


def rand_sphere_xyz(count=1, hemi=0, seed=None):
    """
    Generates random points on unit sphere.

    Returns array of random spherical positions, array dimensions
    is (count, 3), it is count * (x, y, z).

    @param count:   number of returned point, major dimension of returned array
    @param hemi:    if 0 then both hemispheres are filled, if positive then
                    only nothern hemisphere is filled, if negative then only
                    southern hemisphere is filled.
    """

    rs = numpy.random.RandomState(seed=seed)

    r = rs.normal(size=(count, 3))
    r /= numpy.linalg.norm(r, axis=1)[:, numpy.newaxis]

    if hemi != 0:
        numpy.absolute(r[:, 2], out=r[:, 2])
    if hemi < 0:
        numpy.negative(r[:, 2], out=r[:, 2])

    return r


def rand_cone_xyz(direction, open_angle, n=1, seed=None):
    """
    Generate random vectors in a cone around given vector.

    @param direction: XYZ vector of the cone axis, must be numpy array
    @param open_angle: opening angle (full) of a cone, radians
    @param n:         number of generated vectors
    @param seed:      seed for generator
    """
    rs = numpy.random.RandomState(seed=seed)
    xy = rs.normal(size=(n, 2))
    xy /= numpy.linalg.norm(xy, axis=1)[:, numpy.newaxis]

    z0 = math.cos(open_angle / 2)
    zs = rs.uniform(z0, 1., size=n)
    rs = numpy.sqrt(1.0 - zs * zs)

    res = numpy.empty((n, 3), 'f8')
    res[:, :2] = rs[:, numpy.newaxis] * xy
    res[:, 2] = zs

    # rotate
    R = rotation_matrix(numpy.array([0., 0., 1.]), direction)
    return numpy.asarray(numpy.inner(res, R))
