"""
Module defining methods for generating random things.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import math
import numpy
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _rotation_matrix(a, b):
    """
    Create rotation matrix to rotate vector a into b.

    After http://math.stackexchange.com/a/476311
    """

    v = numpy.cross(a, b)
    sin = numpy.linalg.norm(v)
    if sin == 0:
        return numpy.identity(3)
    cos = numpy.vdot(a, b)
    vx = numpy.mat([[0, -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])

    R = numpy.identity(3) + vx + vx * vx * (1 - cos) / (sin ** 2)

    return R

#------------------------
# Exported definitions --
#------------------------

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
    R = _rotation_matrix(numpy.array([0., 0., 1.]), direction)
    return numpy.inner(res, R).A
