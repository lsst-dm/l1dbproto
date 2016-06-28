"""
Module defining methods for generating random things.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import numpy

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

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

