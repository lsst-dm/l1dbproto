"""
Module defining DIA class and related methods.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import math
import numpy

#-----------------------------
# Imports for other modules --
#-----------------------------
from . import generators

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------


class DIA(object):
    """
    Class simulating difference image analysis.

    This class is responsible for producing a set of DiaSources and
    DiaForcedSources.
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, xyz, open_angle, vars, n_trans):
        """
        @param xyz:  unit vector giving pointing direction
        @param open_angle: opening angle (full) of FOV, radians
        @param vars: list (ndarray) of all known variable sources
        @param n_trans: number of transients per visit
        """

        self._xyz = xyz
        self._open_angle = open_angle
        self._vars = vars
        self._n_trans = n_trans

    #-------------------
    #  Public methods --
    #-------------------

    def makeSources(self):
        """
        Generate a set of DiaSources.

        Some sources originate from the known variable sources (specified
        in constructor), for those sources we return their index in the
        know sources list, for transient sources this index is -1.

        Returns tuple of two ndarrays:
        1. triplets of xyz coordinates of every source, array shape is (N, 3)
        2. array of indices of variable sources, 1-dim ndarray, transient
           sources have negative indices.
        """

        cos_open = math.cos(self._open_angle / 2.)

        # calc inner product of every variable source to our pointing direction
        products = numpy.inner(self._xyz, self._vars)

        var_indices = numpy.nonzero(products > cos_open)

        n_trans = numpy.random.poisson(self._n_trans, 1)
        transients = generators.rand_cone_xyz(self._xyz, self._open_angle, n_trans)

        sources = numpy.concatenate((self._vars[var_indices], transients))
        indices = numpy.concatenate((var_indices[0], -numpy.ones(n_trans, numpy.int64)))

        return sources, indices
