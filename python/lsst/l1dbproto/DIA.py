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

"""Module defining DIA class and related methods.
"""
from __future__ import annotations

import math

import numpy

from . import generators


class DIA:
    """Class simulating difference image analysis.

    This class is responsible for producing a set of DiaSources and
    DiaForcedSources.
    """

    def __init__(
        self,
        xyz: numpy.ndarray,
        open_angle: float,
        vars: numpy.ndarray,
        n_trans: int,
        detection_fraction: float = 1.0,
    ):
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
        self._detection_fraction = detection_fraction

    def makeSources(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Generate a set of DiaSources.

        Some sources originate from the known variable sources (specified
        in constructor), for those sources we return their index in the
        know sources list, for transient sources this index is -1.

        Returns tuple of two ndarrays:
        1. triplets of xyz coordinates of every source, array shape is (N, 3)
        2. array of indices of variable sources, 1-dim ndarray, transient
           sources have negative indices.
        """
        cos_open = math.cos(self._open_angle / 2.0)

        # calc inner product of every variable source to our pointing direction
        products = numpy.inner(self._xyz, self._vars)

        var_indices = numpy.nonzero(products > cos_open)
        if self._detection_fraction < 1.0:
            mask = numpy.random.uniform(size=len(var_indices[0])) <= self._detection_fraction
            var_indices = (var_indices[0][mask],)

        n_trans = numpy.random.poisson(self._n_trans)
        transients = generators.rand_cone_xyz(self._xyz, self._open_angle, n_trans)

        sources = numpy.concatenate((self._vars[var_indices], transients))
        indices = numpy.concatenate((var_indices[0], -numpy.ones(n_trans, numpy.int64)))

        return sources, indices
