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

"""Configuration class for l1dbproto
"""

__all__ = ["L1dbprotoConfig"]

import math

from lsst.dax.ppdb import PpdbConfig
from lsst.pex.config import Field


class L1dbprotoConfig(PpdbConfig):

    FOV_deg = Field(dtype=float,
                    doc="FOV in degrees",
                    default=3.5)

    transient_per_visit = Field(dtype=int,
                                doc="average number of transients per visit",
                                default=100)
    false_per_visit = Field(dtype=int,
                            doc="average number of false positives per visit",
                            default=5050)

    htm_level = Field(dtype=int,
                      doc="HTM indexing level",
                      default=20)
    htm_max_ranges = Field(dtype=int,
                           doc="Max number of ranges in HTM envelope",
                           default=64)

    @property
    def FOV_rad(self):
        """FOV in radians.
        """
        return self.FOV_deg * math.pi / 180
