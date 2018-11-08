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

import datetime
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
    divide = Field(dtype=int,
                   doc="Divide FOV into NUM*NUM tiles for parallel processing",
                   default=1)
    interval = Field(dtype=int,
                     doc='Interval between visits in seconds, def: 45',
                     default=45)
    sources_region = Field(dtype=bool,
                           default=False,
                           doc='Use region-based select for DiaSource')
    start_time = Field(dtype=str,
                       default="2020-01-01 03:00:00",
                       doc=('Starting time, format: YYYY-MM-DD hh:mm:ss'
                            '. Time is assumed to be in UTC time zone. Used only at'
                            ' first invocation to intialize database.'))
    start_visit_id = Field(dtype=int,
                           default=1,
                           doc='Starting visit ID. Used only at first invocation'
                           ' to intialize database.')
    sources_file = Field(dtype=str,
                         doc='Name of input file with sources (numpy data)',
                         default="var_sources.npy")

    @property
    def FOV_rad(self):
        """FOV in radians.
        """
        return self.FOV_deg * math.pi / 180

    @property
    def start_time_dt(self):
        """start_time as datetime.
        """
        dt = datetime.datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
        return dt
