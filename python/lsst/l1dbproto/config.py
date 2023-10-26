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

from lsst.daf.base import DateTime
from lsst.pex.config import Config
from lsst.pex.config import Field, ChoiceField


class L1dbprotoConfig(Config):

    FOV_deg = Field(dtype=float,
                    doc="FOV in degrees",
                    default=3.5)
    transient_per_visit = Field(dtype=int,
                                doc="average number of transients per visit",
                                default=100)
    false_per_visit = Field(dtype=int,
                            doc="average number of false positives per visit",
                            default=5050)
    divide = Field(dtype=int,
                   doc=("Divide FOV into NUM*NUM tiles for parallel processing. "
                        "If negative means camera style tiling with 5x5 rafts "
                        "each subdivided in both directions into negated value "
                        "of this parameter."),
                   default=1)
    interval = Field(dtype=int,
                     doc='Interval between visits in seconds, def: 45',
                     default=45)
    forced_cutoff_days = Field(dtype=int,
                               doc=("Period after which we stop forced photometry "
                                    "if there was no observed source, def: 30"),
                               default=30)
    start_time = Field(dtype=str,
                       default="2020-01-01T20:00:00",
                       doc=('Starting time, format: YYYY-MM-DDThh:mm:ss'
                            '. Time is assumed to be in UTC time zone. Used only at'
                            ' first invocation to initialize database.'))
    start_visit_id = Field(dtype=int,
                           default=1,
                           doc='Starting visit ID. Used only at first invocation'
                           ' to intialize database.')
    sources_file = Field(dtype=str,
                         doc='Name of input file with sources (numpy data)',
                         default="var_sources.npy")
    mp_mode = ChoiceField(dtype=str,
                          allowed=dict(fork="Forking mode", mpi="MPI mode"),
                          doc='multiprocessing mode, only for `divide > 1` or `divide < 0',
                          default="fork")
    src_read_duty_cycle = Field(
        dtype=float,
        doc=("Fraction of visits for which (forced) sources are read from database."),
        default=1.
    )
    src_read_period = Field(
        dtype=int,
        doc=("Period for repating read/no-read cycles for (forced) sources."),
        default=1000
    )
    fill_empty_fields = Field(
        dtype=bool,
        doc="If True then store random values for fields not explicitly filled.",
        default=False)
    insert_id_keep_days = Field[int](
        doc=(
            "Number of days of insert_id history to keep during daily cleanups. "
            "Negative number disables cleanups."
        ),
        default=-1,
    )

    @property
    def FOV_rad(self) -> float:
        """FOV in radians.
        """
        return self.FOV_deg * math.pi / 180

    @property
    def start_time_dt(self) -> DateTime:
        """start_time as DateTime.
        """
        dt = DateTime(self.start_time, DateTime.TAI)
        return dt
