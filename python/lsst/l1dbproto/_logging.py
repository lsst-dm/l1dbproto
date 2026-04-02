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

"""Application which simulates AP workflow access to L1 database.

It generates approximately realistic result of difference image analysis,
source-to-object matching, and forced photometry and stores all that in
a database.
"""

__all__ = ["config_logger"]

import logging
from typing import Any


def config_logger(verbosity: int, mpi_rank: int | None = None) -> None:
    """Configure logging based on verbosity level."""
    if mpi_rank is not None:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            """Make logging record that adds MPI rank to a record."""
            record = old_factory(*args, **kwargs)
            record.mpi_rank = mpi_rank
            return record

        logging.setLogRecordFactory(record_factory)
        logfmt = "%(asctime)s [%(levelname)s] [rank=%(mpi_rank)03d] %(name)s: %(message)s"
    else:
        logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)
    logging.getLogger("cassandra").setLevel(logging.INFO)
