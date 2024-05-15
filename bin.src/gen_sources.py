#!/bin/env python

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

"""Application to generate a list of random positions for sources.

Generates and saves the list of sky coordinates for the sources to
be used by other applications.
"""

import logging
import math
from argparse import ArgumentParser

import numpy
from lsst.l1dbproto import generators


def _configLogger(verbosity: int) -> None:
    """Configure logging based on verbosity level"""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)


FOV = 3.5  # degrees


def main() -> None:
    """Generate sources based on command line arguments."""
    descr = "One-line application description."
    parser = ArgumentParser(description=descr)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="More verbose output, can use several times."
    )
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        default="xyz",
        help='Defines type of output data, possible values are "xyz", def: "xyz"',
    )
    parser.add_argument(
        "-H",
        "--hemi",
        type=int,
        default=0,
        help="Zero for whole sky, negative for southern hemisphere, " "positive for northern, def: 0",
    )
    parser.add_argument(
        "-n", "--counts", type=int, default=10000, help="Number of sources per visit, def: 10000"
    )
    parser.add_argument("file", help="Name of output file")
    args = parser.parse_args()

    # configure logging
    _configLogger(args.verbose)

    # total number of sources
    area = 2 * math.pi if args.hemi != 0 else 4 * math.pi
    visit_area = math.pi * (FOV / 2 * math.pi / 180) ** 2
    counts = int(args.counts * area / visit_area)
    logging.info("Total sources: %d", counts)

    points = generators.rand_sphere_xyz(counts, args.hemi)
    numpy.save(args.file, points)


#
#  run application when imported as a main module
#
if __name__ == "__main__":
    main()
