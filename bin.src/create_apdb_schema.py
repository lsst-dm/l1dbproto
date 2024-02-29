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

"""Script to create prototype schema for L1 tables.

This is based initially on baseline schema from `cat` package, but it may
be modified as prototype evolves.
"""

from argparse import ArgumentParser
import logging
import sys

from lsst.dax.apdb import (Apdb, ApdbSqlConfig, ApdbCassandraConfig)


def _configLogger(verbosity):
    """ configure logging based on verbosity level """

    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)


def main():

    descr = 'Create schema for Prompt Products Database.'
    parser = ArgumentParser(description=descr)
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='More verbose output, can use several times.')
    parser.add_argument('--backend', default="sql", choices=["sql", "cassandra"],
                        help='Backend type, def: %(default)s')
    parser.add_argument('-d', '--dump-config', default=False, action="store_true",
                        help='Dump configuration to standard output and quit.')
    parser.add_argument('--drop', action='store_true', default=False,
                        help='Drop existing schema first, this will delete '
                        'all data in the tables, use with extreme caution')
    parser.add_argument('-c', '--config', default=None, metavar='PATH',
                        help='Name of the database config file (pex.config)')
    args = parser.parse_args()

    # configure logging
    _configLogger(args.verbose)

    if args.backend == "sql":

        config = ApdbSqlConfig()
        if args.config:
            config.load(args.config)
        if args.dump_config:
            config.saveToStream(sys.stdout)
            return 0

    elif args.backend == "cassandra":

        config = ApdbCassandraConfig()
        if args.config:
            config.load(args.config)
        if args.dump_config:
            config.saveToStream(sys.stdout)
            return 0

    Apdb.makeSchema(config, drop=args.drop)


#
#  run application when imported as a main module
#
if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
