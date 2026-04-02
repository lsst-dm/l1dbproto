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

__all__ = ["APProtoVisit"]

import random
from argparse import ArgumentParser

import astropy.time
import numpy

from lsst.dax.apdb import Apdb, monitor
from lsst.sphgeom import Region
from lsst.utils.threads import disable_implicit_threading

from . import L1dbprotoConfig
from ._executors import InMemoryExecutor
from ._logging import config_logger


class APProtoVisit:
    """Implementation of Alert Production prototype."""

    def __init__(self, argv: list[str]):
        descr = "Script which generates data for one visit."
        parser = ArgumentParser(description=descr)
        parser.add_argument(
            "-v",
            "--verbose",
            dest="verbose",
            action="count",
            default=0,
            help="More verbose output, can use several times.",
        )
        parser.add_argument(
            "-c",
            "--config",
            default=None,
            metavar="PATH",
            help="Name of the database config file (pex.config format)",
        )
        parser.add_argument(
            "-r",
            "--rank",
            default=None,
            type=int,
            help="MPI rank",
        )
        parser.add_argument(
            "-a",
            "--app-config",
            default=None,
            metavar="PATH",
            help="Name of the ap_proto config file (pex.config format)",
        )
        parser.add_argument(
            "-U",
            "--no-update",
            default=False,
            action="store_true",
            help="DO not update database, only reading is performed.",
        )
        parser.add_argument(
            "--store-reconnect",
            default=False,
            action="store_true",
            help="Re-connect to Apdb before storing new data.",
        )
        parser.add_argument("visit_id", type=int, help="Visit number.")
        parser.add_argument("visit_time", type=float, help="Visit time in MJD.")
        parser.add_argument("region", type=str, help="Encoded sphgeom region.")
        parser.add_argument("sources", type=str, help="File with ndarray of source data.")
        parser.add_argument("detector", type=int, help="Detector number.")
        parser.add_argument("n_detectors", type=int, help="Total detector count.")
        parser.add_argument("tile", type=str, nargs="?", help="Tile, in format NxM, optional.")

        # parse options
        self.args = parser.parse_args(argv)

        # configure logging
        config_logger(self.args.verbose, self.args.rank)

        self.config = L1dbprotoConfig()

        disable_implicit_threading()

    def run(self) -> int | None:
        """Run whole shebang."""
        random.seed()

        # load configurations
        if self.args.app_config:
            self.config.load(self.args.app_config)

        if self.config.mon_logger:
            mon_handler = monitor.LoggingMonHandler(self.config.mon_logger)
            monitor.MonService().add_handler(mon_handler)
        if self.config.mon_rules:
            rules = self.config.mon_rules.split(",")
            monitor.MonService().set_filters(rules)

        # instantiate db interface
        db = Apdb.from_uri(self.args.config)

        visit_id = self.args.visit_id
        visit_time = astropy.time.Time(self.args.visit_time, format="mjd", scale="tai")
        region = Region.decodeBase64(self.args.region)
        with numpy.load(self.args.sources) as data:
            sources = data["sources"]
            indices = data["indices"]
        tile = None
        if self.args.tile:
            x, _, y = self.args.tile.partition("x")
            tile = (int(x), int(y))

        executor = InMemoryExecutor(
            self.config,
            db,
            self.args.config,
            self.args.no_update,
            self.args.detector,
            self.args.n_detectors,
            self.args.store_reconnect,
            tile,
        )
        executor.visit(visit_id, visit_time, region, sources, indices)

        return 0
