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

__all__ = ["APProto"]

import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
from collections.abc import Iterator
from typing import Any

import astropy.time
import numpy
from mpi4py import MPI

from lsst.dax.apdb import Apdb, monitor, timer
from lsst.sphgeom import Angle, Circle, LonLat, UnitVector3d

from . import DIA, L1dbprotoConfig, generators, geom
from ._executors import APProtoVisitExecutor, InMemoryExecutor, SubprocessExecutor
from ._logging import config_logger
from .visit_info import VisitInfoStore

COLOR_RED = "\033[1;31m"
COLOR_GREEN = "\033[1;32m"
COLOR_YELLOW = "\033[1;33m"
COLOR_BLUE = "\033[1;34m"
COLOR_MAGENTA = "\033[1;35m"
COLOR_CYAN = "\033[1;36m"
COLOR_RESET = "\033[0m"


_LOG = logging.getLogger("ap_proto")

_MON = monitor.MonAgent("ap_proto")


def _isDayTime(visit_time: astropy.time.Time) -> bool:
    """Return true if time is not good for observing."""
    return 6 <= visit_time.datetime.hour < 20


def _visitTimes(
    start_time: astropy.time.Time, interval: astropy.time.TimeDelta, count: int
) -> Iterator[astropy.time.Time]:
    """Generate visit times."""
    visit_time = start_time
    while count > 0:
        if not _isDayTime(visit_time):
            yield visit_time
            count -= 1
        visit_time += interval


# transient ID start value
_TRANSIENT_START_ID = 1000000000


class APProto:
    """Implementation of Alert Production prototype."""

    def __init__(self, argv: list[str]):
        self.lastObjectId = _TRANSIENT_START_ID

        descr = "Script which simulates AP workflow access to L1 database."
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
            "-n",
            "--num-visits",
            type=int,
            default=1,
            metavar="NUMBER",
            help="Number of visits to process, def: 1",
        )
        parser.add_argument(
            "-f",
            "--visits-file",
            default="ap_proto_visits.dat",
            metavar="PATH",
            help="File to keep visit information, def: %(default)s",
        )
        parser.add_argument(
            "-c",
            "--config",
            default=None,
            metavar="PATH",
            help="Name of the database config file (pex.config format)",
        )
        parser.add_argument(
            "-a",
            "--app-config",
            default=None,
            metavar="PATH",
            help="Name of the ap_proto config file (pex.config format)",
        )
        parser.add_argument(
            "-d",
            "--dump-config",
            default=False,
            action="store_true",
            help="Dump configuration to standard output and quit.",
        )
        parser.add_argument(
            "-U",
            "--no-update",
            default=False,
            action="store_true",
            help="DO not update database, only reading is performed.",
        )
        parser.add_argument(
            "-R",
            "--re-connect",
            default=False,
            action="store_true",
            help="Create new APDB connection on each visit.",
        )
        parser.add_argument(
            "--store-reconnect",
            default=False,
            action="store_true",
            help="Re-connect to Apdb before storing new data.",
        )
        parser.add_argument(
            "-S",
            "--sub-process",
            default=False,
            action="store_true",
            help="Run in sub-process mode, only when MPI.",
        )

        # parse options
        self.args = parser.parse_args(argv)

        # configure logging
        rank = MPI.COMM_WORLD.Get_rank() if MPI.COMM_WORLD.Get_size() > 1 else None
        config_logger(self.args.verbose, rank)

        self.config = L1dbprotoConfig()

    def _executor(
        self, db: Apdb | None, detector: int, n_detectors: int, tile: tuple[int, int] | None = None
    ) -> APProtoVisitExecutor:
        if self.args.sub_process:
            return SubprocessExecutor(
                self.args.app_config,
                self.args.config,
                no_update=self.args.no_update,
                verbose=self.args.verbose,
                detector=detector,
                n_detectors=n_detectors,
                tile=tile,
                rank=MPI.COMM_WORLD.Get_rank() if MPI.COMM_WORLD.Get_size() > 1 else None,
                store_reconnect=self.args.store_reconnect,
            )
        else:
            return InMemoryExecutor(
                self.config,
                db,
                self.args.config,
                no_update=self.args.no_update,
                detector=detector,
                n_detectors=n_detectors,
                store_reconnect=self.args.store_reconnect,
                tile=tile,
            )

    def run(self) -> int | None:
        """Run whole shebang."""
        random.seed()

        # load configurations
        if self.args.app_config:
            self.config.load(self.args.app_config)

        if self.args.dump_config:
            self.config.saveToStream(sys.stdout)
            return 0

        if self.config.mon_logger:
            mon_handler = monitor.LoggingMonHandler(self.config.mon_logger)
            monitor.MonService().add_handler(mon_handler)
        if self.config.mon_rules:
            rules = self.config.mon_rules.split(",")
            monitor.MonService().set_filters(rules)

        # instantiate db interface
        db: Apdb | None = None
        if not self.args.re_connect:
            db = Apdb.from_uri(self.args.config)

        visitInfoStore = VisitInfoStore(self.args.visits_file)

        num_tiles = 1
        if self.config.divide != 1:
            tiles = geom.make_tiles(self.config.FOV_rad, self.config.divide)
            num_tiles = len(tiles)

            # check that we have reasonable MPI setup
            if self.config.mp_mode == "mpi":
                comm = MPI.COMM_WORLD
                num_proc = comm.Get_size()
                rank = comm.Get_rank()
                node = MPI.Get_processor_name()
                _LOG.info(
                    COLOR_YELLOW + "MPI job rank=%d size=%d, node %s" + COLOR_RESET,
                    rank,
                    num_proc,
                    node,
                )
                if num_proc != num_tiles:
                    raise ValueError(
                        f"Number of MPI processes ({num_proc}) does not match number of tiles ({num_tiles})"
                    )
                if rank != 0:
                    # run simple loop for all non-master processes
                    self.run_mpi_tile_loop(db, comm)
                    return None
                else:
                    for k, v in sorted(os.environ.items()):
                        _LOG.debug("%s=%s", k, v)

        # Initialize starting values from database visits table
        last_visit = visitInfoStore.lastVisit()
        if last_visit is not None:
            start_visit_id = last_visit.visitId + 1
            start_time = last_visit.visitTime + self.config.interval_astropy
        else:
            start_visit_id = self.config.start_visit_id
            start_time = self.config.start_time_astropy

        if self.config.divide > 1:
            _LOG.info("Will divide FOV into %d regions", num_tiles)

        src_read_period = self.config.src_read_period
        src_read_visits = round(self.config.src_read_period * self.config.src_read_duty_cycle)
        _LOG.info(
            "Will read sources for %d visits out of %d",
            src_read_visits,
            src_read_period,
        )

        # read sources file
        _LOG.info("Start loading variable sources from %r", self.config.sources_file)
        var_sources = numpy.load(self.config.sources_file)
        _LOG.info("Finished loading variable sources, count = %s", len(var_sources))

        # diaObjectId for last new DIA object, for variable sources we use
        # their index as objectId, for transients we want to use ID outside
        # that range
        if last_visit is not None and last_visit.lastObjectId is not None:
            self.lastObjectId = max(self.lastObjectId, last_visit.lastObjectId)
        if self.lastObjectId < len(var_sources):
            _LOG.error("next object id is too low: %s", self.lastObjectId)
            return 1
        _LOG.debug("lastObjectId: %s", self.lastObjectId)

        # loop over visits
        visitTimes = _visitTimes(start_time, self.config.interval_astropy, self.args.num_visits)
        for visit_id, visit_time in enumerate(visitTimes, start_visit_id):
            with _MON.context_tags({"visit": visit_id}):
                _LOG.info(
                    COLOR_GREEN + "+++ Start processing visit %s at %s" + COLOR_RESET,
                    visit_id,
                    visit_time.isot,
                )
                loop_timer = timer.Timer("total_visit_time").start()

                with timer.Timer("DIA", _LOG):
                    # point telescope in random southern direction
                    pointing_xyz = generators.rand_sphere_xyz(1, -1)[0]
                    pointing_v = UnitVector3d(pointing_xyz[0], pointing_xyz[1], pointing_xyz[2])
                    ra = LonLat.longitudeOf(pointing_v).asDegrees()
                    dec = LonLat.latitudeOf(pointing_v).asDegrees()

                    # sphgeom.Circle opening angle is actually a half of
                    # opening angle
                    region = Circle(pointing_v, Angle(self.config.FOV_rad / 2))

                    _LOG.info("Pointing ra, dec = %s, %s; xyz = %s", ra, dec, pointing_xyz)

                    # Simulating difference image analysis
                    dia = DIA.DIA(
                        pointing_xyz,
                        self.config.FOV_rad,
                        var_sources,
                        self.config.false_per_visit + self.config.transient_per_visit,
                        self.config.detection_fraction,
                    )
                    sources, indices = dia.makeSources()
                    _LOG.info("DIA generated %s sources", len(sources))

                    # assign IDs to transients
                    for i in range(len(sources)):
                        if indices[i] < 0:
                            self.lastObjectId += 1
                            indices[i] = self.lastObjectId

                # numpy seems to do some multi-threaded stuff which "leaks" CPU
                # cycles to the code below and it gets counted as resource
                # usage in timers, add a short delay here so that threads
                # finish and don't influence our timers below.
                time.sleep(0.1)

                if self.config.divide == 1:
                    # do it in-process
                    with timer.Timer("visit_processing_time", _MON, _LOG):
                        detector = 0
                        n_detectors = 1
                        executor = self._executor(db, detector, n_detectors)
                        executor.visit(visit_id, visit_time, region, sources, indices)

                else:
                    if self.config.mp_mode == "fork":
                        tiles = geom.make_tiles(self.config.FOV_rad, self.config.divide, pointing_v)
                        n_detectors = len(tiles)

                        with timer.Timer("visit_processing_time", _MON, _LOG):
                            # spawn subprocesses to handle individual tiles
                            children = []
                            for detector, (ix, iy, region) in enumerate(tiles):
                                tile = (ix, iy)
                                tags = {"tile": f"{ix}x{iy}"}
                                with _MON.context_tags(tags):
                                    pid = os.fork()
                                    if pid == 0:
                                        # child
                                        executor = self._executor(db, detector, n_detectors, tile)
                                        executor.visit(
                                            visit_id,
                                            visit_time,
                                            region,
                                            sources,
                                            indices,
                                        )
                                        # stop here
                                        sys.exit(0)

                                    else:
                                        _LOG.debug("Forked process %d for tile %s", pid, tile)
                                        children.append(pid)

                                # wait until all children finish
                                for pid in children:
                                    try:
                                        pid, status = os.waitpid(pid, 0)
                                        if status != 0:
                                            _LOG.warning(
                                                COLOR_RED + "Child process PID=%s failed: %s" + COLOR_RESET,
                                                pid,
                                                status,
                                            )
                                    except OSError as exc:
                                        _LOG.warning(
                                            COLOR_RED + "wait failed for PID=%s: %s" + COLOR_RESET,
                                            pid,
                                            exc,
                                        )

                    elif self.config.mp_mode == "mpi":
                        tiles = geom.make_tiles(self.config.FOV_rad, self.config.divide, pointing_v)
                        n_detectors = len(tiles)
                        _LOG.info("Split FOV into %d tiles for MPI", len(tiles))

                        tile_data = []
                        for detector, (ix, iy, region) in enumerate(tiles):
                            tile = (ix, iy)
                            tile_data += [
                                (
                                    visit_id,
                                    visit_time,
                                    region,
                                    sources,
                                    indices,
                                    tile,
                                    detector,
                                    n_detectors,
                                )
                            ]

                        with timer.Timer("visit_processing_time", _MON, _LOG):
                            _LOG.info("Scatter sources to %d tile processes", len(tile_data))
                            self.run_mpi_tile(db, MPI.COMM_WORLD, tile_data)

                if not self.args.no_update:
                    # store last visit info
                    visitInfoStore.saveVisit(visit_id, visit_time, self.lastObjectId, 0)

                _LOG.info(
                    COLOR_BLUE + "--- Finished processing visit %s, time: %s" + COLOR_RESET,
                    visit_id,
                    loop_timer,
                )
                _MON.add_record("total_visit_time", values=loop_timer.as_dict(), tags={"visit": visit_id})

        # stop MPI slaves
        if num_tiles > 1 and self.config.mp_mode == "mpi":
            _LOG.info("Stopping MPI tile processes")
            tile_data_stop = [None] * num_tiles
            self.run_mpi_tile(db, MPI.COMM_WORLD, tile_data_stop)

        return 0

    def run_mpi_tile_loop(self, db: Apdb | None, comm: Any) -> None:
        """Execute visit loop inside non-master MPI process"""
        while self.run_mpi_tile(db, comm):
            pass

    def run_mpi_tile(self, db: Apdb | None, comm: Any, tile_data: Any = None) -> Any:
        """Execute single-visit processing in each MPI tile process.

        Parameters
        ----------
        comm : `MPI.Comm`
            MPI communicator
        tile_data : `list`, optional
            Data to scatter to tile processes, only used for single sending
            process (with rank=0). Size of the list must be equal to the
            number of processes in communicator. To signal the end of
            processing list should contain None for each element.

        Returns
        -------
        For rank=0 process this returns a list of data gathered from each
        tile process. For other tile processes it returns True. When
        `tile_data` contains all `None` then None is returned to all
        processes.
        """
        _LOG.debug(
            "MPI rank %d scatter with %r",
            comm.Get_rank(),
            None if tile_data is None else len(tile_data),
        )
        tile_data = comm.scatter(tile_data, root=0)
        if tile_data is None:
            # this signals stop running
            _LOG.debug("MPI rank %d scatter returned None", comm.Get_rank())
            return None
        else:
            (
                visit_id,
                visit_time,
                region,
                sources,
                indices,
                tile,
                detector,
                n_detectors,
            ) = tile_data
            _LOG.debug(
                "MPI rank %d scatter returned visit=%r tile=%r",
                comm.Get_rank(),
                visit_id,
                tile,
            )
            _LOG.info(
                COLOR_MAGENTA + "+++ Start processing visit %s tile %s at %s" + COLOR_RESET,
                visit_id,
                tile,
                visit_time,
            )
            tags = {"visit": visit_id, "rank": MPI.COMM_WORLD.Get_rank()}
            if tile is not None:
                tags["tile"] = "{}x{}".format(*tile)
            with _MON.context_tags(tags):
                with timer.Timer("tile_visit_time", _MON) as loop_timer:
                    try:
                        executor = self._executor(db, detector, n_detectors, tile)
                        executor.visit(visit_id, visit_time, region, sources, indices)
                    except Exception as exc:
                        _LOG.error("Exception in visit processing: %s", exc, exc_info=True)
                    _LOG.info(
                        COLOR_CYAN + "--- Finished processing visit %s tile %s, time: %s" + COLOR_RESET,
                        visit_id,
                        tile,
                        loop_timer,
                    )

            # TODO: send something more useful?
            _LOG.debug("Doing gather")
            data = comm.gather(True, root=0)
            _LOG.debug("Tile %s gather returned %r", tile, data)
            if data is None:
                # non-root
                return True
            else:
                # return gathered data to root
                return data
