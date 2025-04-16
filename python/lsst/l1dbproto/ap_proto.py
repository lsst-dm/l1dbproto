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
import string
import sys
import time
from argparse import ArgumentParser
from collections.abc import Iterator
from datetime import timedelta
from typing import Any, cast

import astropy.time
import felis.datamodel
import numpy
import numpy.random
import pandas
from lsst.dax.apdb import Apdb, ApdbReplica, ApdbTables, monitor, timer
from lsst.geom import SpherePoint
from lsst.sphgeom import Angle, Circle, LonLat, Region, UnitVector3d, Vector3d
from lsst.utils.iteration import chunk_iterable
from mpi4py import MPI

from . import DIA, L1dbprotoConfig, generators, geom
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


def _configLogger(verbosity: int) -> None:
    """
    Configure logging based on verbosity level.
    """
    if MPI.COMM_WORLD.Get_size() > 1:
        rank = MPI.COMM_WORLD.Get_rank()
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            """Make logging record that adds MPI rank to a record."""
            record = old_factory(*args, **kwargs)
            record.mpi_rank = rank
            return record

        logging.setLogRecordFactory(record_factory)
        logfmt = "%(asctime)s [%(levelname)s] [rank=%(mpi_rank)03d] %(name)s: %(message)s"
    else:
        logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)
    # logging.getLogger("cassandra").setLevel(logging.INFO)


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


def _nrows(table: pandas.DataFrame | None) -> int:
    if table is None:
        return 0
    else:
        return len(table)


# special code to mark sources outside region
_OUTSIDER = -666

# transient ID start value
_TRANSIENT_START_ID = 1000000000


class APProto:
    """Implementation of Alert Production prototype."""

    def __init__(self, argv: list[str]):
        self.lastObjectId = _TRANSIENT_START_ID
        self.lastSourceId = 0

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

        # parse options
        self.args = parser.parse_args(argv)

        # configure logging
        _configLogger(self.args.verbose)

        self.config = L1dbprotoConfig()

    def run(self) -> int | None:
        """Run whole shebang."""
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

        # Initialize starting values from database visits table
        last_visit = visitInfoStore.lastVisit()
        prev_visit_time: astropy.time.Time | None = None
        if last_visit is not None:
            start_visit_id = last_visit.visitId + 1
            prev_visit_time = last_visit.visitTime
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

        # diaSourceId for last DIA source stored in database
        if last_visit is not None and last_visit.lastSourceId is not None:
            self.lastSourceId = max(self.lastSourceId, last_visit.lastSourceId)
        _LOG.info("lastSourceId: %s", self.lastSourceId)

        # loop over visits
        visitTimes = _visitTimes(start_time, self.config.interval_astropy, self.args.num_visits)
        for visit_id, visit_time in enumerate(visitTimes, start_visit_id):
            with _MON.context_tags({"visit": visit_id}):
                if prev_visit_time is not None:
                    delta_to_prev = visit_time - prev_visit_time
                    # If delta to previous is much longer than interval means
                    # we just skipped day time.
                    if delta_to_prev > self.config.interval_astropy * 100:
                        midday = prev_visit_time + delta_to_prev / 2
                        _LOG.info(
                            COLOR_YELLOW + "+++ Start daily activities at %s" + COLOR_RESET,
                            midday.isot,
                        )
                        db.dailyJob()
                        self._daily_insert_id_cleanup(self.args.config, midday)
                        _LOG.info(COLOR_YELLOW + "+++ Done with daily activities" + COLOR_RESET)

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
                    )
                    sources, indices = dia.makeSources()
                    _LOG.info("DIA generated %s sources", len(sources))

                    # assign IDs to transients
                    for i in range(len(sources)):
                        if indices[i] < 0:
                            self.lastObjectId += 1
                            indices[i] = self.lastObjectId

                # print current database row counts, this takes long time
                # so only do it once in a while
                modu = 200 if visit_id <= 10000 else 1000
                if visit_id % modu == 0:
                    if hasattr(db, "tableRowCount"):
                        counts = db.tableRowCount()
                        for tbl, count in sorted(counts.items()):
                            _LOG.info("%s row count: %s", tbl, count)

                # numpy seems to do some multi-threaded stuff which "leaks" CPU
                # cycles to the code below and it gets counted as resource
                # usage in timers, add a short delay here so that threads
                # finish and don't influence our timers below.
                time.sleep(0.1)

                if self.config.divide == 1:
                    # do it in-process
                    with timer.Timer("visit_processing_time", _MON, _LOG):
                        self.visit(db, visit_id, visit_time, region, sources, indices)

                else:
                    if self.config.mp_mode == "fork":
                        tiles = geom.make_tiles(self.config.FOV_rad, self.config.divide, pointing_v)

                        with timer.Timer("visit_processing_time", _MON, _LOG):
                            # spawn subprocesses to handle individual tiles
                            children = []
                            for ix, iy, region in tiles:
                                # make sure lastSourceId is unique in in each
                                # process
                                self.lastSourceId += len(sources)
                                tile = (ix, iy)
                                tags = {"tile": f"{ix}x{iy}"}
                                with _MON.context_tags(tags):
                                    pid = os.fork()
                                    if pid == 0:
                                        # child

                                        self.visit(
                                            db,
                                            visit_id,
                                            visit_time,
                                            region,
                                            sources,
                                            indices,
                                            tile,
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
                        _LOG.info("Split FOV into %d tiles for MPI", len(tiles))

                        # spawn subprocesses to handle individual tiles,
                        # special care needed for self.lastSourceId because
                        # it's propagated back from (0, 0)
                        lastSourceId = self.lastSourceId
                        tile_data = []
                        for ix, iy, region in tiles:
                            lastSourceId += len(sources)
                            tile = (ix, iy)
                            tile_data += [
                                (
                                    visit_id,
                                    visit_time,
                                    region,
                                    sources,
                                    indices,
                                    tile,
                                    lastSourceId,
                                )
                            ]
                            # make sure lastSourceId is unique in in each
                            # process

                        with timer.Timer("visit_processing_time", _MON, _LOG):
                            _LOG.info("Scatter sources to %d tile processes", len(tile_data))
                            self.run_mpi_tile(db, MPI.COMM_WORLD, tile_data)
                        self.lastSourceId = lastSourceId

                if not self.args.no_update:
                    # store last visit info
                    visitInfoStore.saveVisit(visit_id, visit_time, self.lastObjectId, self.lastSourceId)

                prev_visit_time = visit_time

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

    def run_mpi_tile_loop(self, db: Apdb, comm: Any) -> None:
        """Execute visit loop inside non-master MPI process"""
        while self.run_mpi_tile(db, comm):
            pass

    def run_mpi_tile(self, db: Apdb, comm: Any, tile_data: Any = None) -> Any:
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
                lastSourceId,
            ) = tile_data
            _LOG.debug(
                "MPI rank %d scatter returned visit=%r tile=%r",
                comm.Get_rank(),
                visit_id,
                tile,
            )
            self.lastSourceId = lastSourceId
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
                        self.visit(db, visit_id, visit_time, region, sources, indices, tile)
                    except Exception as exc:
                        _LOG.error("Exception in visit processing: %s", exc, exc_info=True)
                    _LOG.info(
                        COLOR_CYAN + "--- Finished processing visit %s tile %s, time: %s" + COLOR_RESET,
                        visit_id,
                        tile,
                        loop_timer,
                    )

            # TODO: send something more useful?
            data = comm.gather(True, root=0)
            _LOG.debug("Tile %s gather returned %r", tile, data)
            if data is None:
                # non-root
                return True
            else:
                # return gathered data to root
                return data

    def visit(
        self,
        db: Apdb,
        visit_id: int,
        visit_time: astropy.time.Time,
        region: Region,
        sources: numpy.ndarray,
        indices: numpy.ndarray,
        tile: tuple[int, int] | None = None,
    ) -> None:
        """AP processing of a single visit (with known sources)

        Parameters
        ----------
        db : `Apdb`
            APDB interface
        visit_id : `int`
            Visit ID.
        visit_time : `astropy.time.Time`
            Time of visit
        region : `sphgeom.Region`
            Region, could be the whole FOV (Circle) or small piece of it
        sources : `numpy.array`
            Array of xyz coordinates of sources, this has all visit sources,
            not only current tile
        indices : `numpy.array`
            array of indices of sources, 1-dim ndarray, transient sources
            have negative indices
        tile : `tuple`
            tile position (x, y)
        """
        name = ""
        detector = 0
        if tile is not None:
            name = "tile={}x{} ".format(*tile)
            detector = tile[0] * 100 + tile[1]

        src_read_period = self.config.src_read_period
        src_read_visits = round(self.config.src_read_period * self.config.src_read_duty_cycle)
        do_read_src = visit_id % src_read_period < src_read_visits

        # make a mask
        for i in range(len(sources)):
            xyz = sources[i]
            if not region.contains(UnitVector3d(xyz[0], xyz[1], xyz[2])):
                indices[i] = _OUTSIDER

        counts: dict[str, int] = {}

        with timer.Timer(name + "Objects-read", _LOG):
            # Retrieve DiaObjects (latest versions) from database for matching,
            # this will produce wider coverage so further filtering is needed
            latest_objects = db.getDiaObjects(region)
            _LOG.info(name + "database found %s objects", _nrows(latest_objects))
            counts["objects"] = _nrows(latest_objects)

            # filter database objects to a mask
            latest_objects = self._filterDiaObjects(latest_objects, region)
            _LOG.info(name + "after filtering %s objects", _nrows(latest_objects))
            counts["objects_filtered"] = _nrows(latest_objects)

        with timer.Timer(name + "S2O-matching", _LOG):
            # make all sources
            srcs = self._makeDiaSources(sources, indices, visit_time, visit_id, detector)

            # create all new DiaObjects
            objects = self._makeDiaObjects(sources, indices, latest_objects, visit_time)

            # do forced photometry (can extends objects)
            fsrcs = self._forcedPhotometry(objects, visit_time, visit_id, detector)

            objects = self._fillRandomData(objects, ApdbTables.DiaObject, db)
            srcs = self._fillRandomData(srcs, ApdbTables.DiaSource, db)
            fsrcs = self._fillRandomData(fsrcs, ApdbTables.DiaForcedSource, db)

        if do_read_src:
            with timer.Timer(name + "Source-read", _LOG):
                latest_objects_ids = list(latest_objects["diaObjectId"])

                read_srcs = db.getDiaSources(region, latest_objects_ids, visit_time)
                _LOG.info(name + "database found %s sources", _nrows(read_srcs))
                counts["sources"] = _nrows(read_srcs)

                read_srcs = db.getDiaForcedSources(region, latest_objects_ids, visit_time)
                _LOG.info(name + "database found %s forced sources", _nrows(read_srcs))
                counts["forcedsources"] = _nrows(read_srcs)
        else:
            _LOG.info("skipping reading of sources for this visit")

        _MON.add_record("read_counts", values=counts)

        if not self.args.no_update:
            with timer.Timer("tile_store_time", _MON, _LOG):
                # store new versions of objects
                _LOG.info(name + "will store %d Objects", len(objects))
                _LOG.info(name + "will store %d Sources", len(srcs))
                _LOG.info(name + "will store %d ForcedSources", len(fsrcs))
                db.store(visit_time, objects, srcs, fsrcs)
                counts = {
                    "objects": len(objects),
                    "sources": len(srcs),
                    "forcedsources": len(fsrcs),
                }
                _MON.add_record("store_counts", values=counts)

    def _filterDiaObjects(self, latest_objects: pandas.DataFrame, region: Region) -> pandas.DataFrame:
        """Filter out objects from a catalog which are outside region.

        Parameters
        ----------
        latest_objects : `pandas.DataFrame`
            Catalog containing DiaObject records
        region : `sphgeom.Region`

        Returns
        -------
        Filtered `pandas.DataFrame` containing only records contained
        in the region.
        """
        if latest_objects.empty:
            return latest_objects

        def in_region(obj: Any) -> bool:
            lonLat = LonLat.fromDegrees(obj["ra"], obj["dec"])
            dir_obj = UnitVector3d(lonLat)
            return region.contains(dir_obj)

        mask = latest_objects.apply(in_region, axis=1, result_type="reduce")
        return cast(pandas.DataFrame, latest_objects[mask])

    def _makeDiaObjects(
        self,
        sources: numpy.ndarray,
        indices: numpy.ndarray,
        known_objects: pandas.DataFrame,
        visit_time: astropy.time.Time,
    ) -> pandas.DataFrame:
        """Over-simplified implementation of source-to-object matching and
        new DiaObject generation.

        Currently matching is based on info passed along by source
        generator and does not even use DiaObjects from database (meaning that
        matching is 100% perfect).

        Parameters
        ----------
        sources : `numpy.array`
            (x, y, z) coordinates of sources, array dimension is (N, 3)
        indices : `numpy.array`
            array of indices of sources, 1-dim ndarray, transient sources
            have negative indices
        known_objects : `pandas.DataFrame`
            Catalog of DiaObjects read from APDB.
        visit_time : `astropy.time.Time`
            Visit time.

        Returns
        -------
        catalog : `pandas.DataFrame`
            Catalog of DiaObjects.
        """

        def polar(row: Any) -> pandas.Series:
            v3d = Vector3d(row.x, row.y, row.z)
            sp = SpherePoint(v3d)
            return pandas.Series([sp.getRa().asDegrees(), sp.getDec().asDegrees()], index=["ra", "dec"])

        catalog = pandas.DataFrame(sources, columns=["x", "y", "z"])
        catalog["diaObjectId"] = indices
        catalog = cast(pandas.DataFrame, catalog[catalog["diaObjectId"] != _OUTSIDER])

        if len(catalog) == 0:
            return pandas.DataFrame(
                columns=["ra", "dec", "diaObjectId", "nDiaSources", "lastNonForcedSource"]
            )

        cat_polar = cast(pandas.DataFrame, catalog.apply(polar, axis=1, result_type="expand"))
        cat_polar["diaObjectId"] = catalog["diaObjectId"]
        catalog = cat_polar

        # Set nDiaSources for each object, update from existing objects.
        # Could do it with some pandas magic, but it's insane.
        count_map = {
            obj_id: count
            for obj_id, count in known_objects[["diaObjectId", "nDiaSources"]].itertuples(index=False)
        }

        def _count_sources(row: Any) -> pandas.Series:
            count = count_map.get(row.diaObjectId, 0) + 1
            return pandas.Series([count], index=["nDiaSources"])

        catalog["nDiaSources"] = catalog.apply(_count_sources, axis=1, result_type="expand")

        catalog["lastNonForcedSource"] = visit_time.datetime

        n_trans = sum(catalog["diaObjectId"] >= _TRANSIENT_START_ID)
        _LOG.info("found %s matching objects and %s transients/noise", _nrows(catalog) - n_trans, n_trans)

        return catalog

    def _forcedPhotometry(
        self,
        objects: pandas.DataFrame,
        visit_time: astropy.time.Time,
        visit_id: int,
        detector: int,
    ) -> pandas.DataFrame:
        """Do forced photometry on latest_objects which are not in objects.

        Extends objects catalog with new DiaObjects.

        Parameters
        ----------
        objects : `pandas.DataFrame`
            Catalog containing DiaObject records
        visit_time : `astropy.time.Time`
            Visit time.
        visit_id : `int`
            Visit ID.
        """
        midpointMjdTai = visit_time.tai.mjd

        # Do forced photometry on objects with nDiaSources > 1, and only
        # for 30 days after last detection
        objects = cast(pandas.DataFrame, objects[objects["nDiaSources"] > 1])
        cutoff = visit_time.datetime - timedelta(days=self.config.forced_cutoff_days)
        objects = cast(pandas.DataFrame, objects[objects["lastNonForcedSource"] > cutoff])

        if objects.empty:
            return pandas.DataFrame(columns=["diaObjectId", "visit", "detector", "midpointMjdTai"])

        catalog = pandas.DataFrame(
            {
                "diaObjectId": objects["diaObjectId"],
                "visit": visit_id,
                "detector": detector,
                "midpointMjdTai": midpointMjdTai,
            }
        )

        return catalog

    def _makeDiaSources(
        self,
        sources: numpy.ndarray,
        indices: numpy.ndarray,
        visit_time: astropy.time.Time,
        visit_id: int,
        detector: int,
    ) -> pandas.DataFrame:
        """Generate catalog of DiaSources to store in a database

        Parameters
        ----------
        sources : `numpy.ndarray`
            (x, y, z) coordinates of sources, array dimension is (N, 3)
        indices : `numpy.array`
            array of indices of sources, 1-dim ndarray, transient sources
            have negative indices
        visit_time : `astropy.time.Time`
            Visit time.
        visit_id : `int`
            ID of the visit

        Returns
        -------
        catalog : `pandas.DataFrame`
            Catalog of DIASources.
        """

        def polar(row: Any) -> pandas.Series:
            v3d = Vector3d(row.x, row.y, row.z)
            sp = SpherePoint(v3d)
            return pandas.Series([sp.getRa().asDegrees(), sp.getDec().asDegrees()], index=["ra", "dec"])

        midpointMjdTai = visit_time.tai.mjd

        catalog = pandas.DataFrame(sources, columns=["x", "y", "z"])
        catalog["diaObjectId"] = indices
        catalog = cast(pandas.DataFrame, catalog[catalog["diaObjectId"] != _OUTSIDER])

        if len(catalog) == 0:
            cat_polar = pandas.DataFrame([], columns=["ra", "dec", "diaObjectId"])
        else:
            cat_polar = cast(pandas.DataFrame, catalog.apply(polar, axis=1, result_type="expand"))
        cat_polar["diaObjectId"] = catalog["diaObjectId"]
        catalog = cat_polar
        catalog["visit"] = visit_id
        catalog["detector"] = detector
        catalog["parentDiaSourceId"] = 0
        catalog["psFlux"] = 1.0
        catalog["psFluxErr"] = 0.01
        catalog["midpointMjdTai"] = midpointMjdTai

        nrows = catalog.shape[0]
        catalog["diaSourceId"] = range(self.lastSourceId + 1, self.lastSourceId + 1 + nrows)
        self.lastSourceId += nrows

        return catalog

    def _fillRandomData(self, catalog: pandas.DataFrame, table: ApdbTables, db: Apdb) -> pandas.DataFrame:
        """Add missing fields to a catalog and fill it with random numbers.

        Parameters
        ----------
        catalog : `pandas.DataFrame`
            Catalog to extend and fill.
        table : `ApdbTables`
            Table type.
        db : `Apdb`
            APDB interface
        """
        rng = numpy.random.default_rng()
        table_def = db.tableDef(table)
        if table_def is None:
            return catalog
        count = len(catalog)
        if count == 0:
            return catalog
        columns = []
        for colDef in table_def.columns:
            if table is ApdbTables.DiaObject and colDef.name in (
                "validityStart",
                "validityEnd",
            ):
                continue
            if colDef.name == "pixelId":
                continue
            if colDef.nullable and not self.config.fill_empty_fields:
                # only fill non-null columns in this mode
                continue
            if colDef.name not in catalog.columns:
                # need to make a new column
                data: Any
                if colDef.datatype is felis.datamodel.DataType.float:
                    data = rng.random(count, dtype=numpy.float32)
                elif colDef.datatype is felis.datamodel.DataType.double:
                    data = rng.random(count, dtype=numpy.float64)
                elif colDef.datatype is felis.datamodel.DataType.int:
                    data = rng.integers(0, 1000, count, dtype=numpy.int32)
                elif colDef.datatype is felis.datamodel.DataType.long:
                    data = rng.integers(0, 1000, count, dtype=numpy.int64)
                elif colDef.datatype is felis.datamodel.DataType.short:
                    data = rng.integers(0, 1000, count, dtype=numpy.int16)
                elif colDef.datatype is felis.datamodel.DataType.byte:
                    data = rng.integers(0, 255, count, dtype=numpy.int8)
                elif colDef.datatype is felis.datamodel.DataType.boolean:
                    data = rng.integers(0, 1, count, dtype=numpy.int8)
                elif colDef.datatype is felis.datamodel.DataType.binary:
                    data = [rng.bytes(colDef.length or 3) for i in range(count)]
                elif colDef.datatype in (
                    felis.datamodel.DataType.char,
                    felis.datamodel.DataType.string,
                    felis.datamodel.DataType.unicode,
                    felis.datamodel.DataType.text,
                ):
                    chars = string.ascii_letters + string.digits
                    random_strings = []
                    for i in range(count):
                        indices = rng.integers(0, len(chars), colDef.length, dtype=numpy.int16)
                        random_strings.append("".join([chars[idx] for idx in indices]))
                    data = random_strings
                elif colDef.datatype is felis.datamodel.DataType.timestamp:
                    data = rng.integers(1500000000, 1600000000, count, dtype=numpy.int64)
                    data = numpy.array(data, dtype="datetime64[s]")
                else:
                    data = rng.random(count)
                series = pandas.Series(data, name=colDef.name, index=catalog.index)
                columns.append(series)
        if columns:
            catalog = pandas.concat([catalog] + columns, axis="columns")
        return catalog

    def _daily_insert_id_cleanup(self, config_uri: str, visit_time: astropy.time.Time) -> None:
        """Remove old data from all InsertId tables.

        Parameters
        ----------
        config_uri : `str`
            Path to config file for Apdb.
        visit_time : `astropy.time.Time`
            Time of next visit.
        """
        cleanup_days = self.config.replica_chunk_keep_days
        if cleanup_days < 0:
            # no cleanup
            return

        replica = ApdbReplica.from_uri(config_uri)
        chunks = replica.getReplicaChunks()
        if not chunks:
            return

        # Find latest InsertId. InsertIds should be ordered, but it's not
        # guaranteed.
        latest_time = max(chunk.last_update_time for chunk in chunks)
        drop_time = latest_time - astropy.time.TimeDelta(self.config.replica_chunk_keep_days, format="jd")
        chunks_to_remove = [chunk for chunk in chunks if chunk.last_update_time < drop_time]
        _LOG.info(COLOR_YELLOW + "Will remove %d inserts" + COLOR_RESET, len(chunks_to_remove))
        for to_remove in chunk_iterable(chunks_to_remove, 10_000):
            try:
                replica.deleteReplicaChunks(to_remove)
            except Exception as exc:
                _LOG.error(
                    COLOR_RED + "Error while removing next chunk of inserts: %s" + COLOR_RESET,
                    exc,
                )
