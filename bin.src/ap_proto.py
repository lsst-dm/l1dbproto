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

"""Script which simulates AP workflow access to L1 database.

It generates approximately realistic result of difference image analysis,
source-to-object matching, and forced photometry and stores all that in
a database.
"""

from argparse import ArgumentParser
from datetime import datetime, timedelta
import logging
import os
import sys
import time

import numpy
import lsst.afw.table as afwTable
from lsst.geom import SpherePoint
from lsst.l1dbproto import constants, DIA, generators, geom
from lsst.dax.ppdb import (Ppdb, PpdbConfig, make_minimal_dia_object_schema,
                           make_minimal_dia_source_schema, timer)
from lsst.sphgeom import Angle, Circle, HtmPixelization, LonLat, UnitVector3d, Vector3d


COLOR_RED = '\033[1;31m'
COLOR_GREEN = '\033[1;32m'
COLOR_YELLOW = '\033[1;33m'
COLOR_BLUE = '\033[1;34m'
COLOR_MAGENTA = '\033[1;35m'
COLOR_CYAN = '\033[1;36m'
COLOR_RESET = '\033[0m'


_LOG = logging.getLogger('ap_proto')


def _configLogger(verbosity):
    """
    Configure logging based on verbosity level.
    """

    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)


_dt_format = ('%Y-%m-%d %H:%M:%S', 'YYYY-MM-DD hh:mm:ss')


def _timeParse(dt_str):
    """
    Convert string to datetime object.
    """
    dt = datetime.strptime(dt_str, _dt_format[0])
    return dt


def _isDayTime(dt):
    """
    Returns true if time is not good for observing.
    """
    return 6 <= dt.hour < 20


def _visitTimes(start_time, interval_sec, count):
    """
    Generator for visit times.
    """
    dt = start_time
    delta = timedelta(seconds=interval_sec)
    while count > 0:
        if not _isDayTime(dt):
            yield dt
            count -= 1
        dt += delta


def _htm_indices(region):
    """Generate a set of HTM indices covering specified field of view.

    Parameters
    ----------
    region: `sphgeom.Region`
        Region that needs to be indexed

    Returns
    -------
    Sequence of ranges, range is a tuple (minHtmID, maxHtmID).
    """

    _LOG.debug('region: %s', region)
    pixelator = HtmPixelization(constants.HTM_LEVEL)
    indices = pixelator.envelope(region, constants.HTM_MAX_RANGES)
    for irange in indices.ranges():
        _LOG.debug('range: %s %s', pixelator.toString(irange[0]),
                   pixelator.toString(irange[1]))

    return indices.ranges()

# def _utc_seconds(dt):
#     """Convert datetime to POSIX seconds
#     """
#     return int((dt - datetime.utcfromtimestamp(0)).total_seconds())


# special code to makr sources ouside reagion
_OUTSIDER = -666

# transient ID start value
_TRANSIENT_START_ID = 1000000000


class APProto(object):
    """Implementation of Alert Production prototype.
    """

    def __init__(self):

        self.lastObjectId = _TRANSIENT_START_ID
        self.lastSourceId = 0

        descr = 'Script which simulates AP workflow access to L1 database.'
        parser = ArgumentParser(description=descr)
        parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0,
                            help='More verbose output, can use several times.')
        parser.add_argument('-n', '--num-visits', type=int, default=1, metavar='NUMBER',
                            help='Numer of visits to process, def: 1')
        parser.add_argument('-i', '--interval', type=int, default=45, metavar='SECONDS',
                            help='Interval between visits in seconds, def: 45')
        parser.add_argument('-c', '--config', default=None, metavar='PATH',
                            help='Name of the database config file (pex.config)')
        parser.add_argument('-d', '--dump-config', default=False, action="store_true",
                            help='Dump configuration to standard output and quit.')
        parser.add_argument('-U', '--no-update', default=False, action='store_true',
                            help='DO not update database, only reading is performed.')
        parser.add_argument('--start-time', type=_timeParse, default=None,
                            help='Starting time, format: ' + _dt_format[1] +
                            '. Time is assumed to be in UTC time zone. Used only at'
                            ' first invocation to intialize database.')
        parser.add_argument('--start-visit-id', type=int, default=1,
                            help='Starting visit ID. Used only at first invocation'
                            ' to intialize database. def: 1')
        parser.add_argument('--divide', type=int, default=0, metavar='NUM',
                            help='Divide FOV into NUM*NUM tiles for parallel processing')
        parser.add_argument('--sources-region', default=False, action='store_true',
                            help='Use region-based select for DiaSource')

        parser.add_argument('sources_file', help='Name of input file with sources')
        # add more command line options
        self.args = parser.parse_args()

        # configure logging
        _configLogger(self.args.verbose)

    def run(self):
        """Run whole shebang.
        """

        config = PpdbConfig()
        if self.args.config:
            config.load(self.args.config)

        if self.args.dump_config:
            config.saveToStream(sys.stdout)
            return 0

        # instantiate db interface
        db = Ppdb(config)

        # Initialize starting values from database visits table
        last_visit = db.lastVisit()
        if last_visit is None:
            # database is empty
            if self.args.start_time is None:
                _LOG.error('Database is not initialized, --start-time option is required')
                return 1
        else:
            self.args.start_visit_id = last_visit.visitId + 1
            self.args.start_time = last_visit.visitTime + timedelta(seconds=self.args.interval)

        if self.args.divide > 1:
            _LOG.info("Will divide FOV into %dx%d regions", self.args.divide, self.args.divide)
        _LOG.info("Max. number of ranges for pixelator: %d", constants.HTM_MAX_RANGES)

        # read sources file
        _LOG.info("Start loading variable sources from %r", self.args.sources_file)
        var_sources = numpy.load(self.args.sources_file)
        _LOG.info("Finished loading variable sources, count = %s", len(var_sources))

        # diaObjectId for last new DIA object, for variable sources we use their
        # index as objectId, for transients we want to use ID outside that range
        if last_visit is not None and last_visit.lastObjectId is not None:
            self.lastObjectId = max(self.lastObjectId, last_visit.lastObjectId)
        if self.lastObjectId < len(var_sources):
            _LOG.error('next object id is too low: %s', self.lastObjectId)
            return 1
        _LOG.debug("lastObjectId: %s", self.lastObjectId)

        # diaSourceId for last DIA source stored in database
        if last_visit is not None and last_visit.lastSourceId is not None:
            self.lastSourceId = max(self.lastSourceId, last_visit.lastSourceId)
        _LOG.info("lastSourceId: %s", self.lastSourceId)

        # loop over visits
        visitTimes = _visitTimes(self.args.start_time, self.args.interval, self.args.num_visits)
        for visit_id, dt in enumerate(visitTimes, self.args.start_visit_id):

            if visit_id % 1000 == 0:
                _LOG.info(COLOR_YELLOW + "+++ Start daily activities" + COLOR_RESET)
                db.dailyJob()
                _LOG.info(COLOR_YELLOW + "+++ Done with daily activities" + COLOR_RESET)

            _LOG.info(COLOR_GREEN + "+++ Start processing visit %s at %s" + COLOR_RESET, visit_id, dt)
            loop_timer = timer.Timer().start()

            with timer.Timer("DIA"):
                # point telescope in random southern direction
                pointing_xyz = generators.rand_sphere_xyz(1, -1)[0]
                pointing_v = UnitVector3d(pointing_xyz[0], pointing_xyz[1], pointing_xyz[2])
                ra = LonLat.longitudeOf(pointing_v).asDegrees()
                decl = LonLat.latitudeOf(pointing_v).asDegrees()

                # sphgeom.Circle opening angle is actually a half of opening angle
                region = Circle(pointing_v, Angle(constants.FOV_rad/2))

                _LOG.info("Pointing ra, decl = %s, %s; xyz = %s", ra, decl, pointing_xyz)

                # Simulating difference image analysis
                dia = DIA.DIA(pointing_xyz, constants.FOV_rad, var_sources,
                              constants.N_FALSE_PER_VISIT + constants.N_TRANS_PER_VISIT)
                sources, indices = dia.makeSources()
                _LOG.info("DIA generated %s sources", len(sources))

                # assign IDs to transients
                for i in range(len(sources)):
                    if indices[i] < 0:
                        self.lastObjectId += 1
                        indices[i] = self.lastObjectId

            # print current database row counts, this takes long time
            # so only do it once in a while
            if visit_id % 50 == 0:
                counts = db.tableRowCount()
                for tbl, count in sorted(counts.items()):
                    _LOG.info('%s row count: %s', tbl, count)

            # numpy seems to do some multi-threaded stuff which "leaks" CPU cycles to the code below
            # and it gets counted as resource usage in timers, add a short delay here so that threads
            # finish and don't influence our timers below.
            time.sleep(0.1)

            with timer.Timer("VisitProcessing"):

                if self.args.divide > 1:

                    tiles = geom.make_square_tiles(
                        constants.FOV_rad, self.args.divide, self.args.divide, pointing_v)

                    # spawn subprocesses to handle individual tiles
                    children = []
                    for ix, iy, region in tiles:

                        # make sure lastSourceId is unique in in each process
                        self.lastSourceId += len(sources)
                        tile = (ix, iy)

                        pid = os.fork()
                        if pid == 0:
                            # child

                            self.visit(db, visit_id, dt, region, sources, indices, tile)
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
                                _LOG.warning(COLOR_RED + "Child process PID=%s failed: %s" +
                                             COLOR_RESET, pid, status)
                        except OSError as exc:
                            _LOG.warning(COLOR_RED + "wait failed for PID=%s: %s" +
                                         COLOR_RESET, pid, exc)

                else:

                    # do it in-process
                    self.visit(db, visit_id, dt, region, sources, indices)

                if not self.args.no_update:
                    # store last visit info
                    db.saveVisit(visit_id, dt)

            _LOG.info(COLOR_BLUE + "--- Finished processing visit %s, time: %s" +
                      COLOR_RESET, visit_id, loop_timer)

        return 0

    def visit(self, db, visit_id, dt, region, sources, indices, tile=None):
        """AP processing of a single visit (with known sources)

        Parameters
        ----------
        visit_id : `int`
        dt : `datetime`
            Time of visit
        region : `sphgem.Region`
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
        if tile is not None:
            name = "tile={}x{} ".format(*tile)

        # make a mask
        for i in range(len(sources)):
            xyz = sources[i]
            if not region.contains(UnitVector3d(xyz[0], xyz[1], xyz[2])):
                indices[i] = _OUTSIDER

        with timer.Timer(name+"Objects-read"):

            # determine indices that we need
            ranges = _htm_indices(region)

            # Retrieve DiaObjects (latest versions) from database for matching,
            # this will produce wider coverage so further filtering is needed
            latest_objects = db.getDiaObjects(ranges)
            _LOG.info(name+'database found %s objects', len(latest_objects))

            # filter database obects to a mask
            latest_objects = self._filterDiaObjects(latest_objects, region)
            _LOG.info(name+'after filtering %s objects', len(latest_objects))

        with timer.Timer(name+"S2O-matching"):

            # create all new DiaObjects
            objects = self._makeDiaObjects(sources, indices, dt)

            # do forced photometry on non-detected objects (extends objects)
            self._forcedPhotometry(objects, latest_objects, dt)

            # Store all sources
            srcs = self._makeDiaSources(sources, indices, visit_id)

            # make forced sources for every object
            fsrcs = self._makeDiaForcedSources(objects, visit_id)

        with timer.Timer(name+"Source-read"):

            latest_objects_ids = [obj['id'] for obj in latest_objects]
            if self.args.sources_region:
                read_srcs = db.getDiaSourcesInRegion(ranges, dt)
            else:
                read_srcs = db.getDiaSources(latest_objects_ids, dt)
            _LOG.info(name+'database found %s sources', len(read_srcs or []))

            read_srcs = db.getDiaForcedSources(latest_objects_ids, dt)
            _LOG.info(name+'database found %s forced sources', len(read_srcs or []))

        if not self.args.no_update:

            with timer.Timer(name+"L1-store"):

                # store new versions of objects
                _LOG.info(name+'will store %d Objects', len(objects))
                if objects:
                    db.storeDiaObjects(objects, dt)

                # store all sources
                _LOG.info(name+'will store %d Sources', len(srcs))
                if srcs:
                    db.storeDiaSources(srcs)

                # store all forced sources
                _LOG.info(name+'will store %d ForcedSources', len(fsrcs))
                if fsrcs:
                    db.storeDiaForcedSources(fsrcs)

    def _filterDiaObjects(self, latest_objects, region):
        """Filter out objects from a catalog which are outside region.

        Parameters
        ----------
        latest_objects : `afw.table.BaseCatalog`
            Catalog containing DiaObject records
        region : `sphgem.Region`

        Returns
        -------
        Filtered `afw.table.BaseCatalog` containing only records contained
        in the region.
        """
        mask = numpy.ndarray(len(latest_objects), dtype=bool)
        for i, obj in enumerate(latest_objects):
            # TODO: For now we use PPDB units (degrees), will have to be changed
            # in case we adopt afw units.
            lonLat = LonLat.fromRadians(obj['coord_ra'].asRadians(), obj['coord_dec'].asRadians())
            dir_obj = UnitVector3d(lonLat)
            mask[i] = region.contains(dir_obj)

        return latest_objects.subset(mask)

    def _makeDiaObjectSchema(self):
        """Make afw.table schema for DiaObjects.

        Schema should be compatible with PPDB schema and it should contain
        all fields that have no default values in the schema.
        """
        schema = make_minimal_dia_object_schema()
        return schema

    def _makeDiaObjects(self, sources, indices, dt):
        """Over-simplified implementation of source-to-object matching and
        new DiaObject generation.

        Currently matching is based on info passed along by source
        generator and does not even use DiaObjects from database.

        Parameters
        ----------
        sources : `numpy.array`
            (x, y, z) coordinates of sources, array dimension is (N, 3)
        indices : `numpy.array`
            array of indices of sources, 1-dim ndarray, transient sources
            have negative indices
        dt : `datetime.datetime`
            Visit time.

        Returns
        -------
        `afw.table.BaseCatalog`
        """

        schema = self._makeDiaObjectSchema()
        catalog = afwTable.SourceCatalog(schema)
        n_trans = 0
        for i in range(len(sources)):

            var_idx = int(indices[i])
            if var_idx == _OUTSIDER:
                continue

            xyz = sources[i]
            if var_idx >= _TRANSIENT_START_ID:
                n_trans += 1

            v3d = Vector3d(xyz[0], xyz[1], xyz[2])
            sp = SpherePoint(v3d)

            dir_v = UnitVector3d(v3d)
            pixelator = HtmPixelization(constants.HTM_LEVEL)
            index = pixelator.index(dir_v)

            record = catalog.addNew()
            record.set("id", var_idx)
            # record.set("validityStart", _utc_seconds(dt))
            # record.set("validityEnd", 0)
            # record.set("lastNonForcedSource", _utc_seconds(dt))
            record.set("coord_ra", sp.getRa())
            record.set("coord_dec", sp.getDec())
            record.set("pixelId", index)

        _LOG.info('found %s matching objects and %s transients/noise', len(catalog) - n_trans, n_trans)

        return catalog

    def _forcedPhotometry(self, objects, latest_objects, dt):
        """Do forced photometry on latest_objects which are not in objects.

        Extends objects catalog with new DiaObjects.

        Parameters
        ----------
        objects : `afw.table.BaseCatalog`
            Catalog containing DiaObject records
        latest_objects : `afw.table.BaseCatalog`
            Catalog containing DiaObject records
        """

        # Ids of the detected objects
        ids = set(obj['id'] for obj in objects)

        for obj in latest_objects:
            # only do it for 30 days after last detection
            if obj['id'] in ids:
                continue
            lastNonForcedSource = datetime.utcfromtimestamp(obj['lastNonForcedSource'])
            delta = dt - lastNonForcedSource
            if delta > timedelta(days=30):
                continue

            record = objects.addNew()
            record.set("id", obj['id'])
            # record.set("validityStart", _utc_seconds(dt))
            # record.set("validityEnd", 0)
            # record.set("lastNonForcedSource", obj['lastNonForcedSource'])
            record.set("coord_ra", obj["coord_ra"])
            record.set("coord_dec", obj["coord_dec"])
            record.set("pixelId", obj["pixelId"])

    def _makeDiaSourceSchema(self):
        """Make afw.table schema for DiaSource.

        Schema should be compatible with PPDB schema and it should contain
        all fields that have no default values in the schema.
        """
        schema = make_minimal_dia_source_schema()
        return schema

    def _makeDiaSources(self, sources, indices, visit_id):
        """Generate catalog of DiaSources to store in a database

        Parameters
        ----------
        sources : `numpy.ndarray`
            (x, y, z) coordinates of sources, array dimension is (N, 3)
        indices : `numpy.array`
            array of indices of sources, 1-dim ndarray, transient sources
            have negative indices
        visit_id : `int`
            ID of the visit

        Returns
        -------
        `afw.table.BaseCatalog`
        """

        schema = self._makeDiaSourceSchema()
        catalog = afwTable.BaseCatalog(schema)
        for i in range(len(sources)):

            var_idx = int(indices[i])
            if var_idx == _OUTSIDER:
                continue

            xyz = sources[i]
            v3d = Vector3d(xyz[0], xyz[1], xyz[2])
            sp = SpherePoint(v3d)

            dir_v = UnitVector3d(v3d)
            pixelator = HtmPixelization(constants.HTM_LEVEL)
            index = pixelator.index(dir_v)

            self.lastSourceId += 1

            record = catalog.addNew()
            record.set("id", self.lastSourceId)
            record.set("ccdVisitId", visit_id)
            record.set("diaObjectId", var_idx)
            record.set("parent", 0)
            record.set("coord_ra", sp.getRa())
            record.set("coord_dec", sp.getDec())
            record.set("flags", 0)
            record.set("pixelId", index)

        return catalog

    def _makeDiaForcedSourceSchema(self):
        """Make afw.table schema for DiaForcedSource.

        Schema should be compatible with PPDB schema and it should contain
        all fields that have no default values in the schema.
        """
        schema = afwTable.Schema()

        # PPDB-specific stuff
        schema.addField("diaObjectId", "L")
        schema.addField("ccdVisitId", "L")
        schema.addField("flags", "L")

        return schema

    def _makeDiaForcedSources(self, objects, visit_id):
        """Generate catalog of DiaForcedSources to store in a database.

        Parameters
        ----------
        objects : `afw.table.BaseCatalog`
            Catalog containing DiaObject records
        visit_id : `int`
            ID of the visit

        Returns
        -------
        `afw.table.BaseCatalog`
        """

        schema = self._makeDiaForcedSourceSchema()
        catalog = afwTable.BaseCatalog(schema)
        for obj in objects:

            record = catalog.addNew()
            record.set("diaObjectId", obj['id'])
            record.set("ccdVisitId", visit_id)
            record.set("flags", 0)

        return catalog


#
#  run application when imported as a main module
#
if __name__ == "__main__":
    app = APProto()
    rc = app.run()
    sys.exit(rc)
