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

__all__ = ["APProtoVisitExecutor", "InMemoryExecutor", "SubprocessExecutor"]

import abc
import base64
import logging
import os
import random
import string
import tempfile
import time
from datetime import timedelta
from typing import Any, cast

import astropy.time
import felis.datamodel
import numpy
import pandas

from lsst.dax.apdb import Apdb, ApdbTables, monitor, timer
from lsst.geom import SpherePoint
from lsst.sphgeom import LonLat, Region, UnitVector3d, Vector3d

from . import L1dbprotoConfig, geom

_LOG = logging.getLogger("ap_proto")

_MON = monitor.MonAgent("ap_proto")

# special code to mark sources outside region
_OUTSIDER = -666

# transient ID start value
_TRANSIENT_START_ID = 1000000000


def _nrows(table: pandas.DataFrame | None) -> int:
    if table is None:
        return 0
    else:
        return len(table)


class APProtoVisitExecutor(abc.ABC):
    """Interface for objects generating data for one visit."""

    @abc.abstractmethod
    def visit(
        self,
        visit_id: int,
        visit_time: astropy.time.Time,
        region: Region,
        sources: numpy.ndarray,
        indices: numpy.ndarray,
    ) -> None:
        """AP processing of a single visit (with known sources)

        Parameters
        ----------
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
        """
        raise NotImplementedError()


class InMemoryExecutor(APProtoVisitExecutor):
    """In-process generation of visit data."""

    def __init__(
        self,
        config: L1dbprotoConfig,
        db: Apdb | None,
        db_uri: str,
        no_update: bool,
        detector: int,
        n_detectors: int,
        store_reconnect: bool,
        tile: tuple[int, int] | None = None,
    ):
        self.config = config
        self.db = db
        self.db_uri = db_uri
        self.no_update = no_update
        self.detector = detector
        self.n_detectors = n_detectors
        self.store_reconnect = store_reconnect
        self.tile = tile

    def visit(
        self,
        visit_id: int,
        visit_time: astropy.time.Time,
        region: Region,
        sources: numpy.ndarray,
        indices: numpy.ndarray,
    ) -> None:
        """AP processing of a single visit (with known sources)

        Parameters
        ----------
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
        """
        if self.config.random_delay > 0:
            delay = random.uniform(0.0, self.config.random_delay)
            _LOG.info("Sleeping for %.2f seconds", delay)
            time.sleep(delay)

        db = self.db
        if db is None:
            db = Apdb.from_uri(self.db_uri)

        name = ""

        src_read_period = self.config.src_read_period
        src_read_visits = round(self.config.src_read_period * self.config.src_read_duty_cycle)
        do_read_src = visit_id % src_read_period < src_read_visits

        # make a mask
        for i in range(len(sources)):
            xyz = sources[i]
            if not region.contains(UnitVector3d(xyz[0], xyz[1], xyz[2])):
                indices[i] = _OUTSIDER

        # Add padding.
        region = geom.padded_region(region, self.config.detector_region_padding)

        counts: dict[str, int] = {}

        with timer.Timer("tile_read_time", _MON, _LOG), _MON.context_tags({"task": "fetch"}):
            with timer.Timer(name + "Objects-read", _LOG):
                # Retrieve DiaObjects (latest versions) from database for
                # matching, this will produce wider coverage so further
                # filtering is needed.
                latest_objects = db.getDiaObjects(region)
                _LOG.info(name + "database found %s objects", _nrows(latest_objects))
                counts["objects"] = _nrows(latest_objects)

                # filter database objects to a mask
                latest_objects = self._filterDiaObjects(latest_objects, region)
                _LOG.info(name + "after filtering %s objects", _nrows(latest_objects))
                counts["objects_filtered"] = _nrows(latest_objects)

            with timer.Timer(name + "S2O-matching", _LOG):
                # make all sources
                srcs = self._makeDiaSources(sources, indices, visit_time, visit_id)

                # create all new DiaObjects
                objects = self._makeDiaObjects(sources, indices, latest_objects, visit_time)

                # do forced photometry (can extends objects)
                fsrcs = self._forcedPhotometry(objects, visit_time, visit_id)

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

        if not self.no_update:
            with _MON.context_tags({"task": "store"}):
                if self.store_reconnect:
                    del db
                    db = Apdb.from_uri(self.db_uri)

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
        count_map = dict(known_objects[["diaObjectId", "nDiaSources"]].itertuples(index=False))

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
                "ra": objects["ra"],
                "dec": objects["dec"],
                "visit": visit_id,
                "detector": self.detector,
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
        catalog["detector"] = self.detector
        catalog["parentDiaSourceId"] = 0
        catalog["psFlux"] = 1.0
        catalog["psFluxErr"] = 0.01
        catalog["midpointMjdTai"] = midpointMjdTai

        start_id = (visit_id * self.n_detectors + self.detector) * 1_000_000 + 1
        nrows = catalog.shape[0]
        catalog["diaSourceId"] = range(start_id, start_id + nrows)

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
                    data = rng.integers(0, 1, count, dtype=numpy.bool_)
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


class SubprocessExecutor(APProtoVisitExecutor):
    """Sub-process generation of visit data."""

    def __init__(
        self,
        app_config: str,
        db_uri: str,
        *,
        no_update: bool,
        verbose: bool,
        detector: int,
        n_detectors: int,
        tile: tuple[int, int] | None = None,
        rank: int | None = None,
        store_reconnect: bool = False,
    ):
        self.app_config = app_config
        self.db_uri = db_uri
        self.verbose = verbose
        self.no_update = no_update
        self.detector = detector
        self.n_detectors = n_detectors
        self.tile = tile
        self.rank = rank
        self.store_reconnect = store_reconnect

    def visit(
        self,
        visit_id: int,
        visit_time: astropy.time.Time,
        region: Region,
        sources: numpy.ndarray,
        indices: numpy.ndarray,
    ) -> None:
        cmd: list[str] = ["ap_proto_visit.py"]
        cmd += [f"--app-config={self.app_config}"]
        cmd += [f"--config={self.db_uri}"]
        if self.verbose:
            cmd += ["--verbose"]
        if self.no_update:
            cmd += ["--no-update"]
        if self.store_reconnect:
            cmd += ["--store-reconnect"]
        if self.rank is not None:
            cmd += [f"--rank={self.rank}"]
        cmd += [str(visit_id)]
        cmd += [f"{visit_time.tai.mjd:.10f}"]
        cmd += [base64.b64encode(region.encode()).decode()]
        with tempfile.NamedTemporaryFile(delete_on_close=False) as npz:
            numpy.savez(npz, sources=sources, indices=indices)
            cmd += [npz.name]
            npz.close()

            cmd += [str(self.detector)]
            cmd += [str(self.n_detectors)]
            if self.tile:
                cmd += [f"{self.tile[0]}x{self.tile[1]}"]

            _LOG.debug("Suprocess command: %s", cmd)
            env = {k: v for k, v in os.environ.items() if "MPI" not in k}
            code = os.spawnvpe(os.P_WAIT, cmd[0], cmd, env)
            if code != 0:
                raise OSError(f"Subprocess terminated with exit code {code}")
            _LOG.debug("Suprocess completed")
