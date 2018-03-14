"""
Module defining L1db class and related methods.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
from collections import namedtuple
from datetime import datetime
import logging
import math
import sys
try:
    from ConfigParser import ConfigParser, NoSectionError
except ImportError:
    from configparser import ConfigParser, NoSectionError

#-----------------------------
# Imports for other modules --
#-----------------------------
from . import constants, timer, l1dbschema
from lsst.db import engineFactory
from lsst import sphgeom
import sqlalchemy
from sqlalchemy import (Column, engine, event, func, Index,
                        PrimaryKeyConstraint, sql, Table)
from sqlalchemy.pool import NullPool

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_LOG = logging.getLogger(__name__)


class Timer(object):
    """Timer class defining context manager which tracks execution timing.

    Typical use:

        with Timer("timer_name"):
            do_something

    On exit from block it will print elapsed time.

    See also :py:mod:`timer` module.
    """
    def __init__(self, name, log_before_cursor_execute=False):
        self._log_before_cursor_execute = log_before_cursor_execute
        self._timer1 = timer.Timer(name)
        self._timer2 = timer.Timer(name + " (before/after cursor)")

    def __enter__(self):
        """
        Enter context, start timer
        """
#         event.listen(engine.Engine, "before_cursor_execute", self._start_timer)
#         event.listen(engine.Engine, "after_cursor_execute", self._stop_timer)
        self._timer1.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context, stop and dump timer
        """
        if exc_type is None:
            self._timer1.stop()
            self._timer1.dump()
#         event.remove(engine.Engine, "before_cursor_execute", self._start_timer)
#         event.remove(engine.Engine, "after_cursor_execute", self._stop_timer)
        return False

    def _start_timer(self, conn, cursor, statement, parameters, context, executemany):
        if self._log_before_cursor_execute:
            _LOG.info("before_cursor_execute")
        self._timer2.start()

    def _stop_timer(self, conn, cursor, statement, parameters, context, executemany):
        self._timer2.stop()
        self._timer2.dump()

#------------------------
# Exported definitions --
#------------------------

#
# Data classes (named tuples) for data accepted or returned from database methods
#


# Information about single visit
Visit = namedtuple('Visit', 'visitId visitTime lastObjectId lastSourceId')

# Truncated version of DIAObject, this is returned from database query and it
# should contain enough info for Source matching
DiaObject_short = namedtuple('DiaObject_short', """
    diaObjectId lastNonForcedSource ra decl raSigma declSigma ra_decl_Cov htmId20
    """)

# Full version of DIAObject data, this is what we store in the database
DiaObject = namedtuple('DiaObject', """
    diaObjectId validityStart validityEnd lastNonForcedSource
    ra decl raSigma declSigma ra_decl_Cov
    radecTai pmRa pmRaSigma pmDecl pmDeclSigma pmRa_pmDecl_Cov
    parallax parallaxSigma pmRa_parallax_Cov pmDecl_parallax_Cov
    pmParallaxLnL pmParallaxChi2 pmParallaxNdata
    flags htmId20
    """)

# Truncated version of DIASource data
DiaSource = namedtuple('DiaSource', """
    diaSourceId ccdVisitId diaObjectId
    prv_procOrder midPointTai
    ra raSigma decl declSigma ra_decl_Cov
    x xSigma y ySigma x_y_Cov
    apFlux apFluxErr snr
    flags htmId20
    """)

# Full version of DIASource data
DiaSource_full = namedtuple('DiaSource_full', """
    diaSourceId ccdVisitId diaObjectId ssObjectId parentDiaSourceId
    prv_procOrder ssObjectReassocTime midPointTai
    ra raSigma decl declSigma ra_decl_Cov
    x xSigma y ySigma x_y_Cov
    apFlux apFluxErr snr
    psFlux psFluxSigma psRa psRaSigma psDecl psDeclSigma
    psFlux_psRa_Cov psFlux_psDecl_Cov psRa_psDecl_Cov psLnL psChi2 psNdata
    trailFlux trailFluxSigma trailRa trailRaSigma trailDecl trailDeclSigma
    trailLength trailLengthSigma trailAngle trailAngleSigma
    trailFlux_trailRa_Cov trailFlux_trailDecl_Cov trailFlux_trailLength_Cov
    trailFlux_trailAngle_Cov trailRa_trailDecl_Cov trailRa_trailLength_Cov
    trailRa_trailAngle_Cov trailDecl_trailLength_Cov trailDecl_trailAngle_Cov
    trailLength_trailAngle_Cov trailLnL trailChi2 trailNdata
    dipMeanFlux dipMeanFluxSigma dipFluxDiff dipFluxDiffSigma dipRa dipRaSigma
    dipDecl dipDeclSigma dipLength dipLengthSigma dipAngle dipAngleSigma
    dipMeanFlux_dipFluxDiff_Cov dipMeanFlux_dipRa_Cov dipMeanFlux_dipDecl_Cov
    dipMeanFlux_dipLength_Cov dipMeanFlux_dipAngle_Cov dipFluxDiff_dipRa_Cov
    dipFluxDiff_dipDecl_Cov dipFluxDiff_dipLength_Cov dipFluxDiff_dipAngle_Cov
    dipRa_dipDecl_Cov dipRa_dipLength_Cov dipRa_dipAngle_Cov dipDecl_dipLength_Cov
    dipDecl_dipAngle_Cov dipLength_dipAngle_Cov dipLnL dipChi2 dipNdata
    totFlux totFluxErr diffFlux diffFluxErr fpBkgd fpBkgdErr
    ixx ixxSigma iyy iyySigma ixy ixySigma ixx_iyy_Cov ixx_ixy_Cov iyy_ixy_Cov ixxPSF iyyPSF ixyPSF
    extendedness spuriousness
    flags htmId20
    """)

# DIAForcedSource data
DiaForcedSource = namedtuple('DiaForcedSource', """
    diaObjectId  ccdVisitId
    psFlux psFluxSigma
    x y
    flags
    """)


def _row2nt(row, tupletype):
    """
    Covert result row into an named tuple.
    """
    return tupletype(**dict(row))

def _htm_indices(region):
    """
    Generate a set of HTM indices covering specified field of view.

    Retuns sequence of ranges, range is a tuple (minHtmID, maxHtmID).

    @param region: sphgeom Region instance
    """

    _LOG.debug('circle: %s', region)
    pixelator = sphgeom.HtmPixelization(constants.HTM_LEVEL)
    indices = pixelator.envelope(region, constants.HTM_MAX_RANGES)
    for range in indices.ranges():
        _LOG.debug('range: %s %s', pixelator.toString(range[0]), pixelator.toString(range[1]))

    return indices.ranges()

#---------------------
#  Class definition --
#---------------------


class L1db(object):
    """Interface to L1 database, hides all database access details.

    The implementation is configured via configuration file (to simplify
    prototyping) which is an INI-format file with all parameters. For an
    example check cfg/ forder.

    Parameters
    ----------
    config : str
        Name of the configuration file.
    """
    def __init__(self, config):

        parser = ConfigParser()
        parser.readfp(open(config), config)

        # engine is reused between multiple processes, make sure that we don't
        # share connections by disabling pool (by using NullPool class)
        options = dict(parser.items("database"))
        self._engine = sqlalchemy.create_engine(options['url'], poolclass=NullPool,
                                                isolation_level='READ_COMMITTED')

        # get all parameters from config file
        try:
            options = dict(parser.items("l1db"))
        except NoSectionError:
            options = {}
        self._dia_object_index = options.get('dia_object_index', 'baseline')
        self._dia_object_nightly = bool(int(options.get('dia_object_nightly', 0)))
        self._months_sources = int(options.get('read_sources_months', 0))
        self._months_fsources = int(options.get('read_forced_sources_months', 0))
        self._read_full_objects = bool(int(options.get('read_full_objects', 0)))
        self._source_select = options.get('source_select', "by-fov")
        self._object_last_replace = bool(int(options.get('object_last_replace', 0)))
        schema_file = options.get('schema_file')
        extra_schema_file = options.get('extra_schema_file')

        if self._dia_object_index not in ('baseline', 'htm20_id_iov', 'last_object_table'):
            raise ValueError('unexpected dia_object_index value: ' + str(self._dia_object_index))
        if self._source_select not in ('by-fov', 'by-oid'):
            raise ValueError('unexpected source_select value: ' + self._source_select)

        self._schema = l1dbschema.L1dbSchema(engine=self._engine,
                                             dia_object_index=self._dia_object_index,
                                             dia_object_nightly=self._dia_object_nightly,
                                             schema_file=schema_file,
                                             extra_schema_file=extra_schema_file)

        _LOG.info("L1DB Configuration:")
        _LOG.info("    dia_object_index: %s", self._dia_object_index)
        _LOG.info("    dia_object_nightly: %s", self._dia_object_nightly)
        _LOG.info("    read_sources_months: %s", self._months_sources)
        _LOG.info("    read_forced_sources_months: %s", self._months_fsources)
        _LOG.info("    read_full_objects: %s", self._read_full_objects)
        _LOG.info("    source_select: %s", self._source_select)
        _LOG.info("    object_last_replace: %s", self._object_last_replace)
        _LOG.info("    schema_file: %s", schema_file)
        _LOG.info("    extra_schema_file: %s", extra_schema_file)

    #-------------------
    #  Public methods --
    #-------------------

    def lastVisit(self):
        """Returns last visit information or `None` if visits table is empty.

        Visits table is used by ap_proto to track visit information, it is
        not a part of the regular L1DB schema.

        Returns
        -------
        Instance of :py:class:`Visit` class or None.
        """

        with self._engine.begin() as conn:

            stmnt = sql.select([sql.func.max(self._schema.visits.c.visitId),
                                sql.func.max(self._schema.visits.c.visitTime)])
            res = conn.execute(stmnt)
            row = res.fetchone()
            if row[0] is None:
                return None

            visitId = row[0]
            visitTime = row[1]

            # get max IDs from corresponding tables
            stmnt = sql.select([sql.func.max(self._schema.objects.c.diaObjectId)])
            lastObjectId = conn.scalar(stmnt)
            stmnt = sql.select([sql.func.max(self._schema.sources.c.diaSourceId)])
            lastSourceId = conn.scalar(stmnt)

            return Visit(visitId=visitId, visitTime=visitTime,
                         lastObjectId=lastObjectId, lastSourceId=lastSourceId)

    def saveVisit(self, visitId, visitTime):
        """Store visit information.

        Parameters
        ----------
        visitId : int
            Visit identifier
        visitTime : `datetime.datetime`
            Visit timestamp.
        """

        ins = self._schema.visits.insert().values(visitId=visitId,
                                                  visitTime=visitTime)
        self._engine.execute(ins)

    def tableRowCount(self):
        """Returns dictionary with the table names and row counts.

        Used by ap_proto to keep track of the size of the database tables.
        Depending on database technology this could be expensive operation.

        Returns
        -------
        Dict where key is a table mane and value is a row count.
        """
        res = {}
        tables = [self._schema.objects, self._schema.sources, self._schema.forcedSources]
        if self._dia_object_index == 'last_object_table':
            tables.append(self._schema.objects_last)
        for table in tables:
            stmt = sql.select([func.count()]).select_from(table)
            count = self._engine.scalar(stmt)
            res[table.name] = count

        return res

    def getDiaObjects(self, region, explain=False):
        """Returns the list of DiaObject instances from given region.

        Returns only the last version of the the DIAObject. Object are
        searched based on HTM index. Returned list can contain objects
        outside given region, further filtering should be applied.

        Parameters
        ----------
        region: `sphgeom.Region`
            Sky region, can be FOV or single CCD.
        explain: bool
            If true then also run database query with EXPLAIN, may be
            useful for understanding query performance.

        Returns
        -------
        List of `DiaObject` instances.
        """

        query = "SELECT "

        # decide what columns we need
        if self._dia_object_index == 'last_object_table':
            table = self._schema.objects_last
        else:
            table = self._schema.objects
        if self._read_full_objects:
            query += "*"
        else:
            query += '"diaObjectId","lastNonForcedSource","ra","decl","raSigma","declSigma","ra_decl_Cov","htmId20"'
        query += ' FROM "' + table.name + '" WHERE ('

        # determine indices that we need
        ranges = _htm_indices(region)

        # build selection
        exprlist = []
        for low, up in ranges:
            up -= 1
            if low == up:
                exprlist.append('"htmId20" = ' + str(low))
            else:
                exprlist.append('"htmId20" BETWEEN {} AND {}'.format(low, up))
        query += ' OR '.join(exprlist)
        query += ')'

        # select latest version of objects
        if self._dia_object_index != 'last_object_table':
            query += ' AND "validityEnd" IS NULL'

        _LOG.debug("query: %s", query)

        if explain:
            # run the same query with explain
            self._explain(query, self._engine)

        # execute select
        with Timer('DiaObject select'):
            res = self._engine.execute(sql.text(query))
        obj_type = DiaObject if self._read_full_objects else DiaObject_short
        objects = [_row2nt(row, obj_type) for row in res]
        _LOG.debug("found %s DiaObjects", len(objects))
        return objects

    def getDiaSources(self, region, objects, explain=False):
        """Returns the list of DiaSource instances from given region or
        matching given DiaObjects.

        Depending on configuration this method can chose to do spatial
        qury based on HTM index or select based on DiaObject IDs.

        Parameters
        ----------
        region: `sphgeom.Region`
            Sky region, can be FOV or single CCD.
        objects : `list`
            List of `DiaObject` instances
        explain: bool
            If true then also run database query with EXPLAIN, may be
            useful for understanding query performance.

        Returns
        -------
        List of `DiaSource` instances.
        """

        if self._months_sources == 0:
            _LOG.info("Skip DiaSources fetching")
            return []

        if not objects:
            _LOG.info("Skip DiaSources fetching - no Objects")
            return []

        table = self._schema.sources
        query = 'SELECT *  FROM "' + table.name + '" WHERE '

        if self._source_select == 'by-fov':
            # determine indices that we need
            ranges = _htm_indices(region)

            # build selection
            exprlist = []
            for low, up in ranges:
                up -= 1
                if low == up:
                    exprlist.append('"htmId20" = ' + str(low))
                else:
                    exprlist.append('"htmId20" BETWEEN {} AND {}'.format(low, up))
            query += '(' + ' OR '.join(exprlist) + ')'
        else:
            # select by object id
            ids = sorted([obj.diaObjectId for obj in objects])
            ids = ",".join(str(id) for id in ids)
            query += '"diaObjectId" IN (' + ids + ') '

        # execute select
        with Timer('DiaSource select'):
            res = self._engine.execute(sql.text(query))
        sources = [_row2nt(row, DiaSource_full) for row in res]
        _LOG.debug("found %s DiaSources", len(sources))
        return sources

    def getDiaFSources(self, objects, explain=False):
        """Returns the list of DiaForceSource instances matching given
        DiaObjects.

        Parameters
        ----------
        objects : `list`
            List of `DiaObject` instances
        explain: bool
            If true then also run database query with EXPLAIN, may be
            useful for understanding query performance.

        Returns
        -------
        List of `DiaForcedSource` instances.
        """

        if self._months_fsources == 0:
            _LOG.info("Skip DiaForceSources fetching")
            return []

        if not objects:
            _LOG.info("Skip DiaSources fetching - no Objects")
            return []

        table = self._schema.forcedSources
        query = 'SELECT *  FROM "' + table.name + '" WHERE '

        # select by object id
        ids = sorted([obj.diaObjectId for obj in objects])
        ids = ",".join(str(id) for id in ids)
        query += '"diaObjectId" IN (' + ids + ') '

        # execute select
        with Timer('DiaForcedSource select'):
            res = self._engine.execute(sql.text(query))
        sources = [_row2nt(row, DiaForcedSource) for row in res]
        _LOG.debug("found %s DiaForcedSources", len(sources))
        return sources

    def storeDiaObjects(self, objs, dt, explain=False):
        """Store a set of DIAObjects from current visit.

        Parameters
        ----------
        objs : `list`
            List of `DiaObject` instances to store
        dt : `datetime.datetime`
            Time of the visit
        explain : bool
            If true then also run database query with EXPLAIN, may be
            useful for understanding query performance.
        """

        ids = sorted([obj.diaObjectId for obj in objs])
        _LOG.info("first object ID: %d", ids[0])

        # everything to be done in single transaction
        with self._engine.begin() as conn:

            ids = ",".join(str(id) for id in ids)

            if self._dia_object_index == 'last_object_table':

                # insert and replace all records in LAST table, mysql and postgres have
                # non-standard features (handled in _storeObjects)
                table = self._schema.objects_last
                do_replace = self._object_last_replace
                if not do_replace:
                    query = 'DELETE FROM "' + table.name + '" '
                    query += 'WHERE "diaObjectId" IN (' + ids + ') '

                    if explain:
                        # run the same query with explain
                        self._explain(query, conn)

                    with Timer(table.name + ' delete'):
                        res = conn.execute(sql.text(query))
                    _LOG.debug("deleted %s objects", res.rowcount)

                self._storeObjects(DiaObject, objs, conn, table, explain, do_replace)

            else:

                # truncate existing validity intervals
                table = self._schema.objects
                query = 'UPDATE "' + table.name + '" '
                query += "SET \"validityEnd\" = '" + str(dt) + "' "
                query += 'WHERE "diaObjectId" IN (' + ids + ') '
                query += 'AND "validityEnd" IS NULL'

                # _LOG.debug("query: %s", query)

                if explain:
                    # run the same query with explain
                    self._explain(query, conn)

                with Timer(table.name + ' truncate'):
                    res = conn.execute(sql.text(query))
                _LOG.debug("truncated %s intervals", res.rowcount)

            # insert new versions
            if self._dia_object_nightly:
                table = self._schema.objects_nightly
            else:
                table = self._schema.objects
            self._storeObjects(DiaObject, objs, conn, table, explain)

    def storeDiaSources(self, sources, explain=False):
        """Store a set of DIASources from current visit.

        Parameters
        ----------
        sources : `list`
            List of `DiaSource` instances to store
        explain : bool
            If true then also run database query with EXPLAIN, may be
            useful for understanding query performance.
        """

        # everything to be done in single transaction
        with self._engine.begin() as conn:

            table = self._schema.sources
            self._storeObjects(DiaSource, sources, conn, table, explain)

    def storeDiaForcedSources(self, sources, explain=False):
        """Store a set of DIAForcedSources from current visit.

        Parameters
        ----------
        sources : `list`
            List of `DiaForcedSource` instances to store
        explain : bool
            If true then also run database query with EXPLAIN, may be
            useful for understanding query performance.
        """

        # everything to be done in single transaction
        with self._engine.begin() as conn:

            table = self._schema.forcedSources
            self._storeObjects(DiaForcedSource, sources, conn, table, explain)

    def dailyJob(self):
        """Implement daily activities like cleanup/vacuum.

        What should be done during daily cleanup is determined by
        configuration/schema.
        """

        # move data from DiaObjectNightly into DiaObject
        query = 'INSERT INTO "' + self._schema.objects.name + '" '
        query += 'SELECT * FROM "' + self._schema.objects_nightly.name + '"'
        with Timer('DiaObjectNightly copy'):
            res = self._engine.execute(sql.text(query))

        query = 'DELETE FROM "' + self._schema.objects_nightly.name + '"'
        with Timer('DiaObjectNightly delete'):
            res = self._engine.execute(sql.text(query))

        if self._engine.name == 'postgresql':

            # do VACUUM on all tables
            _LOG.info("Running VACUUM on all tables")
            connection = self._engine.raw_connection()
            ISOLATION_LEVEL_AUTOCOMMIT = 0
            connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = connection.cursor()
            cursor.execute("VACUUM ANALYSE")

    def makeSchema(self, drop=False, mysql_engine='InnoDB'):
        """Create or re-create all tables.

        Parameters
        ----------
        drop : boolean
            If True then drop tables before creating new ones.
        """
        self._schema.makeSchema(drop=drop, mysql_engine=mysql_engine)

    def _explain(self, query, conn):
        # run the query with explain

        _LOG.info("explain for query: %s...", query[:64])

        if conn.engine.name == 'mysql':
            query = "EXPLAIN EXTENDED " + query
        else:
            query = "EXPLAIN " + query

        res = conn.execute(sql.text(query))
        if res.returns_rows:
            _LOG.info("explain: %s", res.keys())
            for row in res:
                _LOG.info("explain: %s", row)
        else:
            _LOG.info("EXPLAIN returned nothing")

    def _storeObjects(self, object_type, objects, conn, table, explain=False, replace=False):
        """
        Generic store method.

        Stores a bunch of objects as records in a table.

        Parameters
        ----------
        object_type : `type`
            Namedtuple type, e.g. DiaSource
        objects : `list`
            Sequence of objects of the `object_type` type
        conn :
            Database connection
        table :
            Database table (SQLAlchemy table instance)
        explain : bool
            If True then do EXPLAIN on INSERT query
        """

        def quoteValue(v):
            if v is None:
                v = "NULL"
            elif isinstance(v, datetime):
                v = "'" + str(v) + "'"
            elif isinstance(v, str):
                # we don't expect nasty stuff in strings
                v = "'" + v + "'"
            else:
                # assume numeric stuff
                v = str(v)
            return v

        fields = ['"' + f + '"' for f in object_type._fields]

        if replace and conn.engine.name == 'mysql':
            query = 'REPLACE INTO'
        else:
            query = 'INSERT INTO'
        query += ' "' + table.name + '" (' + ','.join(fields) + ') VALUES '

        values = []
        for obj in objects:
            row = [quoteValue(v) for v in obj]
            values.append('(' + ','.join(row) + ')')

        if explain:
            # run the same query with explain, only give it one row of data
            self._explain(query + values[0], conn)

        query += ','.join(values)

        if replace and conn.engine.name == 'postgresql':
            # This depends on that "replace" can only be true for DiaObjectLast table
            pk = ('htmId20', 'diaObjectId')
            query += " ON CONFLICT (\"{}\", \"{}\") DO UPDATE SET ".format(*pk)
            fields = ['"{0}" = EXCLUDED."{0}"'.format(field)
                      for field in object_type._fields if field not in pk]
            query += ', '.join(fields)

        # _LOG.debug("query: %s", query)
        _LOG.info("%s: will store %d records", table.name, len(objects))
        with Timer(table.name + ' insert'):
            res = conn.execute(sql.text(query))
        _LOG.debug("inserted %s intervals", res.rowcount)
