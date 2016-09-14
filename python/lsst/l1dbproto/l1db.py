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
from . import constants, timer
from lsst.db import engineFactory
from lsst import sphgeom
import sqlalchemy
from sqlalchemy import (Column, engine, event, func, Index, MetaData,
                        PrimaryKeyConstraint, sql, Table)

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_LOG = logging.getLogger(__name__)


class Timer(object):

    def __init__(self, name):
        self._timer1 = timer.Timer(name)
        self._timer2 = timer.Timer(name + " (before/after cursor)")

    def __enter__(self):
        """
        Enter context, start timer
        """
        event.listen(engine.Engine, "before_cursor_execute", self._start_timer)
        event.listen(engine.Engine, "after_cursor_execute", self._stop_timer)
        self._timer1.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context, stop and dump timer
        """
        if exc_type is None:
            self._timer1.stop()
            self._timer1.dump()
        event.remove(engine.Engine, "before_cursor_execute", self._start_timer)
        event.remove(engine.Engine, "after_cursor_execute", self._stop_timer)
        return False

    def _start_timer(self, conn, cursor, statement, parameters, context, executemany):
        self._timer2.start()

    def _stop_timer(self, conn, cursor, statement, parameters, context, executemany):
        self._timer2.stop()
        self._timer2.dump()

#------------------------
# Exported definitions --
#------------------------

Visit = namedtuple('Visit', 'visitId visitTime lastObjectId lastSourceId')
DiaObject_short = namedtuple('DiaObject_short', """
    diaObjectId lastNonForcedSource ra decl raSigma declSigma ra_decl_Cov htmId20
    """)
DiaObject = namedtuple('DiaObject', """
    diaObjectId validityStart validityEnd lastNonForcedSource
    ra decl raSigma declSigma ra_decl_Cov
    muRa muRaSigma muDecl muDecSigma muRa_muDeclCov
    parallax parallaxSigma muRa_parallax_Cov muDecl_parallax_Cov
    lnL chi2 N
    flags htmId20
    """)
DiaSource = namedtuple('DiaSource', """
    diaSourceId ccdVisitId diaObjectId
    filterName prv_procOrder midPointTai
    ra raSigma decl declSigma ra_decl_Cov
    x xSigma y ySigma x_y_Cov snr
    flags htmId20
    """)
DiaSource_full = namedtuple('DiaSource_full', """
    diaSourceId ccdVisitId diaObjectId ssObjectId parentDiaSourceId
    filterName prv_procOrder ssObjectReassocTime midPointTai
    ra raSigma decl declSigma ra_decl_Cov
    x xSigma y ySigma x_y_Cov snr
    psFlux psFluxSigma psLnL psChi2 psN
    trailFlux trailFluxSigma trailLength trailLengthSigma
    trailAngle trailAngleSigma trailFlux_trailLength_Cov
    trailFlux_trailAngle_Cov trailLength_trailAngle_Cov
    trailLnL trailChi2 trailN
    fpFlux fpFluxSigma diffFlux diffFluxSigma
    fpSky fpSkySigma
    E1 E1Sigma E2 E2Sigma E1_E2_Cov
    mSum mSumSigma extendedness
    apMeanSb01 apMeanSb01Sigma
    apMeanSb02 apMeanSb02Sigma
    apMeanSb03 apMeanSb03Sigma
    apMeanSb04 apMeanSb04Sigma
    apMeanSb05 apMeanSb05Sigma
    apMeanSb06 apMeanSb06Sigma
    apMeanSb07 apMeanSb07Sigma
    apMeanSb08 apMeanSb08Sigma
    apMeanSb09 apMeanSb09Sigma
    apMeanSb10 apMeanSb10Sigma
    flags htmId20
    """)
DiaForcedSource = namedtuple('DiaForcedSource', """
    diaObjectId  ccdVisitId
    psFlux psFlux_Sigma
    x y
    flags
    """)


def _row2nt(row, tupletype):
    """
    Covert result row into an named tuple.
    """
    return tupletype(**dict(row))

def _htm_indices(xyz, FOV_rad):
    """
    Generate a set of HTM indices covering specified field of view.

    Retuns sequence of ranges, range is a tuple (minHtmID, maxHtmID).

    @param xyz: pointing direction
    @param FOV_rad: field of view, radians
    """

    dir_v = sphgeom.UnitVector3d(xyz[0], xyz[1], xyz[2])
    circle = sphgeom.Circle(dir_v, sphgeom.Angle(FOV_rad))
    _LOG.debug('circle: %s', circle)
    pixelator = sphgeom.HtmPixelization(constants.HTM_LEVEL)
    indices = pixelator.envelope(circle, constants.HTM_MAX_RANGES)
    for range in indices.ranges():
        _LOG.debug('range: %s %s', pixelator.toString(range[0]), pixelator.toString(range[1]))

    return indices.ranges()

#---------------------
#  Class definition --
#---------------------


class L1db(object):
    """
    Interface to L1 database, hides all database access details.
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, config):
        """
        @param config:  Configuration file name
        """
        # instantiate db engine
        self._engine = engineFactory.getEngineFromFile(config)
        self._metadata = MetaData(self._engine)
        self._tables = {}

        parser = ConfigParser()
        parser.readfp(open(config), config)
        try:
            options = dict(parser.items("l1db"))
        except NoSectionError:
            options = {}
        self._dia_object_index = options.get('dia_object_index', 'baseline')
        self._months_sources = int(options.get('read_sources_months', 0))
        self._months_fsources = int(options.get('read_forced_sources_months', 0))
        self._read_full_objects = bool(options.get('read_full_objects', 0))
        self._source_select = options.get('source_select', "by-fov")

        if self._dia_object_index not in ('baseline', 'htm20_id_iov'):
            raise ValueError('unexpected dia_object_index value: ' + str(self._dia_object_index))
        if self._source_select not in ('by-fov', 'by-oid'):
            raise ValueError('unexpected source_select value: ' + self._source_select)

        _LOG.info("L1DB Configuration:")
        _LOG.info("    dia_object_index: %s", self._dia_object_index)
        _LOG.info("    read_sources_months: %s", self._months_sources)
        _LOG.info("    read_forced_sources_months: %s", self._months_fsources)
        _LOG.info("    read_full_objects: %s", self._read_full_objects)
        _LOG.info("    source_select: %s", self._source_select)

    #-------------------
    #  Public methods --
    #-------------------

    def lastVisit(self):
        """
        Returns last visit information or None if visits table is empty.

        @return instance of Visit class or None
        """

        with self._engine.begin() as conn:

            stmnt = sql.select([sql.func.max(self._visits.c.visitId),
                                sql.func.max(self._visits.c.visitTime)])
            res = conn.execute(stmnt)
            row = res.fetchone()
            if row[0] is None:
                return None

            visitId = row[0]
            visitTime = row[1]

            # get max IDs from corresponding tables
            stmnt = sql.select([sql.func.max(self._objects.c.diaObjectId)])
            lastObjectId = conn.scalar(stmnt)
            stmnt = sql.select([sql.func.max(self._sources.c.diaSourceId)])
            lastSourceId = conn.scalar(stmnt)

            return Visit(visitId=visitId, visitTime=visitTime,
                         lastObjectId=lastObjectId, lastSourceId=lastSourceId)

    def saveVisit(self, visitId, visitTime):
        """
        Store visit information.
        """

        ins = self._visits.insert().values(visitId=visitId,
                                           visitTime=visitTime)
        self._engine.execute(ins)

    def tableRowCount(self):
        """
        Returns dictionary with the table names and row counts.
        """
        res = {}
        tables = [self._objects, self._sources, self._forcedSources]
        for table in tables:
            stmt = sql.select([func.count()]).select_from(table)
            count = self._engine.scalar(stmt)
            res[table] = count

        return res

    def getDiaObjects(self, xyz, FOV_rad, explain=False):
        """
        Returns the list of DiaObject instances around given direction.
        @param xyz: pointing direction
        @param FOV_rad: field of view, radians
        """

        query = "SELECT "

        # decide what columns we need
        table = self._objects
        if self._read_full_objects:
            query += "*"
        else:
            query += '"diaObjectId","lastNonForcedSource","ra","decl","raSigma","declSigma","ra_decl_Cov","htmId20"'
        query += ' FROM "' + table.name + '" WHERE ('

        # determine indices that we need
        ranges = _htm_indices(xyz, FOV_rad)

        # build selection
        exprlist = []
        for low, up in ranges:
            up -= 1
            if low == up:
                exprlist.append('"htmId20" = ' + str(low))
            else:
                exprlist.append('"htmId20" BETWEEN {} AND {}'.format(low, up))
        query += ' OR '.join(exprlist)

        # select latest version of objects
        query += ') AND "validityEnd" IS NULL'

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

    def getDiaSources(self, xyz, FOV_rad, objects, explain=False):
        """
        Returns the list of DiaSource instances around given direction or
        matching given DiaObjects.

        @param xyz: pointing direction
        @param FOV_rad: field of view, radians
        @param objects: list of DiaObject instances
        """

        if self._months_sources == 0:
            _LOG.info("Skip DiaSources fetching")

        table = self._sources
        query = 'SELECT *  FROM "' + table.name + '" WHERE '

        if self._source_select == 'by-fov':
            # determine indices that we need
            ranges = _htm_indices(xyz, FOV_rad)

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
        """
        Returns the list of DiaForceSource instances matching given DiaObjects.

        @param objects: list of DiaObject instances
        """

        if self._months_fsources == 0:
            _LOG.info("Skip DiaForceSources fetching")

        table = self._forcedSources
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
        """
        @param objs:  list of DiaObject instances
        @param dt:       datetime for visit
        @param explain:  if True then do EXPLAIN on INSERT query
        """

        ids = sorted([obj.diaObjectId for obj in objs])

        table = self._objects

        # everything to be done in single transaction
        with self._engine.begin() as conn:

            # truncate existing validity intervals
            ids = ",".join(str(id) for id in ids)
            query = 'UPDATE "' + table.name + '" '
            query += "SET \"validityEnd\" = '" + str(dt) + "' "
            query += 'WHERE "diaObjectId" IN (' + ids + ') '
            query += 'AND "validityEnd" IS NULL'

            # _LOG.debug("query: %s", query)

            if explain:
                # run the same query with explain
                self._explain(query, conn)

            with Timer('DiaObject truncate'):
                res = conn.execute(sql.text(query))
            _LOG.debug("truncated %s intervals", res.rowcount)

            # insert new versions
            table = self._objects
            self._storeObjects(DiaObject, objs, conn, table, explain)

    def storeDiaSources(self, sources, explain=False):
        """
        @param sources:  list of DiaSource instances
        @param explain:  if True then do EXPLAIN on INSERT query
        """

        # everything to be done in single transaction
        with self._engine.begin() as conn:

            table = self._sources
            self._storeObjects(DiaSource, sources, conn, table, explain)

    def storeDiaForcedSources(self, sources, explain=False):
        """
        @param sources:  list of DiaForcedSource instances
        @param explain:  if True then do EXPLAIN on INSERT query
        """

        # everything to be done in single transaction
        with self._engine.begin() as conn:

            table = self._forcedSources
            self._storeObjects(DiaForcedSource, sources, conn, table, explain)

    def makeSchema(self, drop=False):
        """
        Create all tables
        """

        mysql_engine = 'InnoDB'

        # type aliases
        DOUBLE = self._make_doube_type()
        FLOAT = sqlalchemy.types.Float
        DATETIME = sqlalchemy.types.TIMESTAMP
        BIGINT = sqlalchemy.types.BigInteger
        INTEGER = INT = sqlalchemy.types.Integer
        BLOB = sqlalchemy.types.LargeBinary
        CHAR = sqlalchemy.types.CHAR

        if self._dia_object_index == 'htm20_id_iov':
            constraints = [PrimaryKeyConstraint('htmId20', 'diaObjectId', 'validityStart',
                                                name='PK_DiaObject'),
                           Index('IDX_DiaObject_diaObjectId', 'diaObjectId')]
        else:
            constraints = [PrimaryKeyConstraint('diaObjectId', 'validityStart', name='PK_DiaObject'),
                           Index('IDX_DiaObject_validityStart', 'validityStart'),
                           Index('IDX_DiaObject_htmId20', 'htmId20')]
        diaObject = Table('DiaObject', self._metadata,
                          Column('diaObjectId', BIGINT, nullable=False),
                          Column('validityStart', DATETIME, nullable=False),
                          Column('validityEnd', DATETIME, nullable=True),
                          Column('lastNonForcedSource', DATETIME, nullable=False),
                          Column('ra', DOUBLE, nullable=False),
                          Column('decl', DOUBLE, nullable=False),
                          Column('raSigma', FLOAT, nullable=False),
                          Column('declSigma', FLOAT, nullable=False),
                          Column('ra_decl_Cov', FLOAT, nullable=False),
                          Column('muRa', FLOAT, nullable=False),
                          Column('muRaSigma', FLOAT, nullable=False),
                          Column('muDecl', FLOAT, nullable=False),
                          Column('muDecSigma', FLOAT, nullable=False),
                          Column('muRa_muDeclCov', FLOAT, nullable=False),
                          Column('parallax', FLOAT, nullable=False),
                          Column('parallaxSigma', FLOAT, nullable=False),
                          Column('muRa_parallax_Cov', FLOAT, nullable=False),
                          Column('muDecl_parallax_Cov', FLOAT, nullable=False),
                          Column('lnL', FLOAT, nullable=False),
                          Column('chi2', FLOAT, nullable=False),
                          Column('N', INT, nullable=False),
                          Column('uPSFlux', FLOAT, nullable=True),
                          Column('uPSFluxErr', FLOAT, nullable=True),
                          Column('uPSFluxSigma', FLOAT, nullable=True),
                          Column('uFPFlux', FLOAT, nullable=True),
                          Column('uFPFluxErr', FLOAT, nullable=True),
                          Column('uFPFluxSigma', FLOAT, nullable=True),
                          Column('gPSFlux', FLOAT, nullable=True),
                          Column('gPSFluxErr', FLOAT, nullable=True),
                          Column('gPSFluxSigma', FLOAT, nullable=True),
                          Column('gFPFlux', FLOAT, nullable=True),
                          Column('gFPFluxErr', FLOAT, nullable=True),
                          Column('gFPFluxSigma', FLOAT, nullable=True),
                          Column('rPSFlux', FLOAT, nullable=True),
                          Column('rPSFluxErr', FLOAT, nullable=True),
                          Column('rPSFluxSigma', FLOAT, nullable=True),
                          Column('rFPFlux', FLOAT, nullable=True),
                          Column('rFPFluxErr', FLOAT, nullable=True),
                          Column('rFPFluxSigma', FLOAT, nullable=True),
                          Column('iPSFlux', FLOAT, nullable=True),
                          Column('iPSFluxErr', FLOAT, nullable=True),
                          Column('iPSFluxSigma', FLOAT, nullable=True),
                          Column('iFPFlux', FLOAT, nullable=True),
                          Column('iFPFluxErr', FLOAT, nullable=True),
                          Column('iFPFluxSigma', FLOAT, nullable=True),
                          Column('zPSFlux', FLOAT, nullable=True),
                          Column('zPSFluxErr', FLOAT, nullable=True),
                          Column('zPSFluxSigma', FLOAT, nullable=True),
                          Column('zFPFlux', FLOAT, nullable=True),
                          Column('zFPFluxErr', FLOAT, nullable=True),
                          Column('zFPFluxSigma', FLOAT, nullable=True),
                          Column('yPSFlux', FLOAT, nullable=True),
                          Column('yPSFluxErr', FLOAT, nullable=True),
                          Column('yPSFluxSigma', FLOAT, nullable=True),
                          Column('yFPFlux', FLOAT, nullable=True),
                          Column('yFPFluxErr', FLOAT, nullable=True),
                          Column('yFPFluxSigma', FLOAT, nullable=True),
                          Column('uLcPeriodic', BLOB, nullable=True),
                          Column('gLcPeriodic', BLOB, nullable=True),
                          Column('rLcPeriodic', BLOB, nullable=True),
                          Column('iLcPeriodic', BLOB, nullable=True),
                          Column('zLcPeriodic', BLOB, nullable=True),
                          Column('yLcPeriodic', BLOB, nullable=True),
                          Column('uLcNonPeriodic', BLOB, nullable=True),
                          Column('gLcNonPeriodic', BLOB, nullable=True),
                          Column('rLcNonPeriodic', BLOB, nullable=True),
                          Column('iLcNonPeriodic', BLOB, nullable=True),
                          Column('zLcNonPeriodic', BLOB, nullable=True),
                          Column('yLcNonPeriodic', BLOB, nullable=True),
                          Column('nearbyObj1', BIGINT, nullable=True),
                          Column('nearbyObj1Dist', FLOAT, nullable=True),
                          Column('nearbyObj1LnP', FLOAT, nullable=True),
                          Column('nearbyObj2', BIGINT, nullable=True),
                          Column('nearbyObj2Dist', FLOAT, nullable=True),
                          Column('nearbyObj2LnP', FLOAT, nullable=True),
                          Column('nearbyObj3', BIGINT, nullable=True),
                          Column('nearbyObj3Dist', FLOAT, nullable=True),
                          Column('nearbyObj3LnP', FLOAT, nullable=True),
                          Column('flags', BIGINT, nullable=False, default=0),
                          Column('htmId20', BIGINT, nullable=False),
                          *constraints,
                          mysql_engine=mysql_engine)

        diaSource = Table('DiaSource', self._metadata,
                          Column('diaSourceId', BIGINT, nullable=False),
                          Column('ccdVisitId', BIGINT, nullable=False),
                          Column('diaObjectId', BIGINT, nullable=True),
                          Column('ssObjectId', BIGINT, nullable=True),
                          Column('parentDiaSourceId', BIGINT, nullable=True),
                          Column('filterName', CHAR(1), nullable=False),
                          Column('prv_procOrder', INT, nullable=False),
                          Column('ssObjectReassocTime', DATETIME, nullable=True),
                          Column('midPointTai', DOUBLE, nullable=False),
                          Column('ra', DOUBLE, nullable=False),
                          Column('raSigma', FLOAT, nullable=False),
                          Column('decl', DOUBLE, nullable=False),
                          Column('declSigma', FLOAT, nullable=False),
                          Column('ra_decl_Cov', FLOAT, nullable=False),
                          Column('x', FLOAT, nullable=False),
                          Column('xSigma', FLOAT, nullable=False),
                          Column('y', FLOAT, nullable=False),
                          Column('ySigma', FLOAT, nullable=False),
                          Column('x_y_Cov', FLOAT, nullable=False),
                          Column('snr', FLOAT, nullable=False),
                          Column('psFlux', FLOAT, nullable=True),
                          Column('psFluxSigma', FLOAT, nullable=True),
                          Column('psLnL', FLOAT, nullable=True),
                          Column('psChi2', FLOAT, nullable=True),
                          Column('psN', INT, nullable=True),
                          Column('trailFlux', FLOAT, nullable=True),
                          Column('trailFluxSigma', FLOAT, nullable=True),
                          Column('trailLength', FLOAT, nullable=True),
                          Column('trailLengthSigma', FLOAT, nullable=True),
                          Column('trailAngle', FLOAT, nullable=True),
                          Column('trailAngleSigma', FLOAT, nullable=True),
                          Column('trailFlux_trailLength_Cov', FLOAT, nullable=True),
                          Column('trailFlux_trailAngle_Cov', FLOAT, nullable=True),
                          Column('trailLength_trailAngle_Cov', FLOAT, nullable=True),
                          Column('trailLnL', FLOAT, nullable=True),
                          Column('trailChi2', FLOAT, nullable=True),
                          Column('trailN', INT, nullable=True),
                          Column('fpFlux', FLOAT, nullable=True),
                          Column('fpFluxSigma', FLOAT, nullable=True),
                          Column('diffFlux', FLOAT, nullable=True),
                          Column('diffFluxSigma', FLOAT, nullable=True),
                          Column('fpSky', FLOAT, nullable=True),
                          Column('fpSkySigma', FLOAT, nullable=True),
                          Column('E1', FLOAT, nullable=True),
                          Column('E1Sigma', FLOAT, nullable=True),
                          Column('E2', FLOAT, nullable=True),
                          Column('E2Sigma', FLOAT, nullable=True),
                          Column('E1_E2_Cov', FLOAT, nullable=True),
                          Column('mSum', FLOAT, nullable=True),
                          Column('mSumSigma', FLOAT, nullable=True),
                          Column('extendedness', FLOAT, nullable=True),
                          Column('apMeanSb01', FLOAT, nullable=True),
                          Column('apMeanSb01Sigma', FLOAT, nullable=True),
                          Column('apMeanSb02', FLOAT, nullable=True),
                          Column('apMeanSb02Sigma', FLOAT, nullable=True),
                          Column('apMeanSb03', FLOAT, nullable=True),
                          Column('apMeanSb03Sigma', FLOAT, nullable=True),
                          Column('apMeanSb04', FLOAT, nullable=True),
                          Column('apMeanSb04Sigma', FLOAT, nullable=True),
                          Column('apMeanSb05', FLOAT, nullable=True),
                          Column('apMeanSb05Sigma', FLOAT, nullable=True),
                          Column('apMeanSb06', FLOAT, nullable=True),
                          Column('apMeanSb06Sigma', FLOAT, nullable=True),
                          Column('apMeanSb07', FLOAT, nullable=True),
                          Column('apMeanSb07Sigma', FLOAT, nullable=True),
                          Column('apMeanSb08', FLOAT, nullable=True),
                          Column('apMeanSb08Sigma', FLOAT, nullable=True),
                          Column('apMeanSb09', FLOAT, nullable=True),
                          Column('apMeanSb09Sigma', FLOAT, nullable=True),
                          Column('apMeanSb10', FLOAT, nullable=True),
                          Column('apMeanSb10Sigma', FLOAT, nullable=True),
                          Column('flags', BIGINT, nullable=False, default=0),
                          Column('htmId20', BIGINT, nullable=False),
                          PrimaryKeyConstraint('diaSourceId', name='PK_DiaSource'),
                          Index('IDX_DiaSource_ccdVisitId', 'ccdVisitId'),
                          Index('IDX_DiaSource_diaObjectId', 'diaObjectId'),
                          Index('IDX_DiaSource_ssObjectId', 'ssObjectId'),
                          Index('IDX_DiaSource_filterName', 'filterName'),
                          Index('IDX_DiaSource_htmId20', 'htmId20'),
                          mysql_engine=mysql_engine)

        ssObject = Table('SSObject', self._metadata,
                         Column('ssObjectId', BIGINT, nullable=False),
                         Column('q', DOUBLE, nullable=True),
                         Column('qSigma', DOUBLE, nullable=True),
                         Column('e', DOUBLE, nullable=True),
                         Column('eSigma', DOUBLE, nullable=True),
                         Column('i', DOUBLE, nullable=True),
                         Column('iSigma', DOUBLE, nullable=True),
                         Column('lan', DOUBLE, nullable=True),
                         Column('lanSigma', DOUBLE, nullable=True),
                         Column('aop', DOUBLE, nullable=True),
                         Column('oepSigma', DOUBLE, nullable=True),
                         Column('M', DOUBLE, nullable=True),
                         Column('MSigma', DOUBLE, nullable=True),
                         Column('epoch', DOUBLE, nullable=True),
                         Column('epochSigma', DOUBLE, nullable=True),
                         Column('q_e_Cov', DOUBLE, nullable=True),
                         Column('q_i_Cov', DOUBLE, nullable=True),
                         Column('q_lan_Cov', DOUBLE, nullable=True),
                         Column('q_aop_Cov', DOUBLE, nullable=True),
                         Column('q_M_Cov', DOUBLE, nullable=True),
                         Column('q_epoch_Cov', DOUBLE, nullable=True),
                         Column('e_i_Cov', DOUBLE, nullable=True),
                         Column('e_lan_Cov', DOUBLE, nullable=True),
                         Column('e_aop_Cov', DOUBLE, nullable=True),
                         Column('e_M_Cov', DOUBLE, nullable=True),
                         Column('e_epoch_Cov', DOUBLE, nullable=True),
                         Column('i_lan_Cov', DOUBLE, nullable=True),
                         Column('i_aop_Cov', DOUBLE, nullable=True),
                         Column('i_M_Cov', DOUBLE, nullable=True),
                         Column('i_epoch_Cov', DOUBLE, nullable=True),
                         Column('lan_aop_Cov', DOUBLE, nullable=True),
                         Column('lan_M_Cov', DOUBLE, nullable=True),
                         Column('lan_epoch_Cov', DOUBLE, nullable=True),
                         Column('aop_M_Cov', DOUBLE, nullable=True),
                         Column('aop_epoch_Cov', DOUBLE, nullable=True),
                         Column('M_epoch_Cov', DOUBLE, nullable=True),
                         Column('arc', FLOAT, nullable=True),
                         Column('orbFitLnL', FLOAT, nullable=True),
                         Column('orbFitChi2', FLOAT, nullable=True),
                         Column('orbFitN', INTEGER, nullable=True),
                         Column('MOID1', FLOAT, nullable=True),
                         Column('MOID2', FLOAT, nullable=True),
                         Column('moidLon1', DOUBLE, nullable=True),
                         Column('moidLon2', DOUBLE, nullable=True),
                         Column('uH', FLOAT, nullable=True),
                         Column('uHSigma', FLOAT, nullable=True),
                         Column('uG1', FLOAT, nullable=True),
                         Column('uG1Sigma', FLOAT, nullable=True),
                         Column('uG2', FLOAT, nullable=True),
                         Column('uG2Sigma', FLOAT, nullable=True),
                         Column('gH', FLOAT, nullable=True),
                         Column('gHSigma', FLOAT, nullable=True),
                         Column('gG1', FLOAT, nullable=True),
                         Column('gG1Sigma', FLOAT, nullable=True),
                         Column('gG2', FLOAT, nullable=True),
                         Column('gG2Sigma', FLOAT, nullable=True),
                         Column('rH', FLOAT, nullable=True),
                         Column('rHSigma', FLOAT, nullable=True),
                         Column('rG1', FLOAT, nullable=True),
                         Column('rG1Sigma', FLOAT, nullable=True),
                         Column('rG2', FLOAT, nullable=True),
                         Column('rG2Sigma', FLOAT, nullable=True),
                         Column('iH', FLOAT, nullable=True),
                         Column('iHSigma', FLOAT, nullable=True),
                         Column('iG1', FLOAT, nullable=True),
                         Column('iG1Sigma', FLOAT, nullable=True),
                         Column('iG2', FLOAT, nullable=True),
                         Column('iG2Sigma', FLOAT, nullable=True),
                         Column('zH', FLOAT, nullable=True),
                         Column('zHSigma', FLOAT, nullable=True),
                         Column('zG1', FLOAT, nullable=True),
                         Column('zG1Sigma', FLOAT, nullable=True),
                         Column('zG2', FLOAT, nullable=True),
                         Column('zG2Sigma', FLOAT, nullable=True),
                         Column('yH', FLOAT, nullable=True),
                         Column('yHSigma', FLOAT, nullable=True),
                         Column('yG1', FLOAT, nullable=True),
                         Column('yG1Sigma', FLOAT, nullable=True),
                         Column('yG2', FLOAT, nullable=True),
                         Column('yG2Sigma', FLOAT, nullable=True),
                         Column('flags', BIGINT, nullable=False, default=0),
                         PrimaryKeyConstraint('ssObjectId', name='PK_SSObject'),
                         mysql_engine=mysql_engine)

        diaForcedSource = Table('DiaForcedSource', self._metadata,
                                Column('diaObjectId', BIGINT, nullable=False),
                                Column('ccdVisitId', BIGINT, nullable=False),
                                Column('psFlux', FLOAT, nullable=False),
                                Column('psFlux_Sigma', FLOAT, nullable=True),
                                Column('x', FLOAT, nullable=False),
                                Column('y', FLOAT, nullable=False),
                                Column('flags', BIGINT, nullable=False, default=0),
                                PrimaryKeyConstraint('diaObjectId', 'ccdVisitId', name='PK_DiaForcedSource'),
                                Index('IDX_DiaForcedSource_ccdVisitId', 'ccdVisitId'),
                                mysql_engine=mysql_engine)

        o2oMatch = Table('DiaObject_To_Object_Match', self._metadata,
                         Column('diaObjectId', BIGINT, nullable=False),
                         Column('objectId', BIGINT, nullable=False),
                         Column('dist', FLOAT, nullable=False),
                         Column('lnP', FLOAT, nullable=True),
                         Index('IDX_DiaObjectToObjectMatch_diaObjectId', 'diaObjectId'),
                         Index('IDX_DiaObjectToObjectMatch_objectId', 'objectId'),
                         mysql_engine=mysql_engine)

        # special table to track visits, only used by prototype
        visits = Table('L1DbProtoVisits', self._metadata,
                       Column('visitId', BIGINT, nullable=False),
                       Column('visitTime', DATETIME, nullable=False),
                       PrimaryKeyConstraint('visitId', name='PK_L1DbProtoVisits'),
                       Index('IDX_L1DbProtoVisits_visitTime', 'visitTime'),
                       mysql_engine=mysql_engine)

        # create all tables (optionally drop first)
        if drop:
            _LOG.info('dropping all tables')
            self._metadata.drop_all()
        _LOG.info('creating all tables')
        self._metadata.create_all()

    def _make_doube_type(self):
        """
        DOUBLE type is database-specific, select one based on dialect.
        """
        if self._engine.name == 'mysql':
            from sqlalchemy.dialects.mysql import DOUBLE
            return DOUBLE(asdecimal=False)
        elif self._engine.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
            return DOUBLE_PRECISION
        else:
            raise TypeError('cannot determine DOUBLE type, unexpected dialect: ' + self._engine.name)

    @property
    def _visits(self):
        """ Lazy reading of table schema """
        return self._table_schema('L1DbProtoVisits')

    @property
    def _objects(self):
        """ Lazy reading of table schema """
        return self._table_schema('DiaObject')

    @property
    def _sources(self):
        """ Lazy reading of table schema """
        return self._table_schema('DiaSource')

    @property
    def _forcedSources(self):
        """ Lazy reading of table schema """
        return self._table_schema('DiaForcedSource')

    def _table_schema(self, name):
        """ Lazy reading of table schema """
        table = self._tables.get(name)
        if table is None:
            table = Table(name, self._metadata, autoload=True)
            self._tables[name] = table
            _LOG.debug("read table schema for %s: %s", name, table.c)
        return table

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

    def _storeObjects(self, object_type, objects, conn, table, explain=False):
        """
        Generic store method.

        Stores a bunch of objects as records in a table.
        @param object_type: Namedtuple type, e.g. DiaSource
        @param objects:     Sequence of objects of the `object_type` type
        @param conn:        Database connection
        @param table:       Database table
        @param explain:     If True then do EXPLAIN on INSERT query
        """

        def quoteValue(v):
            if v is None:
                v = "NULL"
            elif isinstance(v, datetime):
                v = "'" + str(v) + "'"
            elif isinstance(v, basestring):
                # we don't expect nasty stuff in strings
                v = "'" + v + "'"
            else:
                # assume numeric stuff
                v = str(v)
            return v

        fields = ['"' + f + '"' for f in object_type._fields]

        query = 'INSERT INTO "' + table.name + '" (' + ','.join(fields) + ') VALUES '

        values = []
        for obj in objects:
            row = [quoteValue(v) for v in obj]
            values.append('(' + ','.join(row) + ')')

        if explain:
            # run the same query with explain, only give it one row of data
            self._explain(query + values[0], conn)

        query += ','.join(values)

        # _LOG.debug("query: %s", query)
        with Timer(table.name + ' insert'):
            res = conn.execute(sql.text(query))
        _LOG.debug("inserted %s intervals", res.rowcount)
