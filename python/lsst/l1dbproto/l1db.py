"""
Module defining L1db class and related methods.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
from collections import namedtuple
import logging
import sys

#-----------------------------
# Imports for other modules --
#-----------------------------
from lsst.db import engineFactory
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, PrimaryKeyConstraint, Index, sql

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_LOG = logging.getLogger(__name__)

#------------------------
# Exported definitions --
#------------------------

Visit = namedtuple('Visit', 'visitId visitTime')

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

    #-------------------
    #  Public methods --
    #-------------------

    def lastVisit(self):
        """
        Returns last visit information or None if visits table is empty.

        @return instance of Visit class or None
        """

        stmt = sql.select([sql.func.max(self._visits.c.visitId),
                           sql.func.max(self._visits.c.visitTime)])
        res = self._engine.execute(stmt)
        row = res.fetchone()
        if row[0] is None:
            return None
        else:
            return Visit(visitId=row[0], visitTime=row[1])


    def saveVisit(self, visitId, visitTime):
        """
        Store visit information.
        """

        ins = self._visits.insert().values(visitId=visitId, visitTime=visitTime)
        self._engine.execute(ins)


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
        BLOB = sqlalchemy.types.BLOB
        CHAR = sqlalchemy.types.CHAR

        diaObject = Table('DiaObject', self._metadata,
            Column('diaObjectId', BIGINT, nullable=False),
            Column('validityStart', DATETIME, nullable=False),
            Column('validityEnd', DATETIME, nullable=False),
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
            PrimaryKeyConstraint('diaObjectId', 'validityStart', name='PK_DiaObject'),
            Index('IDX_DiaObject_validityStart', 'validityStart'),
            Index('IDX_DiaObject_htmId20', 'htmId20'),
            mysql_engine=mysql_engine)

        diaSource = Table('DiaSource', self._metadata,
            Column('diaSourceId', BIGINT , nullable=False),
            Column('ccdVisitId', BIGINT , nullable=False),
            Column('diaObjectId', BIGINT , nullable=True),
            Column('ssObjectId', BIGINT , nullable=True),
            Column('parentDiaSourceId', BIGINT , nullable=True),
            Column('filterName', CHAR(1) , nullable=False),
            Column('prv_procOrder', INT , nullable=False),
            Column('ssObjectReassocTime', DATETIME , nullable=True),
            Column('midPointTai', DOUBLE , nullable=False),
            Column('ra', DOUBLE , nullable=False),
            Column('raSigma', FLOAT , nullable=False),
            Column('decl', DOUBLE , nullable=False),
            Column('declSigma', FLOAT , nullable=False),
            Column('ra_decl_Cov', FLOAT , nullable=False),
            Column('x', FLOAT , nullable=False),
            Column('xSigma', FLOAT , nullable=False),
            Column('y', FLOAT , nullable=False),
            Column('ySigma', FLOAT , nullable=False),
            Column('x_y_Cov', FLOAT , nullable=False),
            Column('snr', FLOAT , nullable=False),
            Column('psFlux', FLOAT , nullable=True),
            Column('psFluxSigma', FLOAT , nullable=True),
            Column('psLnL', FLOAT , nullable=True),
            Column('psChi2', FLOAT , nullable=True),
            Column('psN', INT , nullable=True),
            Column('trailFlux', FLOAT , nullable=True),
            Column('trailFluxSigma', FLOAT , nullable=True),
            Column('trailLength', FLOAT , nullable=True),
            Column('trailLengthSigma', FLOAT , nullable=True),
            Column('trailAngle', FLOAT , nullable=True),
            Column('trailAngleSigma', FLOAT , nullable=True),
            Column('trailFlux_trailLength_Cov', FLOAT , nullable=True),
            Column('trailFlux_trailAngle_Cov', FLOAT , nullable=True),
            Column('trailLength_trailAngle_Cov', FLOAT , nullable=True),
            Column('trailLnL', FLOAT , nullable=True),
            Column('trailChi2', FLOAT , nullable=True),
            Column('trailN', INT , nullable=True),
            Column('fpFlux', FLOAT , nullable=True),
            Column('fpFluxSigma', FLOAT , nullable=True),
            Column('diffFlux', FLOAT , nullable=True),
            Column('diffFluxSigma', FLOAT , nullable=True),
            Column('fpSky', FLOAT , nullable=True),
            Column('fpSkySigma', FLOAT , nullable=True),
            Column('E1', FLOAT , nullable=True),
            Column('E1Sigma', FLOAT , nullable=True),
            Column('E2', FLOAT , nullable=True),
            Column('E2Sigma', FLOAT , nullable=True),
            Column('E1_E2_Cov', FLOAT , nullable=True),
            Column('mSum', FLOAT , nullable=True),
            Column('mSumSigma', FLOAT , nullable=True),
            Column('extendedness', FLOAT , nullable=True),
            Column('apMeanSb01', FLOAT , nullable=True),
            Column('apMeanSb01Sigma', FLOAT , nullable=True),
            Column('apMeanSb02', FLOAT , nullable=True),
            Column('apMeanSb02Sigma', FLOAT , nullable=True),
            Column('apMeanSb03', FLOAT , nullable=True),
            Column('apMeanSb03Sigma', FLOAT , nullable=True),
            Column('apMeanSb04', FLOAT , nullable=True),
            Column('apMeanSb04Sigma', FLOAT , nullable=True),
            Column('apMeanSb05', FLOAT , nullable=True),
            Column('apMeanSb05Sigma', FLOAT , nullable=True),
            Column('apMeanSb06', FLOAT , nullable=True),
            Column('apMeanSb06Sigma', FLOAT , nullable=True),
            Column('apMeanSb07', FLOAT , nullable=True),
            Column('apMeanSb07Sigma', FLOAT , nullable=True),
            Column('apMeanSb08', FLOAT , nullable=True),
            Column('apMeanSb08Sigma', FLOAT , nullable=True),
            Column('apMeanSb09', FLOAT , nullable=True),
            Column('apMeanSb09Sigma', FLOAT , nullable=True),
            Column('apMeanSb10', FLOAT , nullable=True),
            Column('apMeanSb10Sigma', FLOAT , nullable=True),
            Column('flags', BIGINT , nullable=False, default=0),
            Column('htmId20', BIGINT , nullable=False),
            PrimaryKeyConstraint('diaSourceId', name='PK_DiaSource'),
            Index('IDX_DiaSource_ccdVisitId', 'ccdVisitId'),
            Index('IDX_DiaSource_diaObjectId', 'diaObjectId'),
            Index('IDX_DiaSource_ssObjectId', 'ssObjectId'),
            Index('IDX_DiaSource_filterName', 'filterName'),
            Index('IDX_DiaObject_htmId20', 'htmId20'),
            mysql_engine=mysql_engine)

        ssObject = Table('SSObject', self._metadata,
            Column('ssObjectId', BIGINT , nullable=False),
            Column('q', DOUBLE , nullable=True),
            Column('qSigma', DOUBLE , nullable=True),
            Column('e', DOUBLE , nullable=True),
            Column('eSigma', DOUBLE , nullable=True),
            Column('i', DOUBLE , nullable=True),
            Column('iSigma', DOUBLE , nullable=True),
            Column('lan', DOUBLE , nullable=True),
            Column('lanSigma', DOUBLE , nullable=True),
            Column('aop', DOUBLE , nullable=True),
            Column('oepSigma', DOUBLE , nullable=True),
            Column('M', DOUBLE , nullable=True),
            Column('MSigma', DOUBLE , nullable=True),
            Column('epoch', DOUBLE , nullable=True),
            Column('epochSigma', DOUBLE , nullable=True),
            Column('q_e_Cov', DOUBLE , nullable=True),
            Column('q_i_Cov', DOUBLE , nullable=True),
            Column('q_lan_Cov', DOUBLE , nullable=True),
            Column('q_aop_Cov', DOUBLE , nullable=True),
            Column('q_M_Cov', DOUBLE , nullable=True),
            Column('q_epoch_Cov', DOUBLE , nullable=True),
            Column('e_i_Cov', DOUBLE , nullable=True),
            Column('e_lan_Cov', DOUBLE , nullable=True),
            Column('e_aop_Cov', DOUBLE , nullable=True),
            Column('e_M_Cov', DOUBLE , nullable=True),
            Column('e_epoch_Cov', DOUBLE , nullable=True),
            Column('i_lan_Cov', DOUBLE , nullable=True),
            Column('i_aop_Cov', DOUBLE , nullable=True),
            Column('i_M_Cov', DOUBLE , nullable=True),
            Column('i_epoch_Cov', DOUBLE , nullable=True),
            Column('lan_aop_Cov', DOUBLE , nullable=True),
            Column('lan_M_Cov', DOUBLE , nullable=True),
            Column('lan_epoch_Cov', DOUBLE , nullable=True),
            Column('aop_M_Cov', DOUBLE , nullable=True),
            Column('aop_epoch_Cov', DOUBLE , nullable=True),
            Column('M_epoch_Cov', DOUBLE , nullable=True),
            Column('arc', FLOAT , nullable=True),
            Column('orbFitLnL', FLOAT , nullable=True),
            Column('orbFitChi2', FLOAT , nullable=True),
            Column('orbFitN', INTEGER , nullable=True),
            Column('MOID1', FLOAT , nullable=True),
            Column('MOID2', FLOAT , nullable=True),
            Column('moidLon1', DOUBLE , nullable=True),
            Column('moidLon2', DOUBLE , nullable=True),
            Column('uH', FLOAT , nullable=True),
            Column('uHSigma', FLOAT , nullable=True),
            Column('uG1', FLOAT , nullable=True),
            Column('uG1Sigma', FLOAT , nullable=True),
            Column('uG2', FLOAT , nullable=True),
            Column('uG2Sigma', FLOAT , nullable=True),
            Column('gH', FLOAT , nullable=True),
            Column('gHSigma', FLOAT , nullable=True),
            Column('gG1', FLOAT , nullable=True),
            Column('gG1Sigma', FLOAT , nullable=True),
            Column('gG2', FLOAT , nullable=True),
            Column('gG2Sigma', FLOAT , nullable=True),
            Column('rH', FLOAT , nullable=True),
            Column('rHSigma', FLOAT , nullable=True),
            Column('rG1', FLOAT , nullable=True),
            Column('rG1Sigma', FLOAT , nullable=True),
            Column('rG2', FLOAT , nullable=True),
            Column('rG2Sigma', FLOAT , nullable=True),
            Column('iH', FLOAT , nullable=True),
            Column('iHSigma', FLOAT , nullable=True),
            Column('iG1', FLOAT , nullable=True),
            Column('iG1Sigma', FLOAT , nullable=True),
            Column('iG2', FLOAT , nullable=True),
            Column('iG2Sigma', FLOAT , nullable=True),
            Column('zH', FLOAT , nullable=True),
            Column('zHSigma', FLOAT , nullable=True),
            Column('zG1', FLOAT , nullable=True),
            Column('zG1Sigma', FLOAT , nullable=True),
            Column('zG2', FLOAT , nullable=True),
            Column('zG2Sigma', FLOAT , nullable=True),
            Column('yH', FLOAT , nullable=True),
            Column('yHSigma', FLOAT , nullable=True),
            Column('yG1', FLOAT , nullable=True),
            Column('yG1Sigma', FLOAT , nullable=True),
            Column('yG2', FLOAT , nullable=True),
            Column('yG2Sigma', FLOAT , nullable=True),
            Column('flags', BIGINT , nullable=False, default=0),
            PrimaryKeyConstraint('ssObjectId', name='PK_SSObject'),
            mysql_engine=mysql_engine)

        diaForcedSource = Table('DiaForcedSource', self._metadata,
            Column('diaObjectId', BIGINT , nullable=False),
            Column('ccdVisitId', BIGINT , nullable=False),
            Column('psFlux', FLOAT , nullable=False),
            Column('psFlux_Sigma', FLOAT , nullable=True),
            Column('x', FLOAT , nullable=False),
            Column('y', FLOAT , nullable=False),
            Column('flags', BIGINT , nullable=False, default=0),
            PrimaryKeyConstraint('diaObjectId', 'ccdVisitId', name='PK_DiaForcedSource'),
            Index('IDX_DiaForcedSource_ccdVisitId', 'ccdVisitId'),
            mysql_engine=mysql_engine)

        o2oMatch = Table('DiaObject_To_Object_Match', self._metadata,
            Column('diaObjectId', BIGINT , nullable=False),
            Column('objectId', BIGINT , nullable=False),
            Column('dist', FLOAT , nullable=False),
            Column('lnP', FLOAT , nullable=True),
            Index('IDX_DiaObjectToObjectMatch_diaObjectId', 'diaObjectId'),
            Index('IDX_DiaObjectToObjectMatch_objectId', 'objectId'),
            mysql_engine=mysql_engine)

        # special table to track visits, only used by prototype
        visits = Table('L1DbProtoVisits', self._metadata,
            Column('visitId', BIGINT , nullable=False),
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


    def _table_schema(self, name):
        """ Lazy reading of table schema """
        table = self._tables.get(name)
        if table is None:
            table = Table(name, self._metadata, autoload=True)
            self._tables[name] = table
        return table
