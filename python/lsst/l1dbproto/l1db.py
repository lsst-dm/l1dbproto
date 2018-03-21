"""
Module defining L1db class and related methods.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime
import logging
from configparser import ConfigParser, NoSectionError

#-----------------------------
# Imports for other modules --
#-----------------------------
import lsst.afw.table as afwTable
import sqlalchemy
from sqlalchemy import (func, sql)
from sqlalchemy.pool import NullPool
from . import timer, l1dbschema

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
        """Start counting"""
        if self._log_before_cursor_execute:
            _LOG.info("before_cursor_execute")
        self._timer2.start()

    def _stop_timer(self, conn, cursor, statement, parameters, context, executemany):
        """Stop counting"""
        self._timer2.stop()
        self._timer2.dump()

#------------------------
# Exported definitions --
#------------------------

# Information about single visit
Visit = namedtuple('Visit', 'visitId visitTime lastObjectId lastSourceId')

@contextmanager
def _ansi_session(engine):
    """Returns a connection, makes sure that ANSI mode is set for MySQL
    """
    with engine.begin() as conn:
        if engine.name == 'mysql':
            conn.execute(sql.text("SET SESSION SQL_MODE = 'ANSI'"))
        yield conn
    return

#---------------------
#  Class definition --
#---------------------


class L1db(object):
    """Interface to L1 database, hides all database access details.

    The implementation is configured via configuration file (to simplify
    prototyping) which is an INI-format file with all parameters. For an
    example check cfg/ folder.

    Parameters
    ----------
    config : str
        Name of the configuration file.
    afw_schemas : `dict`, optional
        Dictionary with table name for a key and `afw.table.Schema`
        for a value. Columns in schema will be added to standard
        L1DB schema.
    """

    def __init__(self, config, afw_schemas=None):

        parser = ConfigParser()
        with open(config) as cfgFile:
            parser.read_file(cfgFile, config)

        # logging.getLogger('sqlalchemy').setLevel(logging.INFO)

        # engine is reused between multiple processes, make sure that we don't
        # share connections by disabling pool (by using NullPool class)
        options = dict(parser.items("database"))
        isolation_level = options.get('isolation_level', 'READ_COMMITTED')
        self._engine = sqlalchemy.create_engine(options['url'], poolclass=NullPool,
                                                isolation_level=isolation_level)

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
        self._object_last_replace = bool(int(options.get('object_last_replace', 0)))
        self._do_explain = bool(int(options.get('explain', 0)))
        schema_file = options.get('schema_file')
        extra_schema_file = options.get('extra_schema_file')
        column_map = options.get('column_map')

        if self._dia_object_index not in ('baseline', 'htm20_id_iov', 'last_object_table'):
            raise ValueError('unexpected dia_object_index value: ' + str(self._dia_object_index))

        self._schema = l1dbschema.L1dbSchema(engine=self._engine,
                                             dia_object_index=self._dia_object_index,
                                             dia_object_nightly=self._dia_object_nightly,
                                             schema_file=schema_file,
                                             extra_schema_file=extra_schema_file,
                                             column_map=column_map,
                                             afw_schemas=afw_schemas)

        _LOG.info("L1DB Configuration:")
        _LOG.info("    dia_object_index: %s", self._dia_object_index)
        _LOG.info("    dia_object_nightly: %s", self._dia_object_nightly)
        _LOG.info("    read_sources_months: %s", self._months_sources)
        _LOG.info("    read_forced_sources_months: %s", self._months_fsources)
        _LOG.info("    read_full_objects: %s", self._read_full_objects)
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
            _LOG.info("lastVisit: visitId: %s visitTime: %s (%s)", visitId,
                      visitTime, type(visitTime))

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
        visitId : `int`
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
        Dict where key is a table name and value is a row count.
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

    def getDiaObjects(self, pixel_ranges):
        """Returns catalog of DiaObject instances from given region.

        Objects are searched based on pixelization index and region is
        determined by the set of indices. There is no assumption on a
        particular type of index, client is responsible for consistency
        when calculating pixelization indices.

        This methods returns `afw.table` catalog with schema determined by
        the schema of L1 database table. Re-mapping of the column names is
        done for some columns (based on column map passed to constructor)
        but types or units are not changed.

        Returns only the last version of the the DiaObject.

        Parameters
        ----------
        pixel_ranges: `list` of `tuple`
            Sequence of ranges, range is a tuple (minPixelID, maxPixelID).
            This defines set of pixel indices to be included in result.

        Returns
        -------
        `afw.table.BaseCatalog` instance
        """

        # decide what columns we need
        if self._dia_object_index == 'last_object_table':
            table = self._schema.objects_last
        else:
            table = self._schema.objects
        if self._read_full_objects:
            query = table.select()
        else:
            columns = [table.c.diaObjectId, table.c.lastNonForcedSource,
                       table.c.ra, table.c.decl, table.c.raSigma, table.c.declSigma,
                       table.c.ra_decl_Cov, table.c.pixelId]
            query = sql.select(columns)

        # build selection
        exprlist = []
        for low, upper in pixel_ranges:
            upper -= 1
            if low == upper:
                exprlist.append(table.c.pixelId == low)
            else:
                exprlist.append(sql.expression.between(table.c.pixelId, low, upper))
        query = query.where(sql.expression.or_(*exprlist))

        # select latest version of objects
        if self._dia_object_index != 'last_object_table':
            query = query.where(table.c.validityEnd == None)

        _LOG.debug("query: %s", query)

        if self._do_explain:
            # run the same query with explain
            self._explain(query, self._engine)

        # execute select
        with Timer('DiaObject select'):
            with self._engine.begin() as conn:
                res = conn.execute(query)
                objects = self._convertResult(res, "DiaObject")
        _LOG.debug("found %s DiaObjects", len(objects))
        return objects

    def getDiaSourcesInRegion(self, pixel_ranges, objects):
        """Returns catalog of DiaSource instances from given region.

        Sources are searched based on pixelization index and region is
        determined by the set of indices. There is no assumption on a
        particular type of index, client is responsible for consistency
        when calculating pixelization indices.

        This methods returns `afw.table` catalog with schema determined by
        the schema of L1 database table. Re-mapping of the column names is
        done for some columns (based on column map passed to constructor)
        but types or units are not changed.

        Parameters
        ----------
        pixel_ranges: `list` of `tuple`
            Sequence of ranges, range is a tuple (minPixelID, maxPixelID).
            This defines set of pixel indices to be included in result.

        Returns
        -------
        `afw.table.BaseCatalog` instance
        """

        if self._months_sources == 0:
            _LOG.info("Skip DiaSources fetching")
            return []

        table = self._schema.sources
        query = 'SELECT *  FROM "' + table.name + '" WHERE '

        # build selection
        exprlist = []
        for low, upper in pixel_ranges:
            upper -= 1
            if low == upper:
                exprlist.append('"pixelId" = ' + str(low))
            else:
                exprlist.append('"pixelId" BETWEEN {} AND {}'.format(low, upper))
        query += '(' + ' OR '.join(exprlist) + ')'

        # execute select
        with Timer('DiaSource select'):
            with _ansi_session(self._engine) as conn:
                res = conn.execute(sql.text(query))
                sources = self._convertResult(res, "DiaSource")
        _LOG.debug("found %s DiaSources", len(sources))
        return sources

    def getDiaSources(self, object_ids):
        """Returns catalog of DiaSource instances given set of DiaObject IDs.

        This methods returns `afw.table` catalog with schema determined by
        the schema of L1 database table. Re-mapping of the column names is
        done for some columns (based on column map passed to constructor)
        but types or units are not changed.

        Parameters
        ----------
        object_ids :
            Collection of DiaObject IDs

        Returns
        -------
        `afw.table.BaseCatalog` instance
        """

        if self._months_sources == 0:
            _LOG.info("Skip DiaSources fetching")
            return []

        if not object_ids:
            _LOG.info("Skip DiaSources fetching - no Objects")
            return []

        table = self._schema.sources
        query = 'SELECT *  FROM "' + table.name + '" WHERE '

        # select by object id
        ids = sorted(object_ids)
        ids = ",".join(str(id) for id in ids)
        query += '"diaObjectId" IN (' + ids + ') '

        # execute select
        with Timer('DiaSource select'):
            with _ansi_session(self._engine) as conn:
                res = conn.execute(sql.text(query))
                sources = self._convertResult(res, "DiaSource")
        _LOG.debug("found %s DiaSources", len(sources))
        return sources

    def getDiaForcedSources(self, object_ids):
        """Returns catalog of DiaForceSource instances matching given
        DiaObjects.

        This methods returns `afw.table` catalog with schema determined by
        the schema of L1 database table. Re-mapping of the column names may
        be done for some columns (based on column map passed to constructor)
        but types or units are not changed.

        Parameters
        ----------
        object_ids :
            Collection of DiaObject IDs

        Returns
        -------
        `afw.table.BaseCatalog` instance
        """

        if self._months_fsources == 0:
            _LOG.info("Skip DiaForceSources fetching")
            return []

        if not object_ids:
            _LOG.info("Skip DiaForceSources fetching - no Objects")
            return []

        table = self._schema.forcedSources
        query = 'SELECT *  FROM "' + table.name + '" WHERE '

        # select by object id
        ids = sorted(object_ids)
        ids = ",".join(str(id) for id in ids)
        query += '"diaObjectId" IN (' + ids + ') '

        # execute select
        with Timer('DiaForcedSource select'):
            with _ansi_session(self._engine) as conn:
                res = conn.execute(sql.text(query))
                sources = self._convertResult(res, "DiaForcedSource")
        _LOG.debug("found %s DiaForcedSources", len(sources))
        return sources

    def storeDiaObjects(self, objs, dt):
        """Store catalog of DiaObjects from current visit.

        This methods takes `afw.table` catalog, its schema must be
        compatible with the schema of L1 database table:
          - column names must correspond to database table columns
          - some columns names are re-mapped based on column map passed to
            constructor
          - types and units of the columns must match database definitions,
            no unit conversion is performed presently
          - columns that have default values in database schema can be
            omitted from afw schema
          - this method knows how to fill interval-related columns
            (validityStart, validityEnd) they do not need to appear in
            afw schema

        Parameters
        ----------
        objs : `afw.table.BaseCatalog`
            Catalog with DiaObject records
        dt : `datetime.datetime`
            Time of the visit
        """

        ids = sorted([obj['id'] for obj in objs])
        _LOG.info("first object ID: %d", ids[0])

        # NOTE: workaround for sqlite, need this here to avoid
        # "database is locked" error.
        table = self._schema.objects

        # everything to be done in single transaction
        with _ansi_session(self._engine) as conn:

            ids = ",".join(str(id) for id in ids)

            if self._dia_object_index == 'last_object_table':

                # insert and replace all records in LAST table, mysql and postgres have
                # non-standard features (handled in _storeObjectsAfw)
                table = self._schema.objects_last
                do_replace = self._object_last_replace
                if not do_replace:
                    query = 'DELETE FROM "' + table.name + '" '
                    query += 'WHERE "diaObjectId" IN (' + ids + ') '

                    if self._do_explain:
                        # run the same query with explain
                        self._explain(query, conn)

                    with Timer(table.name + ' delete'):
                        res = conn.execute(sql.text(query))
                    _LOG.debug("deleted %s objects", res.rowcount)

                extra_columns = dict(lastNonForcedSource=dt, validityStart=dt,
                                     validityEnd=None)
                self._storeObjectsAfw(objs, conn, table, "DiaObject",
                                      replace=do_replace,
                                      extra_columns=extra_columns)

            else:

                # truncate existing validity intervals
                table = self._schema.objects
                query = 'UPDATE "' + table.name + '" '
                query += "SET \"validityEnd\" = '" + str(dt) + "' "
                query += 'WHERE "diaObjectId" IN (' + ids + ') '
                query += 'AND "validityEnd" IS NULL'

                # _LOG.debug("query: %s", query)

                if self._do_explain:
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
            extra_columns = dict(lastNonForcedSource=dt, validityStart=dt,
                                 validityEnd=None)
            self._storeObjectsAfw(objs, conn, table, "DiaObject",
                                  extra_columns=extra_columns)

    def storeDiaSources(self, sources):
        """Store catalog of DIASources from current visit.

        This methods takes `afw.table` catalog, its schema must be
        compatible with the schema of L1 database table:
          - column names must correspond to database table columns
          - some columns names may be re-mapped based on column map passed to
            constructor
          - types and units of the columns must match database definitions,
            no unit conversion is performed presently
          - columns that have default values in database schema can be
            omitted from afw schema

        Parameters
        ----------
        sources : `afw.table.BaseCatalog`
            Catalog containing DiaSource records
        """

        # everything to be done in single transaction
        with _ansi_session(self._engine) as conn:

            table = self._schema.sources
            self._storeObjectsAfw(sources, conn, table, "DiaSource")

    def storeDiaForcedSources(self, sources):
        """Store a set of DIAForcedSources from current visit.

        This methods takes `afw.table` catalog, its schema must be
        compatible with the schema of L1 database table:
          - column names must correspond to database table columns
          - some columns names may be re-mapped based on column map passed to
            constructor
          - types and units of the columns must match database definitions,
            no unit conversion is performed presently
          - columns that have default values in database schema can be
            omitted from afw schema

        Parameters
        ----------
        sources : `afw.table.BaseCatalog`
            Catalog containing DiaForcedSource records
        """

        # everything to be done in single transaction
        with _ansi_session(self._engine) as conn:

            table = self._schema.forcedSources
            self._storeObjectsAfw(sources, conn, table, "DiaForcedSource")

    def dailyJob(self):
        """Implement daily activities like cleanup/vacuum.

        What should be done during daily cleanup is determined by
        configuration/schema.
        """

        # move data from DiaObjectNightly into DiaObject
        if self._dia_object_nightly:
            with _ansi_session(self._engine) as conn:
                query = 'INSERT INTO "' + self._schema.objects.name + '" '
                query += 'SELECT * FROM "' + self._schema.objects_nightly.name + '"'
                with Timer('DiaObjectNightly copy'):
                    conn.execute(sql.text(query))

                query = 'DELETE FROM "' + self._schema.objects_nightly.name + '"'
                with Timer('DiaObjectNightly delete'):
                    conn.execute(sql.text(query))

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
        drop : `boolean`
            If True then drop tables before creating new ones.
        mysql_engine : `str`, optional
            Name of the MySQL engine to use for new tables.
        """
        self._schema.makeSchema(drop=drop, mysql_engine=mysql_engine)

    def _explain(self, query, conn):
        """Run the query with explain
        """

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

    def _storeObjectsAfw(self, objects, conn, table, schema_table_name,
                         replace=False, extra_columns=None):
        """Generic store method.

        Takes catalog of records and stores a bunch of objects in a table.

        Parameters
        ----------
        objects : `afw.table.BaseCatalog`
            Catalog containing object records
        conn :
            Database connection
        table : `sqlalchemy.Table`
            Database table
        schema_table_name : `str`
            Name of the table to be used for finding table schema.
        replace : `boolean`
            If `True` then use replace instead of INSERT (should be more efficient)
        extra_columns : `dict`, optional
            Mapping (column_name, column_value) which gives column values to add
            to every row, only if column is missing in catalog records.
        """

        def quoteValue(v):
            """Quote and escape values"""
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

        schema = objects.getSchema()
        afw_fields = [field.getName() for key, field in schema]

        column_map = self._schema.getAfwColumns(schema_table_name)

        # list of columns (as in cat schema)
        fields = ['"' + column_map[field].name + '"' for field in afw_fields]

        # use extra columns that are not in fields already
        extra_fields = (extra_columns or {}).keys()
        extra_fields = [field for field in extra_fields if field not in fields]

        if replace and conn.engine.name in ('mysql', 'sqlite'):
            query = 'REPLACE INTO'
        else:
            query = 'INSERT INTO'
        query += ' "' + table.name + '" (' + ','.join(fields + extra_fields) + ') VALUES '

        values = []
        for rec in objects:
            row = []
            for field in afw_fields:
                value = rec[field]
                if column_map[field].type == "DATETIME":
                    # convert seconds into datetime
                    value = datetime.utcfromtimestamp(value)
                row.append(quoteValue(value))
            for field in extra_fields:
                row.append(quoteValue(extra_columns[field]))
            values.append('(' + ','.join(row) + ')')

        if self._do_explain:
            # run the same query with explain, only give it one row of data
            self._explain(query + values[0], conn)

        query += ','.join(values)

        if replace and conn.engine.name == 'postgresql':
            # This depends on that "replace" can only be true for DiaObjectLast table
            pks = ('pixelId', 'diaObjectId')
            query += " ON CONFLICT (\"{}\", \"{}\") DO UPDATE SET ".format(*pks)
            fields = [column_map[field].name for field in afw_fields]
            fields = ['"{0}" = EXCLUDED."{0}"'.format(field)
                      for field in fields if field not in pks]
            query += ', '.join(fields)

        # _LOG.debug("query: %s", query)
        _LOG.info("%s: will store %d records", table.name, len(objects))
        with Timer(table.name + ' insert'):
            res = conn.execute(sql.text(query))
        _LOG.debug("inserted %s intervals", res.rowcount)

    def _convertResult(self, res, table_name):
        """Convert result set into output catalog.

        Parameters
        ----------
        res : `sqlalchemy.ResultProxy`
            SQLAlchemy result set returned by query.
        table_name : `str`
            Name of the table.

        Returns
        -------
        `afw.table.BaseCatalog` instance
        """
        # make catalog schema
        columns = res.keys()
        schema, col_map = self._schema.getAfwSchema(table_name, columns)
        _LOG.debug("_convertResult: schema: %s", schema)
        _LOG.debug("_convertResult: col_map: %s", col_map)

        # fill catalog
        catalog = afwTable.BaseCatalog(schema)
        for row in res:
            record = catalog.addNew()
            for col, value in row.items():
                if isinstance(value, datetime):
                    # convert datetime to number of seconds
                    value = int((value - datetime.utcfromtimestamp(0)).total_seconds())
                if value is not None:
                    record.set(col_map[col], value)

        return catalog
