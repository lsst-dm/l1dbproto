"""Module responsible for L1DB schema operations.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import logging
import os
import sys
import yaml

#-----------------------------
# Imports for other modules --
#-----------------------------
import sqlalchemy
from sqlalchemy import (Column, Index, MetaData, PrimaryKeyConstraint,
                        UniqueConstraint, Table)

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_LOG = logging.getLogger(__name__)

#---------------------
#  Class definition --
#---------------------


class L1dbSchema(object):
    """Class for management of L1DB schema.

    Parameters
    ----------
    engine : `Engine`
        SQLAlchemy engine instance
    dia_object_index : `str`
        Indexing mode for DiaObject table, see :py:mod:`l1db` module.
    dia_object_nightly : `boolean`
        If `True` then create per-night DiaObject table as well.
    schema_file : `str`, optional
        Name of the YAML schema file, by default look for
        $L1DBPROTO_DIR/data/l1db-schema.yaml file.
    extra_schema_file : str
        Name of the YAML schema file fith extra column definitions, by
        default look for $L1DBPROTO_DIR/data/l1db-schema-extra.yaml file.
    """

    # map afw type names into cat type names
    _afw_type_map = dict(I="INT",
                         L="BIGINT",
                         F="FLOAT",
                         D="DOUBLE",
                         Angle="DOUBLE")

    def __init__(self, engine, dia_object_index, dia_object_nightly,
                 schema_file=None, extra_schema_file=None, column_map=None):

        self._engine = engine
        self._dia_object_index = dia_object_index
        self._dia_object_nightly = dia_object_nightly

        self._metadata = MetaData(self._engine)
        self._tables = {}

        # read schema from yaml
        if not schema_file:
            schema_file = os.path.join(os.environ.get("L1DBPROTO_DIR"),
                                       "data", "l1db-schema.yaml")
        if not extra_schema_file:
            extra_schema_file = os.path.join(os.environ.get("L1DBPROTO_DIR"),
                                             "data", "l1db-schema-extra.yaml")
        if not column_map:
            column_map = os.path.join(os.environ.get("L1DBPROTO_DIR"),
                                      "data", "l1db-afw-map.yaml")

        _LOG.debug("Reading schema file %s", schema_file)
        with open(schema_file) as yaml_stream:
            tables = list(yaml.load_all(yaml_stream))
            # index it by table name
            self._schemas = {table['table']: table for table in tables}
        _LOG.debug("Read %d tables from schema", len(self._schemas))

        _LOG.debug("Reading schema file %s", extra_schema_file)
        with open(extra_schema_file) as yaml_stream:
            tables = list(yaml.load_all(yaml_stream))
            # index it by table name
            self._schemas_extra = {table['table']: table for table in tables}

        _LOG.debug("Reading column map file %s", column_map)
        with open(column_map) as yaml_stream:
            self._column_map = yaml.load(yaml_stream)
            _LOG.debug("column map: %s", self._column_map)
        self._column_map_reverse = {}
        for table, cmap in self._column_map.items():
            self._column_map_reverse[table] = {v: k for k, v in cmap.items()}
        _LOG.debug("reverse column map: %s", self._column_map_reverse)

        # map cat column types to alchemy
        self._type_map = dict(DOUBLE=self._make_doube_type(),
                              FLOAT=sqlalchemy.types.Float,
                              DATETIME=sqlalchemy.types.TIMESTAMP,
                              BIGINT=sqlalchemy.types.BigInteger,
                              INTEGER=sqlalchemy.types.Integer,
                              INT=sqlalchemy.types.Integer,
                              TINYINT=sqlalchemy.types.Integer,
                              BLOB=sqlalchemy.types.LargeBinary,
                              CHAR=sqlalchemy.types.CHAR)

    def makeSchema(self, schema_dict=None, drop=False, mysql_engine='InnoDB'):
        """Create or re-create all tables.

        Parameters
        ----------
        schema_dict : `dict`, optional
            Dictionary with table name for a key and `afw.table.Schema`
            for a value. Columns in schema will be added to standard
            L1DB schema.
        drop : boolean
            If True then drop tables before creating new ones.
        mysql_engine : `str`
            MySQL engine type to use for new tables.
        """

        afw_schema = schema_dict.get('DiaObject')
        if self._dia_object_index == 'htm20_id_iov':
            constraints = [PrimaryKeyConstraint('htmId20', 'diaObjectId', 'validityStart',
                                                name='PK_DiaObject'),
                           Index('IDX_DiaObject_diaObjectId', 'diaObjectId')]
        else:
            constraints = [PrimaryKeyConstraint('diaObjectId', 'validityStart', name='PK_DiaObject'),
                           Index('IDX_DiaObject_validityStart', 'validityStart'),
                           Index('IDX_DiaObject_htmId20', 'htmId20')]
        table = Table('DiaObject', self._metadata,
                      *(self._table_columns('DiaObject', afw_schema) + constraints),
                      mysql_engine=mysql_engine)

        if self._dia_object_nightly:
            table = Table('DiaObjectNightly', self._metadata,
                          *self._table_columns('DiaObject', afw_schema),
                          mysql_engine=mysql_engine)

        if self._dia_object_index == 'last_object_table':
            constraints = [PrimaryKeyConstraint('htmId20', 'diaObjectId', name='PK_DiaObjectLast'),
                           Index('IDX_DiaObjectLast_diaObjectId', 'diaObjectId')]
            table = Table('DiaObjectLast', self._metadata,
                          *(self._table_columns('DiaObject', afw_schema) + constraints),
                          mysql_engine=mysql_engine)

        # for all other tables use index definitions in schema
        for table_name in ('DiaSource', 'SSObject', 'DiaForcedSource', 'DiaObject_To_Object_Match'):
            afw_schema = schema_dict.get(table_name)
            table = Table(table_name, self._metadata,
                          *(self._table_columns(table_name, afw_schema) +
                            self._table_indices(table_name)),
                          mysql_engine=mysql_engine)

        # special table to track visits, only used by prototype
        visits = Table('L1DbProtoVisits', self._metadata,
                       Column('visitId', sqlalchemy.types.BigInteger, nullable=False),
                       Column('visitTime', sqlalchemy.types.TIMESTAMP, nullable=False),
                       PrimaryKeyConstraint('visitId', name='PK_L1DbProtoVisits'),
                       Index('IDX_L1DbProtoVisits_visitTime', 'visitTime'),
                       mysql_engine=mysql_engine)

        # create all tables (optionally drop first)
        if drop:
            _LOG.info('dropping all tables')
            self._metadata.drop_all()
        _LOG.info('creating all tables')
        self._metadata.create_all()

    @property
    def visits(self):
        """Returns SQLAlchemy `Table` instance for L1DbProtoVisits table.
        """
        return self._table_schema('L1DbProtoVisits')

    @property
    def objects(self):
        """Returns SQLAlchemy `Table` instance for DiaObject table.
        """
        return self._table_schema('DiaObject')

    @property
    def objects_last(self):
        """Returns SQLAlchemy `Table` instance for DiaObjectLast table.
        """
        return self._table_schema('DiaObjectLast')

    @property
    def objects_nightly(self):
        """Returns SQLAlchemy `Table` instance for DiaObjectNightly table.
        """
        return self._table_schema('DiaObjectNightly')

    @property
    def sources(self):
        """Returns SQLAlchemy `Table` instance for DiaSource table.
        """
        return self._table_schema('DiaSource')

    @property
    def forcedSources(self):
        """Returns SQLAlchemy `Table` instance for DiaForcedSource table.
        """
        return self._table_schema('DiaForcedSource')

    def _table_columns(self, table_name, afw_schema):
        """Return set of columns in a table

        Parameters
        ----------
        table_name : `str`
            Name of the table.
        afw_schema : `afw.table.Schema`, optional
            Schema used by AP pipeline, if there are any columns in the
            schema not defined in standard (cat) schema they will be added
            to the new tables.

        Returns
        -------
        List of `Column` objects.
        """

        # get the list of columns in primary key, they are treated somewhat
        # specially below
        table_schema = self._schemas[table_name]
        pkey_columns = set()
        for index in table_schema.get('indices', []):
            if index['type'] == 'PRIMARY':
                pkey_columns = set(index['columns'])
                break

        # columns in _schemas_extra can override cat definitions, and I also want
        # to preserve order of the original cat schema so it's a bit messy
        columns = table_schema['columns']
        if table_name in self._schemas_extra:
            extra_columns = self._schemas_extra[table_name].get('columns', [])
            extra_columns = {col['name']: col for col in extra_columns}
            _LOG.debug("Extra columns for table %s: %s", table_name, extra_columns.keys())
            columns = []
            for col in table_schema['columns']:
                if col['name'] in extra_columns:
                    columns.append(extra_columns.pop(col['name']))
                else:
                    columns.append(col)
            # add all remaining extra columns
            columns += extra_columns.values()

        if afw_schema:
            # use afw schema to create extra columns
            column_names = {col['name'] for col in columns}
            for k, field in afw_schema:
                column = self._field2dict(field, table_name)
                if column['name'] not in column_names:
                    columns.append(column)

        # convert all column dicts into alchemy Columns
        column_defs = []
        for column in columns:
            kwargs = dict(nullable=column['nullable'])
            if column['name'] in pkey_columns:
                kwargs.update(autoincrement=False)
            ctype = self._type_map[column['type']]
            column_defs.append(Column(column['name'], ctype, **kwargs))

        return column_defs

    def _field2dict(self, field, table_name):
        """Convert afw schema field definition into a dict format.
        """
        column = field.getName()
        column = self._column_map_reverse[table_name].get(column, column)
        type = self._afw_type_map[field.getTypeString()]
        return dict(name=column, type=type, nullable=True, default=None)

    def _table_indices(self, table_name):
        """Return set of constraints/indices in a table

        Parameters
        ----------
        table_name : `str`
            Name of the table.

        Returns
        -------
        List of index/constraint objects.
        """

        table_schema = self._schemas[table_name]

        # convert all index dicts into alchemy Columns
        index_defs = []
        for index in table_schema['indices']:
            if index['type'] == "INDEX":
                index_defs.append(Index(index['name'], *index['columns']))
            else:
                kwargs = {}
                if 'name' in index:
                    kwargs['name'] = index['name']
                if index['type'] == "PRIMARY":
                    index_defs.append(PrimaryKeyConstraint(*index['columns'], **kwargs))
                elif index['type'] == "UNIQUE":
                    index_defs.append(UniqueConstraint(*index['columns'], **kwargs))

        return index_defs

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

    def _table_schema(self, name):
        """ Lazy reading of table schema """
        table = self._tables.get(name)
        if table is None:
            table = Table(name, self._metadata, autoload=True)
            self._tables[name] = table
            _LOG.debug("read table schema for %s: %s", name, table.c)
        return table
