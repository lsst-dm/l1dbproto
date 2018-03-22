"""Module responsible for L1DB schema operations.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
from collections import namedtuple
import logging
import yaml

#-----------------------------
# Imports for other modules --
#-----------------------------
import sqlalchemy
from sqlalchemy import (Column, Index, MetaData, PrimaryKeyConstraint,
                        UniqueConstraint, Table)
import lsst.afw.table as afwTable

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_LOG = logging.getLogger(__name__)

# Classes for representing schema

# Column description:
#    name : column name
#    type : name of cat type (INT, FLOAT, etc.)
#    nullable : True or False
#    default : default value for column, can be None
#    description : documentation, can be None or empty
#    unit : string with unit name, can be None
#    ucd : string with ucd, can be None
ColumnDef = namedtuple('ColumnDef', 'name type nullable default description unit ucd')

# Index description:
#    name : index name, can be None or empty
#    type : one of "PRIMARY", "UNIQUE", "INDEX"
#    columns : list of column names in index
IndexDef = namedtuple('IndexDef', 'name type columns')

# Table description:
#    name : table name
#    description : documentation, can be None or empty
#    columns : list of ColumnDef instances
#    indices : list of IndexDef instances, can be empty or None
TableDef = namedtuple('TableDef', 'name description columns indices')


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
    extra_schema_file : `str`, optional
        Name of the YAML schema file with extra column definitions, by
        default look for $L1DBPROTO_DIR/data/l1db-schema-extra.yaml file.
    column_map : `str`, optional
        Name of the YAML file with column mappings, by default look for
        $L1DBPROTO_DIR/data/l1db-afw-map.yaml file.
    afw_schemas : `dict`, optional
        Dictionary with table name for a key and `afw.table.Schema`
        for a value. Columns in schema will be added to standard L1DB
        schema (only if standard schema does not have matching column).
    """

    # map afw type names into cat type names
    _afw_type_map = dict(I="INT",
                         L="BIGINT",
                         F="FLOAT",
                         D="DOUBLE",
                         Angle="DOUBLE")
    _afw_type_map_reverse = dict(INT="I",
                                 BIGINT="L",
                                 FLOAT="F",
                                 DOUBLE="D",
                                 DATETIME="L")

    def __init__(self, engine, dia_object_index, dia_object_nightly,
                 schema_file=None, extra_schema_file=None, column_map=None,
                 afw_schemas=None):

        self._engine = engine
        self._dia_object_index = dia_object_index
        self._dia_object_nightly = dia_object_nightly

        self._metadata = MetaData(self._engine)
        self._tables = {}

        _LOG.debug("Reading column map file %s", column_map)
        with open(column_map) as yaml_stream:
            # maps cat column name to afw column name
            self._column_map = yaml.load(yaml_stream)
            _LOG.debug("column map: %s", self._column_map)
        self._column_map_reverse = {}
        for table, cmap in self._column_map.items():
            # maps afw column name to cat column name
            self._column_map_reverse[table] = {v: k for k, v in cmap.items()}
        _LOG.debug("reverse column map: %s", self._column_map_reverse)

        # build complete table schema
        self._schemas = self._buildSchemas(schema_file, extra_schema_file,
                                           afw_schemas)

        # map cat column types to alchemy
        self._type_map = dict(DOUBLE=self._getDoubleType(),
                              FLOAT=sqlalchemy.types.Float,
                              DATETIME=sqlalchemy.types.TIMESTAMP,
                              BIGINT=sqlalchemy.types.BigInteger,
                              INTEGER=sqlalchemy.types.Integer,
                              INT=sqlalchemy.types.Integer,
                              TINYINT=sqlalchemy.types.Integer,
                              BLOB=sqlalchemy.types.LargeBinary,
                              CHAR=sqlalchemy.types.CHAR)

    def makeSchema(self, drop=False, mysql_engine='InnoDB'):
        """Create or re-create all tables.

        Parameters
        ----------
        drop : boolean
            If True then drop tables before creating new ones.
        mysql_engine : `str`
            MySQL engine type to use for new tables.
        """

        if self._dia_object_index == 'pix_id_iov':
            # Special PK with HTM column in first position
            constraints = self._tableIndices('DiaObjectIndexHtmFirst')
        else:
            constraints = self._tableIndices('DiaObject')
        Table('DiaObject', self._metadata,
              *(self._tableColumns('DiaObject') + constraints),
              mysql_engine=mysql_engine)

        if self._dia_object_nightly:
            # Same as DiaObject but no index
            Table('DiaObjectNightly', self._metadata,
                  *self._tableColumns('DiaObject'),
                  mysql_engine=mysql_engine)

        if self._dia_object_index == 'last_object_table':
            # Same as DiaObject but with special index
            Table('DiaObjectLast', self._metadata,
                  *(self._tableColumns('DiaObject') +
                    self._tableIndices('DiaObjectLast')),
                  mysql_engine=mysql_engine)

        # for all other tables use index definitions in schema
        for table_name in ('DiaSource', 'SSObject', 'DiaForcedSource', 'DiaObject_To_Object_Match'):
            Table(table_name, self._metadata,
                  *(self._tableColumns(table_name) +
                    self._tableIndices(table_name)),
                  mysql_engine=mysql_engine)

        # special table to track visits, only used by prototype
        Table('L1DbProtoVisits', self._metadata,
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
        """SQLAlchemy `Table` instance for L1DbProtoVisits table.
        """
        return self._table_schema('L1DbProtoVisits')

    @property
    def objects(self):
        """SQLAlchemy `Table` instance for DiaObject table.
        """
        return self._table_schema('DiaObject')

    @property
    def objects_last(self):
        """SQLAlchemy `Table` instance for DiaObjectLast table.
        """
        return self._table_schema('DiaObjectLast')

    @property
    def objects_nightly(self):
        """SQLAlchemy `Table` instance for DiaObjectNightly table.
        """
        return self._table_schema('DiaObjectNightly')

    @property
    def sources(self):
        """SQLAlchemy `Table` instance for DiaSource table.
        """
        return self._table_schema('DiaSource')

    @property
    def forcedSources(self):
        """SQLAlchemy `Table` instance for DiaForcedSource table.
        """
        return self._table_schema('DiaForcedSource')

    def getAfwSchema(self, table_name, columns=None):
        """Return afw schema for given table.

        Parameters
        ----------
        table_name : `str`
            One of known L1DB table names.
        columns : `list` of `str`, optional
            Include only given table columns in schema, by default all columns
            are included.

        Returns
        -------
        schema : `afw.table.Schema`
        column_map : `dict`
            Mapping of the table/result column names into schema key.
        """

        table = self._schemas[table_name]
        col_map = self._column_map.get(table_name, {})

        # make a schema
        col2afw = {}
        schema = afwTable.Schema()
        for column in table.columns:
            if columns and column.name not in columns:
                continue
            afw_col = col_map.get(column.name, column.name)
            #
            # NOTE: degree to radian conversion is not supported (yet)
            #
            if False and column.type in ("DOUBLE", "FLOAT") and column.unit == "deg":
                # angles in afw are radians and have special "Angle" type
                key = schema.addField(afw_col,
                                      type="Angle",
                                      doc=column.description or "",
                                      units="rad")
            elif column.type == "BLOB":
                # No BLOB support for now
                pass
            else:
                units = column.unit or ""
                # some units in schema are not recognized by afw but we do not care
                key = schema.addField(afw_col,
                                      type=self._afw_type_map_reverse[column.type],
                                      doc=column.description or "",
                                      units=units,
                                      parse_strict="silent")
            col2afw[column.name] = key

        return schema, col2afw

    def getAfwColumns(self, table_name):
        """Returns mapping of afw column names to Column definitions.

        Parameters
        ----------
        table_name : `str`
            One of known L1DB table names.

        Returns
        -------
        `dict` with afw column names as keys and `ColumnDef` instances as
        values.
        """
        table = self._schemas[table_name]
        col_map = self._column_map.get(table_name, {})

        cmap = {}
        for column in table.columns:
            afw_name = col_map.get(column.name, column.name)
            cmap[afw_name] = column
        return cmap

    def getColumnMap(self, table_name):
        """Returns mapping of column names to Column definitions.

        Parameters
        ----------
        table_name : `str`
            One of known L1DB table names.

        Returns
        -------
        `dict` with column names as keys and `ColumnDef` instances as
        values.
        """
        table = self._schemas[table_name]
        cmap = {column.name: column for column in table.columns}
        return cmap

    def _buildSchemas(self, schema_file, extra_schema_file, afw_schemas):
        """Create schema definitions for all tables.

        Reads YAML schemas and builds dictionary containing `TableDef`
        instances for each table.

        Parameters
        ----------
        schema_file : `str`
            Name of YAML file with standard cat schema.
        extra_schema_file : `str`
            Name of YAML file with extra table information.
        afw_schemas : `dict`, optional
            Dictionary with table name for a key and `afw.table.Schema`
            for a value. Columns in schema will be added to standard L1DB
            schema (only if standard schema does not have matching column).

        Returns
        -------
        `dict` with table names as keys and `TableDef` instances as values.
        """

        _LOG.debug("Reading schema file %s", schema_file)
        with open(schema_file) as yaml_stream:
            tables = list(yaml.load_all(yaml_stream))
            # index it by table name
        _LOG.debug("Read %d tables from schema", len(tables))

        _LOG.debug("Reading schema file %s", extra_schema_file)
        with open(extra_schema_file) as yaml_stream:
            extras = list(yaml.load_all(yaml_stream))
            # index it by table name
            schemas_extra = {table['table']: table for table in extras}

        # merge extra schema into a regular schema, for now only columns are merged
        for table in tables:
            table_name = table['table']
            if table_name in schemas_extra:
                columns = table['columns']
                extra_columns = schemas_extra[table_name].get('columns', [])
                extra_columns = {col['name']: col for col in extra_columns}
                _LOG.debug("Extra columns for table %s: %s", table_name, extra_columns.keys())
                columns = []
                for col in table['columns']:
                    if col['name'] in extra_columns:
                        columns.append(extra_columns.pop(col['name']))
                    else:
                        columns.append(col)
                # add all remaining extra columns
                table['columns'] = columns + list(extra_columns.values())

                if 'indices' in schemas_extra[table_name]:
                    raise RuntimeError("Extra table definition contains indices, "
                                       "merging is not implemented")

                del schemas_extra[table_name]

        # Pure "extra" table definitions may contain indices
        tables += schemas_extra.values()

        # convert all dicts into named tuples
        schemas = {}
        for table in tables:

            columns = table.get('columns', [])

            table_name = table['table']
            afw_schema = afw_schemas and afw_schemas.get(table_name)
            if afw_schema:
                # use afw schema to create extra columns
                column_names = {col['name'] for col in columns}
                for _, field in afw_schema:
                    column = self._field2dict(field, table_name)
                    if column['name'] not in column_names:
                        columns.append(column)

            table_columns = []
            for col in columns:
                # For prototype set default to 0 even if columns don't specify it
                if "default" not in col:
                    default = None
                    if col['type'] not in ("BLOB", "DATETIME"):
                        default = 0
                else:
                    default = col["default"]

                column = ColumnDef(name=col['name'],
                                   type=col['type'],
                                   nullable=col.get("nullable"),
                                   default=default,
                                   description=col.get("description"),
                                   unit=col.get("unit"),
                                   ucd=col.get("ucd"))
                table_columns.append(column)

            table_indices = []
            for idx in table.get('indices', []):
                index = IndexDef(name=idx.get('name'),
                                 type=idx.get('type'),
                                 columns=idx.get('columns'))
                table_indices.append(index)

            schemas[table_name] = TableDef(name=table_name,
                                           description=table.get('description'),
                                           columns=table_columns,
                                           indices=table_indices)

        return schemas

    def _tableColumns(self, table_name):
        """Return set of columns in a table

        Parameters
        ----------
        table_name : `str`
            Name of the table.

        Returns
        -------
        List of `Column` objects.
        """

        # get the list of columns in primary key, they are treated somewhat
        # specially below
        table_schema = self._schemas[table_name]
        pkey_columns = set()
        for index in table_schema.indices:
            if index.type == 'PRIMARY':
                pkey_columns = set(index.columns)
                break

        # convert all column dicts into alchemy Columns
        column_defs = []
        for column in table_schema.columns:
            kwargs = dict(nullable=column.nullable)
            if column.default is not None:
                kwargs.update(server_default=str(column.default))
            if column.name in pkey_columns:
                kwargs.update(autoincrement=False)
            ctype = self._type_map[column.type]
            column_defs.append(Column(column.name, ctype, **kwargs))

        return column_defs

    def _field2dict(self, field, table_name):
        """Convert afw schema field definition into a dict format.
        """
        column = field.getName()
        column = self._column_map_reverse[table_name].get(column, column)
        ctype = self._afw_type_map[field.getTypeString()]
        return dict(name=column, type=ctype, nullable=True)

    def _tableIndices(self, table_name):
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
        for index in table_schema.indices:
            if index.type == "INDEX":
                index_defs.append(Index(index.name, *index.columns))
            else:
                kwargs = {}
                if index.name:
                    kwargs['name'] = index.name
                if index.type == "PRIMARY":
                    index_defs.append(PrimaryKeyConstraint(*index.columns, **kwargs))
                elif index.type == "UNIQUE":
                    index_defs.append(UniqueConstraint(*index.columns, **kwargs))

        return index_defs

    def _getDoubleType(self):
        """
        DOUBLE type is database-specific, select one based on dialect.
        """
        if self._engine.name == 'mysql':
            from sqlalchemy.dialects.mysql import DOUBLE
            return DOUBLE(asdecimal=False)
        elif self._engine.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
            return DOUBLE_PRECISION
        elif self._engine.name == 'sqlite':
            # all floats in sqlite are 8-byte
            from sqlalchemy.dialects.sqlite import REAL
            return REAL
        else:
            raise TypeError('cannot determine DOUBLE type, unexpected dialect: ' + self._engine.name)

    def _table_schema(self, name):
        """Return SQLAlchemy schema for a table.

        This does lazy reading of table schema from metadata.

        Parameters
        ----------
        name : `str`
            Table name.

        Returns
        -------
        `sqlalchemy.Table` instance.
        """
        table = self._tables.get(name)
        if table is None:
            table = Table(name, self._metadata, autoload=True)
            self._tables[name] = table
            _LOG.debug("read table schema for %s: %s", name, table.c)
        return table
