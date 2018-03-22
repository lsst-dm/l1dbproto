import lsst.l1dbproto.l1db
assert type(config)==lsst.l1dbproto.l1db.L1dbConfig, 'config is of type %s.%s instead of lsst.l1dbproto.l1db.L1dbConfig' % (type(config).__module__, type(config).__name__)

# SQLAlchemy database connection URI
config.db_url = "mysql://localhost/l1dbproto?unix_socket=/var/lib/mysql/mysql.sock"

# Transaction isolation level
# Allowed values:
# 	READ_COMMITTED	Read committed
# 	READ_UNCOMMITTED	Read uncommitted
# 	REPEATABLE_READ	Repeatable read
# 	SERIALIZABLE	Serializable
# 	None	Field is optional
#
config.isolation_level = 'READ_COMMITTED'

# Indexing mode for DiaObject table
# Allowed values:
# 	baseline	Index defined in baseline schema
# 	pix_id_iov	(pixelId, objectId, iovStart) PK
# 	last_object_table	Separate DiaObjectLast table
# 	None	Field is optional
#
config.dia_object_index = 'last_object_table'

# Use separate nightly table for DiaObject
config.dia_object_nightly = False

# Number of months of history to read from DiaSource
config.read_sources_months = 12

# Number of months of history to read from DiaForcedSource
config.read_forced_sources_months = 6

# List of columns to read from DiaObject, by default read all columns
config.dia_object_columns = [
    "diaObjectId", "lastNonForcedSource", "ra", "decl",
    "raSigma", "declSigma", "ra_decl_Cov", "pixelId"
    ]

# If True (default) then use "upsert" for DiaObjects
config.object_last_replace = True

# If True (default) then use "upsert" for DiaObjects
config.explain = False

# Location of (YAML) configuration file with standard schema
# config.schema_file = 'data/l1db-schema.yaml'

# Location of (YAML) configuration file with extra schema
# config.extra_schema_file = 'data/l1db-schema-extra.yaml'

# Location of (YAML) configuration file with column mapping
# config.column_map = 'data/l1db-afw-map.yaml'

# If True then print/log timing information
# config.timer = False
