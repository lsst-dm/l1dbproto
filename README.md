# Package l1dbproto

This package will contain a bunch of utilities for prototyping L1 database.
Its primary purpose is for performance testing of different database schemas
but it may also be useful for other purposes.

There are several related pieces of code in this package:
- `l1db` module which defines database schema and API for database access
- `DIA` module which implements an over-simplified model of Difference Image
  Analysis suitable for testing L1 database
- `gen_sources` script which generates and saves to a file a list of variable
  sources used by `DIA` module
- `create_l1_schema` script to load L1 schema into database
- `ap_proto` script which runs `DIA` repeatedly, simulates source-to-object
  matching and saves results into L1 database.


## Workflow

The prototype workflow should, to the best of my knowledge, more or less
reflect what is happening in the real AP system but with some details missing.
Here is the steps of visit processing that are implemented in `ap_proto`:
- extract DiaSources from the visit images, this is simulated by `DIA` module
  just generating a bunch of DiaSources base on the list of known variable
  sources (plus a number of purely random sources from noise)
- _reads latest version of DiaObjects_ from L1DB corresponding to visit FOV,
  database selection is based on HTM so it returns wider set of Objects that
  needs to be truncated to fit image extent
- perform source-to-object matching against DiaObjects read in previous step
- do forced photometry for DiaObjects without match, create DiaForcedSources
- _retrieve history of DiaSources_ (typically few months) matching all
  DiaObjects
- _same for DiaForcedSources_ (for shorter period)
- build new versions of DiaObjects from DiaSources and DiaForcedSources
- _saves new DiaObjects_ in L1DB
- _saves new DiaSources and DiaForcedSources_ in L1DB

In the above list database operations (_marked with emphasis_) consist of
reading all three types of DIA entities and writing updated ones back to the
database.

## Database API

Class `l1db.L1db` implements database API which defines schema and stores and
restores Dia(Forced)Sources and DiaObjects in the database. The schema can be
complicated and have different optimizations built-in but API is supposed to
handle all complexities and it should operate at the level of science objects
only.

### Data types

`l1db.L1db` methods accept or return DIA records as `afw.table` catalogs. It
supports renaming of some schema columns between L1DB schema (as defined in
`cat` package) and `afw.table` convention:
- `diaObjectId` <-> `id` (for DiaObject table)
- `diaSourceId` <-> `id` (for DiaSource table)
- `ra` <-> `coord_ra`
- `decl` <-> `coord_dec`
- `parentDiaSourceId` <-> `parent` (for DiaSource table)

Other column names are passed without any change.

Current implementation does not support unit conversion and expects that
`afw.table` schema follows L1DB conventions (as defined in DPDD/`cat`).
In particular this means that angles in schema should be specified in
degrees, which is different from standard `afw.table` units (rad).

### Instantiating L1db

Constructor of `l1db.L1db` takes two arguments:
- instance of `l1db.L1dbConfig` type which is a subclass of `pex.config.Config`
- optional dictionary mapping table names into `afw.table.Schema` instances

Configuration type `l1db.L1dbConfig` defines small number of parameters which
determine how to connect to database, what kind of schema L1DB has and few
other options. Field documentation strings provide brief description of what
each parameter does and there are few examples of configuration overrides in
`cfg/` folder:

Second argument to constructor is used to pass `afw.table` schemas that client
code is using for catalogs. Any columns that are not defined in standard DPDD
schema but appear in second argument will be added to the schema managed by `L1db`.
This needs to be defined consistently when creating schema (with `makeSchema()`
method) and other data access methods.

Example of instantiating `L1db`:

    from lsst.l1dbproto import l1db

    afwObjectSchema = afwTable.Schema()
    afwSourceSchema = afwTable.Schema()
    # fill both schemas
    afw_schemas = dict(DiaObject=afwObjectSchema,
                       DiaSource=afwSourceSchema)
    db = l1db.L1db(config, afw_schemas)


### Initializing database schema

There is a single method that creates all necessary tables in a database.

#### `L1db.makeSchema(self, drop=False)`

Optional `drop` argument causes existing tables to be removed if set to `True`.
This method needs to be called once before any other work can be performed
on a database. Script `create_l1_schema` is a command line wrapper for this
method but it only knows about standard DPDD schema. If schema needs to be
extended (by passing `afw.table` schemas to `L1db` constructor) this method
should be called from an application that is aware of the extended schema.


### Reading data from database

There are three methods in `L1db`, each is responsible for reading data of one
type.


#### `L1db.getDiaObjects(self, pixel_ranges)`

Retrieves most recent version of each DiaObject from specified region. Objects
are searched based on pixelization index and region is determined by the set of
index ranges. `pixel_ranges` is a sequence of tuples where each tuple defines
a single range of index IDs in regular Python convention `[low, high)`.

Returns `afw.table` catalog of DiaObject instances. By default all columns in
database table will be returned in a catalog schema, one cause `dia_object_columns`
configuration field to limit the list of columns queried.


#### `L1db.getDiaSources(self, object_ids, dt)`

Retrieves history of the DiaSources matching given DiaObjects. `object_ids`
is a list (or any container) of the matching DiaObjects IDs. `dt` is the
`datetime` of the current visit.

Returned history is supposed to be limited to `read_sources_months` period
(from config object), but in current implementation history is not limited.

Returns `afw.table` catalog of DiaSource records. Schema of returned catalog
is determined by database table schema.


#### `L1db.getDiaForcedSources(self, object_ids, dt)`

Retrieves history of the DiaForcedSources matching given DiaObjects.
`object_ids` is a list (or any container) of the matching DiaObjects IDs.
`dt` is the `datetime` of the current visit.

Returned history is supposed to be limited to `read_forced_sources_months`
period (from config object), but in current implementation history is not
limited.

Returns `afw.table` catalog of DiaForcedSource records. Schema of returned
catalog is determined by database table schema.


### Saving new data to a database

There are three methods in `L1db`, each is responsible for storing data of one
type.


#### `L1db.storeDiaObjects(self, objects, dt)`

Stores DiaObjects from current visit. `objects` is a catalog of DiaObject
records. `dt` is the visit time, an instance of Python `datetime.datetime`
type.


#### `L1db.storeDiaSources(self, sources)`

Stores DiaSources from current visit. `sources` is a catalog of DiaSource
record.


#### `L1db.storeDiaForcedSources(self, sources)`

Stores DiaForcedSources from current visit. `sources` is a catalog of
DiaForcedSource records.


## Applications

Package also builds a small set of applications.


### ap_proto

This is the primary application for performance studies. It runs AP workflow
described above over multiple visits, randomly choosing pointing direction
for each visit.

This is likely not reusable for anything else.


### create_l1_schema

Takes a configuration file as a parameter and creates (or re-creates) L1DB
schema in the database. This is a command line wrapper for `L1db.makeSchema`
method, and it should be run once to initialize L1DB. This script creates
standard schema (as defined in config files derived from `cat` schema).
For alternative schemas one can either update YAML config files or create a
copy of this script which passes `afw.table` schemas to `L1db` constructor.


### gen_sources

Generates a set of variable sources to be used by `ap_proto` application.
Writes generated sources to a file which is then used as input by `ap_proto`.


### log2cvs

Parses `ap_proto` logs and writes timing information into CSV files which are
used by iPython notebooks to produce confusing plots.


### parse_cat

Parses SQL schema files from `cat` package and dumps schema in a format that
is usable by `L1db` class (YAML schema file in `data/l1db-schema.yaml`).
