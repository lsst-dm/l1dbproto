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
- extract DIASources from the visit images, this is simulated by `DIA` module
  just generating a bunch of DIASources base on the list of known variable
  sources (plus a number of purely random sources from noise)
- _reads latest version of DIAObjects_ from L1DB corresponding to visit FOV,
  database selection is based on HTM so it returns wider set of Objects that
  needs to be truncated to fit image extent
- perform source-to-object matching against DIAObjects read in previous step
- do forced photometry for DIAObjects without match, create DIAForcedSources
- _retrieve history of DIASources_ (typically few months) matching all
  DIAObjects
- _same for DIAForcedSources_ (for shorter period)
- build new versions of DIAObjects from DIASources and DIAForcedSources
- _saves new DIAObjects_ in L1DB
- _saves new DIASources and DIAForcedSources_ in L1DB

In the above list database operations (_marked with emphasis_) consist of
reading all three types of DIA entities and writing updated ones back to the
database.

## Database API

Class `l1db.L1db` implements database API which defines schema and stores and
restores DIA(Forced)Sources and DIAObjects in the database. The schema can be
complicated and have different optimizations built-in but API is supposed to
handle all complexities and it should operate at the level of science objects
only.

### Data types

`l1db.L1db` methods accept or return DIA records as instances of special Python
classes. Those classes are defined now in `l1db` modules, here is their brief
description (consult source for more details):
- `l1db.DiaObject_short` - this type is used to return data from
  `getDiaObjects()` method when `read_full_objects` (see configuration
  below) is set to 0.
- `l1db.DiaObject` - this type is expected by `storeDiaObjects()` method,
  it is also used as return data type for `getDiaObjects()` method when
  `read_full_objects` is non-zero.
- `l1db.DiaSource_full` - this type is returned by `getDiaSources()` method.
- `l1db.DiaSource` - this type is accepted by `storeDiaSources()` method (this
  is probably an oversight, should have used DiaSource_full for this too).
- `l1db.DiaForcedSource` - this type is returned by `getDiaFSources()` method
  and is accepted by `storeDiaForcedSources()` method.

For now this types are defined as `namedtuple` types. Time-related attributes
in these types should hold instances of `datetime.datetime` classes.

### Instantiating L1db

Constructor of `l1db.L1db` takes one argument - name of the configuration file
that defines all relevant parameters (this was done to simplify development
during prototyping).

Configuration file contains small number of parameters which determine how to
connect to database, what kind of schema L1DB has and few other options. There
are few examples of configuration files in `cfg/` folder, here is a copy of
configuration used for running recent prototype version against mysql database:

```
#
# Example config file for L1DB using mysql backend
#

[database]
# SQLAlchemy URL for connection
url = mysql://localhost/l1dbproto?unix_socket=/var/lib/mysql/mysql.sock

[l1db]
# DiaObject indexing options, possible values:
#  - baseline: (default) PK = (object ID, validity)
#  - htm20_id_iov:       PK = (htmId, object ID, validity)
#  - last_object_table:  as baseline but use separate table
#                        for last DiaObject version
dia_object_index = last_object_table

# if set to 1 then define special nightly non-indexed DiaObject table
# which is merged with actual table every night
dia_object_nightly = 0

# if non-zero then do "upsert" for DiaObjects which could be more efficient
object_last_replace = 1

# if set to 0 then DiaSources are not read at all,
# actual non-zero value is ignored for now (reads all sources)
read_sources_months = 12

# if set to 0 then DiaForcedSources are not read at all,
# actual non-zero value is ignored for now (reads all sources)
read_forced_sources_months = 12

# set to 1 to read complete DiaObject record (all columns)
read_full_objects = 0

# how to select sources, possible values:
#  - by-fov: selects Sources using FOV region using HTM index
#  - by-oid: selects Sources using DiaObject ID
source_select = by-oid
```


### Reading data from database

There are three methods in `L1db`, each is responsible for reading data of one
type.


#### `L1db.getDiaObjects(self, region, explain=False)`

Retrieves most recent version of each DIAObject from specified region or
near it. `region` is an instance of `sphgeom.Region` type. DIAObject search
is based on HTM index and returned set can also include records outside
specified region, so some filtering will be needed on client side.

Returns list of `l1db.DiaObject_short` instances if `read_full_objects`
(in `l1db` section) is set to zero in config file, otherwise returns
`l1db.DiaObject` instances.


#### `L1db.getDiaSources(self, region, objects, explain=False)`

Retrieves history of the DIASources from a given region matching given
DIAObjects. `region` is an instance of `sphgeom.Region` type, `objects` is
a list of `DiaObject` (or `DiaObject_short`) instances.

One of `region` or `objects` can be ignored when selecting records from
database. Currently if `source_select` config parameter is set to `by-oid`
then DIAOject IDs are used for selection, if `source_select` is set to
`by-fov` then region is used for selection using HTM indexing. `by-oid` is
likely to be faster for current implementation.

Returned history is supposed limited to `read_sources_months` period (from
config file), but in current implementation history is not limited.

Returns list of `l1db.DiaSource_full` instances.


#### `L1db.getDiaFSources(self, objects, explain=False)`

Retrieves history of the DIAForcedSources matching given DIAObjects.
`objects` is a list of `DiaObject` (or `DiaObject_short`) instances.

Returned history is supposed limited to `read_forced_sources_months` period
(from config file), but in current implementation history is not limited.

Returns list of `l1db.DiaForcedSource` instances.


### Saving new data to a database

There are three methods in `L1db`, each is responsible for storing data of one
type.


#### `L1db.storeDiaObjects(self, objects, dt, explain=False)`

Stores DIAObjects from current visit. `objects` is a list of `l1db.DiaObject`
instances. `dt` is the visit time, an instance of Python `datetime.datetime`
type.


#### `L1db.storeDiaSources(self, sources, explain=False)`

Stores DIASources from current visit. `objects` is a list of `l1db.DiaSource`
instances.


#### `L1db.storeDiaForcedSources(self, sources, explain=False)`

Stores DIAForcedSources from current visit. `objects` is a list of
`l1db.DiaForcedSource` instances.


### Other methods

#### `L1db.makeSchema(self, drop=False)`

Creates all necessary tbles for the L1DB schema based on configuration
parameters (from config file). If `drop` is True then removes all tables
from database first.

This method should be called once before all other operations to prepare
L1DB database.


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
method, and it should be run once to initialize L1DB.


### gen_sources

Generates a set of variable sources to be used by `ap_proto` application.
Writes generated sources to a file which is then used as input by `ap_proto`.


### log2cvs

Parses `ap_proto` logs and writes timing information into CSV files which are
used by iPython notebooks to produce confusing plots.


### parse_cat

Parses SQL schema files from `cat` package and dumps schema in a format that
can be pasted into L1db class. (This process should be automated later,
L1DB schema will be generated from `cat` directly).
