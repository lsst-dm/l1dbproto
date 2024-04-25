Metrics in ap_proto
-------------------

Agent name is `ap_proto`.

Many metrics are times, they include `real`, `user`, `sys` keys as values of times in seconds.

Metric names:

- `visit_processing_time` - in single-process times call to `visit()`;
  in `fork` mode times whole visit procesing, including forking;
  in `mpi` mode times whole visit procesing, including scatter-gather.
- `total_visit_time` - times whole visit loop, should be close to `visit_processing_time`.
- `tile_visit_time` - times call to `visit()` in each MPI process (`mpi` mode only).
- `read_counts` - in `visit()` this contains counts of records returned from APDB query,
  value keys include `objects`, `objects_filtered`, `sources`, `forcedsources`.
- `tile_store_time` - in `visit()` this times call to `apdb.store()`.
- `store_counts` - contains count of records passed to `store()`,
  value keys include `objects`, `sources`, `forcedsources`.

Tags include:

- `visit` - visit ID.
- `rank` - MPI rank (in `mpi` mode only).
- `tile` - tile name in "NxM" format (in `mpi` mode only).


Metrics in ApdbCassandra
------------------------

Agent name is `lsst.dax.apdb.cassandra.apdbCassandra`.

Tags include:

- `table` - name of the table.

Metric names:

- `select_time` - times query execution for selection from a table, including conversion to DataFrame.
- `update_time` - times DiaSource reassignment to SSObjects.
- `insert_build_time` - times quiery building for inserts into the corresponding table.
- `insert_time` - times inserts into the corresponding table.
- `select_query_stats` - stats about DiaObject select query, value keys include `num_sp_part`
  for a number of spatial partitions and `num_queries` for a total number of queries.


Metrics in ApdbSql
------------------

Agent name is `lsst.dax.apdb.sql.apdbSql`.

Metric names:

- `select_time` - times query execution for select from a table (`table` tag is defined).
- `delete_time` - times deleting rows from DiaObjectLast table (if that is enabled).
- `insert_time` - times inserting rows into DiaObjectLast table (if that is enabled).
- `truncate_time` - times validity interval truncation (`table` tag should be `DiaObject`).
- `insert_time` - times insertion into `table`.


Metrics in ApdbCassandraReplica
-------------------------------

Agent name is `lsst.dax.apdb.cassandra.apdbCassandraReplica`.

Metric names:

- `chunks_select_time` - times query for selecting known chunks IDs.
- `table_chunk_select_time` - times query for selecting data from replica chunks (`table` tag is defined).
- `chunks_delete_time` - time deletion from chunks ID table.
- `table_chunk_delete_time` - times query for deleting data from replica chunks (`table` tag is defined).


Metrics in ApdbSqlReplica
-------------------------

Agent name is `lsst.dax.apdb.cassandra.apdbSqlReplica`.

Metric names:

- `chunks_select_time` - times query for selecting known chunks IDs.
- `table_chunk_select_time` - times query for selecting data from replica chunks (`table` tag is defined).
- `chunks_delete_time` - time deletion from chunks ID table.
