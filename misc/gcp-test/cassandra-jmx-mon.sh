#!/bin/bash

JYTHON=/home/andy_salnikov_gmail_com/project/jython/bin/jython
APDB_DIR=/home/andy_salnikov_gmail_com/project/apdb-gcloud

# Wait a little bit for Cassandra to start
sleep 15

mkdir -p $APDB_DIR/mon-data
cd $APDB_DIR/mon-data
exec $JYTHON $APDB_DIR/l1dbproto/bin.src/cassandra_jmx_mon.py \
    -v -c $APDB_DIR/l1dbproto/cassandra_jmx_mon.yaml \
    -C -R 1 "$@"

