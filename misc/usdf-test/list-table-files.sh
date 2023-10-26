#!/bin/bash

table=${1:?Table name is required}

ls -1sk /data/disk*/*/data/apdb/${table}-*/*-Data.db |\
    tr / ' ' | \
    gawk '{printf("%'"'"'12d  %s  %s\n", $1, $3, $8)}'
