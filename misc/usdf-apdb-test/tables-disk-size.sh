#!/bin/bash

du -sk  /data/disk*/test-apdb-cluster/data/apdb/* | \
    tr /- '\t\t' | \
    gawk '
    {
        size[$9] += $1/1024./1024.
    }
    END {
        asorti(size, d);
        printf("%-20s %9s\n", "Table", "Size, GiB");
        printf("==============================\n");
        for (var in d) {
            printf("%-20s %9.2f\n", d[var], size[d[var]])
        }
    }'
