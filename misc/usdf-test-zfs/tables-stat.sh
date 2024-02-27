#!/bin/bash

docker compose exec cassandra nodetool tablestats apdb | \
    gawk '
        /Table:/{name=$2}
        /^[\t ]*SSTable count:/ {
            cnt=$3
        }
        /Space used .total.:/ {
            size = $4/1024./1024./1024;
        }
        /Number of partitions/ {
            npart=$5;
            if (header == 0) {
                header = 1;
                printf("%-20s %6s %10s %9s\n", "Table", "Nfiles", "Npart", "Size, GiB");
                printf("================================================\n");
            }
            printf("%-20s %6d %10d %9.2f\n", name, cnt, npart, size);
        }
    '
