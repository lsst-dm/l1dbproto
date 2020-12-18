#!/bin/bash
#
# Script to intialize data folders for Cassandra installation at CGP.
#
# For APDB tests at GCP we use local SSD storage which is not persistent,
# meaning that on each suspend the disks are wiped and we have to partition,
# initialize filesystem and mount those disks again. On guest OS reboot disks
# keep their data but they are not mounted because we don't add them to fstab
# (as the number of attached disks can potentially change).
# 
# This script does all necessary steps to mount all avilable disks at the
# pre-defined location and updates Cassandra config file with that location.
# This script has to be run before Cassandra can start, which should be
# coordinated by systemd. One complication is that Cassandra needs to be
# stopped before suspend and restarted on wakeup (again after execution of
# this script) so suspend/resume cycle needs to be configured in systemd as
# well.
#

set -e

user_id=cassandra.cassandra
cassandra_config=/etc/cassandra/cassandra.yaml
dry_run=


usage() {
    cat <<EOD
Usage: $0 [-h] [-n]

Options:
    -h      Print usage information
    -n      Dry run, do not execute any commands, only print
EOD
}

do_exec() {
    if [ -n "$dry_run" ]; then
        echo "command:" "$@"
    else
        eval "$@"
    fi
}

# make mount location, takes an integer number
# folder_name <folder_index>
folder_name() {
    index=${1:?Folder index required}
    printf "/data/apdb%d" $index
}

# partition_name <disk_device>
partition_name() {
    disk=${1:?Device name required}
    echo ${disk}p1
}

# makes a list of SSD devices
ssd_devices() {
    # device names are /dev/nvmeXnY
    find /dev -name "nvme[0-9]n[0-9]" -o -name "nvme[0-9]n[0-9][0-9]" | sort
}

# makes a list of partitions on SSD devices, only first partition is used
ssd_partitions() {
    # partition names are /dev/nvmeXnYp1
    find /dev -name "nvme[0-9]n[0-9]p1" -o -name "nvme[0-9]n[0-9][0-9]p1" | sort
}

mounted_folders() {
    cut -d' ' -f2 /proc/mounts | grep "^/data/apdb" | sort
}

# make_partition <disk_device>
make_partition() {
    disk=${1:?Device name required}
    do_exec parted --script -a optimal ${disk} mklabel gpt
    do_exec parted --script -a optimal ${disk} mkpart primary xfs 1MiB 100%
}

# make_fs <part_device>
make_fs() {
    partition=${1:?Partition name required}
    do_exec mkfs.xfs -f -s size=4k ${partition}
}

# mount_partition <part_device> <folder>
mount_partition() {
    partition=${1:?Partition name required}
    folder=${2:?Folder name required}

    do_exec mkdir -p "$folder"
    do_exec mount "$partition" "$folder"
    do_exec chown "$user_id" "$folder"
}

# ujpdate cassandra.yaml file `data_file_directories` with the list of mounted
# disks in /data folder
update_cassandra_config() {
    if [ -z "$dry_run" ]; then
        # find all mounted fs in /apdb
        folders=$(mounted_folders)

        tmpf=$(tempfile)
        echo "data_file_directories:" > $tmpf
        for d in $folders; do echo "    - $d/data"; done >> $tmpf

        # delete old `data_file_directories` and replace with new
        cp "$cassandra_config" "$cassandra_config.bck"
        cat "$cassandra_config" | \
        sed -E -n -e'/^data_file_directories: *$/i ___insert_here___' \
            -e '/^data_file_directories: *$/{:lbl;n;/^ +- /blbl}' -e "p" | \
        sed -E -e "/___insert_here___/r $tmpf" -e '/___insert_here___/d' > "$cassandra_config".new && \
        mv "$cassandra_config.new" "$cassandra_config"
        rm $tmpf
    else
        echo "update of $cassandra_config"
    fi
}

# parse command line
while getopts hn opt; do
    case $opt in
        h) usage; exit 0;;
        n) dry_run=y;;
    esac
done

# get count of existing entities
num_devices=$(ssd_devices | wc -l)
num_partitions=$(ssd_partitions | wc -l)
num_mounted=$(mounted_folders | wc -l)
echo "Found $num_devices SSD disks, $num_partitions existing partitions, $num_mounted mounted filesystems"

# check that numbers are consistent
if [ $num_devices -eq 0 ]; then
    echo "No SSD disks found, stopping" 1>&2
    exit 1
fi
if [ $num_partitions -gt 0 -a $num_partitions -ne $num_devices ]; then
    echo "Some but not all partitions exist already, stopping" 1>&2
    exit 1
fi
if [ $num_mounted -gt 0 -a $num_mounted -ne $num_devices ]; then
    echo "Some but not all data folders are mounted already, stopping" 1>&2
    exit 1
fi

# see if we need to make partitions
if [ $num_partitions -eq 0 ]; then
    # make all partitions
    for dev in $(ssd_devices); do
        echo "Make filesystem on device $dev"
        make_partition $dev
        # need to sleep a little until new device appears
        sleep 2
        make_fs $(partition_name $dev)
    done
else
    echo "All partitions already exist (assume filesystems exist too)"
fi

# maybe need to mount them
if [ $num_mounted -eq 0 ]; then
    index=1
    for dev in $(ssd_devices); do
        folder=$(folder_name $index)
        part=$(partition_name $dev)
        echo "Mount partition $part on folder $folder"
        mount_partition $part $folder
        index=$((index + 1))
    done
else
    echo "All data folders are already mounted"
fi

echo "Updating Cassandra config $cassandra_config"
update_cassandra_config
