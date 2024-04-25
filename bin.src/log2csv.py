#!/bin/env python

# This file is part of l1dbproto.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Script to read ap_proto logs and produce CSV file."""

import dataclasses
import gzip
import json
import logging
import re
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any


def _configLogger(verbosity):
    """Configure logging based on verbosity level"""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)


@dataclasses.dataclass
class _Record:

    name: str
    timestamp: float
    tags: dict[str, str| int]
    values: dict[str, Any]
    source: str


class _Stat:
    def __init__(self, cnt=0, sum=0.0):
        self._cnt = cnt
        self._sum = sum

    def value(self):
        if self._cnt == 0:
            return None
        else:
            return self._sum / self._cnt

    def add(self, v):
        self._cnt += 1
        self._sum += v

    def __add__(self, v):
        x = _Stat(self._cnt, self._sum)
        x.add(v)
        return x

    def __iadd__(self, v):
        self.add(v)
        return self

    def __str__(self):
        v = self.value()
        if v is None:
            return "NULL"
        else:
            return "{:.6g}".format(v)



# dictionary with visit statistics, top index is visit, second index is
# stat name
_visits = defaultdict(lambda: defaultdict(lambda: _Stat()))


def _parse_timers(record: _Record) -> None:
    """Parse records with timer info."""
    table_map = {
        "DiaObject": "obj",
        "DiaSource": "src",
        "DiaForcedSource": "fsrc",
        "DiaObjectChunks": "obj_repl",
        "DiaSourceChunks": "src_repl",
        "DiaForcedSourceChunks": "fsrc_repl",
        "DiaObjectLast": "obj_last",
    }
    metrics_map = {
        "select_time": "select",
        "truncate_time": "trunc",
        "delete_time": "delete",
        "insert_time": "insert",
        "tile_store_time": "store",
        "visit_processing_time": "visit_proc",
        "tile_visit_time": "tile_visit",
        "total_visit_time": "visit",
    }

    table_prefix = ""
    if table_name := record.tags.get("table"):
        if short_name := table_map.get(table_name):
            table_prefix = f"{short_name}_"

    if prefix := metrics_map.get(record.name):

        real = record.values["real"]
        cpu = record.values["user"] + record.values["sys"]

        values = _visits[record.tags["visit"]]
        values[f"{table_prefix}{prefix}_real"] += real
        values[f"{table_prefix}{prefix}_cpu"] += cpu


def _parse_select_count(record: _Record) -> None:
    """Parse line with counter of selected rows"""
    values = _visits[record.tags["visit"]]
    if record.name == "read_counts":
        if "forcedsources" in record.values:
            values["fsrc_selected"] += record.values["forcedsources"]
        if "sources" in record.values:
            values["src_selected"] += record.values["sources"]
        values["obj_selected"] += record.values["objects"]
        values["obj_in_fov"] += record.values["objects_filtered"]


def _parse_store_count(record: _Record) -> None:
    """Parse line with counter of stored rows"""
    values = _visits[record.tags["visit"]]
    if record.name == "store_counts":
        values["fsrc_stored"] += record.values["forcedsources"]
        values["src_stored"] += record.values["sources"]
        values["obj_sstored"] += record.values["objects"]


# List of columns (keys in _values dictionary)
_cols = [
    "visit",
    "obj_select_real",
    "obj_select_cpu",
    "obj_last_delete_real",
    "obj_last_insert_real",
    "obj_trunc_real",
    "obj_trunc_cpu",
    "obj_insert_real",
    "obj_repl_insert_real",
    "src_select_real",
    "src_select_cpu",
    "src_insert_real",
    "src_repl_insert_real",
    "fsrc_select_real",
    "fsrc_select_cpu",
    "fsrc_insert_real",
    "fsrc_repl_insert_real",
    "sum_select_real",
    "store_real",
    "store_cpu",
    "tile_visit_real",
    "tile_visit_cpu",
    "visit_proc_real",
    "visit_proc_cpu",
    "visit_real",
    "visit_cpu",
    "obj_selected",
    "src_selected",
    "fsrc_selected",
    "obj_in_fov",
]


def _value(key, stat):
    """Return value for given key, special handling for some
    computed values.
    """
    if key == "sum_select_real":
        sumkeys = ("obj_select_real", "src_select_real", "fsrc_select_real")
        values = [stat[sk].value() for sk in sumkeys]
        values = [value for value in values if value is not None]
        if values:
            return _Stat(1, sum(values))
        else:
            return _Stat()
    return stat[key]


# Flag to print CSV header line once
_header = True


def _end_visit(stats):
    """Dump collected information"""
    global _header
    if _header:
        print(",".join(_cols))
        _header = False
    print(",".join(str(_value(c, stats)) for c in _cols))
    sys.stdout.flush()


# Map line sibstring to method
_dispatch = [
    (re.compile("_time$"), _parse_timers),
    (re.compile("^read_counts$"), _parse_select_count),
    (re.compile("^store_counts$"), _parse_store_count),
]


def _follow(input, stop_timeout_sec=60):
    """Implement reading from a file that is being written into (tail -F)."""
    stop_re = re.compile("Stopping MPI tile processes")

    last_read = time.time()
    buffer = ""
    block_size = 64 * 1024
    stop = False
    while True:
        if "\n" not in buffer:
            if stop:
                return
            # buffer is empty or only has partial line, read more
            more_data = input.read(block_size)
            if not more_data:
                if time.time() > last_read + stop_timeout_sec:
                    # no new data in a while, stop
                    stop = True
                else:
                    # wait a little and restart loop
                    time.sleep(0.1)
                continue
            else:
                buffer += more_data
                last_read = time.time()

        idx = buffer.find("\n")
        if idx >= 0:
            line = buffer[: idx + 1]
            buffer = buffer[idx + 1 :]

            if stop_re.match(line):
                # stop after reading remaing lines
                stop = True

            yield line


def main():
    """Parse log files and generate CSV output."""
    descr = "Read ap_proto log and extract few numbers into csv."
    parser = ArgumentParser(description=descr)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="More verbose output, can use several times."
    )
    parser.add_argument(
        "-s", "--short", action="store_true", default=False, help="Save fewer columns into CSV file."
    )
    parser.add_argument(
        "-F", "--follow", default=False, action="store_true", help="Continue reading as file grows."
    )
    parser.add_argument(
        "--follow-timeout",
        default=60,
        type=int,
        metavar="SECONDS",
        help="Max number of seconds to wait for file to grow, def: %(default)s.",
    )
    parser.add_argument(
        "file", nargs="+", help='Name of input log file, optionally compressed, use "-" for stdin'
    )
    args = parser.parse_args()

    # configure logging
    _configLogger(args.verbose)

    dispatch = _dispatch

    if args.short:
        global _cols
        _cols = [
            "visit",
            "visit_real",
            "visit_cpu",
            "visit_proc_real",
            "visit_proc_cpu",
        ]

    # open each file in order
    for input in args.file:
        if input == "-":
            input = sys.stdin
        else:
            f = gzip.open(input, "rt")
            try:
                f.read(1)
                f.seek(0)
                input = f
            except IOError:
                input = open(input, "rt")
        if args.follow:
            input = _follow(input, args.follow_timeout)

        for line in input:
            marker = " apdb_metrics: "
            pos = line.find(marker)
            if pos > 0:
                data = line[pos + len(marker):]
                data_dict = json.loads(data)
                assert isinstance(data_dict, dict)
                record = _Record(**data_dict)
                for match, method in dispatch:
                    # if line matches then call corresponding method
                    if match.search(record.name):
                        method(record)

                if len(_visits) > 2:
                    visit = min(_visits)
                    stats = _visits.pop(visit)
                    stats["visit"] = visit
                    _end_visit(stats)

        # dump remaining stats
        for visit in sorted(_visits):
            stats = _visits.pop(visit)
            stats["visit"] = visit
            _end_visit(stats)


#
#  run application when imported as a main module
#
if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
