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

"""Script to read ap_proto logs and produce CSV file.
"""

import gzip
import logging
import re
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict


def _configLogger(verbosity):
    """Configure logging based on verbosity level"""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)


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


# dictionary with visit statistics
_values = defaultdict(lambda: _Stat())


def _sort_lines(iterable, num=100):
    """Sort input file according to timestamps.

    Log files are written from multiple process and sometimes ordering can
    be violated. This method sorts inputs according to timestamps.
    """

    def _sort_key(line):
        # extract timestamp
        return line.split()[:2]

    lines = []
    for line in iterable:
        lines.append(line)
        if len(lines) > num:
            lines.sort(key=_sort_key)
            yield lines.pop(0)
    lines.sort(key=_sort_key)
    yield from lines


def _new_visit(line):
    """
    Initialize data structures for new visit.
    """
    _values.clear()
    words = line.split()
    visit = int(words[-4])
    _values["visit"] = visit


def _parse_counts(line):
    """
    Parse line with table row counts.
    """
    words = line.split()
    count = int(words[-1])
    table_name = words[-4]
    if table_name.endswith("DiaObject"):
        _values["obj_count"] += count
    elif table_name.endswith("DiaSource"):
        _values["src_count"] += count
    elif table_name.endswith("DiaForcedSource"):
        _values["fsrc_count"] += count


def _parse_timers(line):
    """
    Parse line with timer info.
    """
    p = line.rfind("\x1b")
    if p > 0:
        line = line[:p]
    words = line.replace("=", " ").split()
    real = float(words[-5])
    cpu = float(words[-3]) + float(words[-1])
    if "DiaObject select: " in line:
        _values["obj_select_real"] += real
        _values["obj_select_cpu"] += cpu
    elif "DiaObject truncate: " in line:
        _values["obj_trunc_real"] += real
        _values["obj_trunc_cpu"] += cpu
    elif "DiaObjectLast delete: " in line:
        _values["obj_last_delete_real"] += real
        _values["obj_last_delete_cpu"] += cpu
    elif "DiaObject insert: " in line or " DiaObjectNightly insert: " in line:
        _values["obj_insert_real"] += real
        _values["obj_insert_cpu"] += cpu
    elif "DiaObjectLast insert: " in line:
        _values["obj_last_insert_real"] += real
        _values["obj_last_insert_cpu"] += cpu
    elif "DiaObjectNightly copy: " in line:
        _values["obj_daily_copy_real"] += real
        _values["obj_daily_copy_cpu"] += cpu
    elif "DiaObjectNightly delete: " in line:
        _values["obj_daily_delete_real"] += real
        _values["obj_daily_delete_cpu"] += cpu
    elif "DiaSource select: " in line:
        _values["src_select_real"] += real
        _values["src_select_cpu"] += cpu
    elif "DiaSource insert: " in line:
        _values["src_insert_real"] += real
        _values["src_insert_cpu"] += cpu
    elif "DiaForcedSource select: " in line:
        _values["fsrc_select_real"] += real
        _values["fsrc_select_cpu"] += cpu
    elif "DiaForcedSource insert: " in line:
        _values["fsrc_insert_real"] += real
        _values["fsrc_insert_cpu"] += cpu
    elif " L1-store: " in line:
        _values["store_real"] += real
        _values["store_cpu"] += cpu
    elif " VisitProcessing: " in line:
        _values["visit_proc_real"] += real
        _values["visit_proc_cpu"] += cpu
    elif " Finished processing visit " in line and " tile " not in line:
        _values["visit_real"] += real
        _values["visit_cpu"] += cpu


def _parse_select_count(line):
    """
    Parse line with counter of selected rows
    """
    words = line.split()
    if "database found" in line:
        if "forced sources" in line:
            _values["fsrc_selected"] += int(words[-3])
        elif "sources" in line:
            _values["src_selected"] += int(words[-2])
        else:
            _values["obj_selected"] += int(words[-2])
    elif "after filtering" in line:
        _values["obj_in_fov"] += int(words[-2])


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
    "obj_insert_cpu",
    "src_select_real",
    "src_select_cpu",
    "src_insert_real",
    "src_insert_cpu",
    "fsrc_select_real",
    "fsrc_select_cpu",
    "fsrc_insert_real",
    "fsrc_insert_cpu",
    "select_real",
    "store_real",
    "store_cpu",
    "visit_proc_real",
    "visit_proc_cpu",
    "visit_real",
    "visit_cpu",
    "obj_selected",
    "src_selected",
    "fsrc_selected",
    "obj_in_fov",
    "obj_count",
    "src_count",
    "fsrc_count",
]


def _value(key):
    """Return value for given key, special handling for some
    computed values.
    """
    if key == "select_real":
        sumkeys = ("obj_select_real", "src_select_real", "fsrc_select_real")
        values = [_values[sk].value() for sk in sumkeys]
        values = [value for value in values if value is not None]
        if values:
            return _Stat(1, sum(values))
        else:
            return _Stat()
    return _values[key]


# Flag to print CSV header line once
_header = True


def _end_visit(line):
    """
    Dump collected information
    """
    global _header
    if _header:
        print(",".join(_cols))
        _header = False
    print(",".join(str(_value(c)) for c in _cols))
    sys.stdout.flush()


# Map line sibstring to method
_dispatch = [
    (re.compile(r"Start processing visit \d+ (?!tile)"), _new_visit),
    (re.compile(" row count: "), _parse_counts),
    (re.compile(": real="), _parse_timers),
    (re.compile(" database found "), _parse_select_count),
    (re.compile(" after filtering "), _parse_select_count),
    (re.compile(r"Finished processing visit \d+, (?!tile)"), _end_visit),  # must be last
]


_daily_dispatch = [
    (re.compile(r"Start processing visit \d+ (?!tile)"), _new_visit),
    (re.compile(": real="), _parse_timers),
    (re.compile("Done with daily activities"), _end_visit),  # must be last
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
        "-d", "--daily", action="store_true", default=False, help="Extract numbers from daily activities."
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
            "obj_count",
            "src_count",
            "fsrc_count",
        ]

    if args.daily:
        dispatch = _daily_dispatch
        _cols = ["visit", "obj_daily_copy_real", "obj_daily_delete_real"]

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
        for line in _sort_lines(input):
            for match, method in dispatch:
                # if line matches then call corresponding method
                if match.search(line):
                    method(line)


#
#  run application when imported as a main module
#
if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
