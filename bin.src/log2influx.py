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

"""Script to read ap_proto logs and produce InfluxDB input file."""

from __future__ import annotations

import gzip
import logging
import re
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

_tz = None


def _configLogger(verbosity: int) -> None:
    """Configure logging based on verbosity level"""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)


class _Stat:
    def __init__(self, cnt: int = 0, sum: float | int = 0.0):
        self._cnt = cnt
        self._sum = sum

    @property
    def average(self) -> float | None:
        if self._cnt == 0:
            return None
        else:
            return round(self._sum / self._cnt, 3)

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def count(self) -> int:
        return self._cnt

    def add(self, v: float) -> None:
        self._cnt += 1
        self._sum += v

    def __add__(self, v: float) -> _Stat:
        x = _Stat(self._cnt, self._sum)
        x.add(v)
        return x

    def __iadd__(self, v: float) -> _Stat:
        self.add(v)
        return self

    def __str__(self) -> str:
        v = self.average
        if v is None:
            return "NULL"
        else:
            return "{:.6g}".format(v)


# dictionary with context info
_context: dict[str, _Stat | int | None] = defaultdict(lambda: _Stat())
_counters: dict[str, _Stat] = defaultdict(lambda: _Stat())
_timers_real: dict[str, _Stat] = defaultdict(lambda: _Stat())
_timers_cpu: dict[str, _Stat] = defaultdict(lambda: _Stat())


def _sort_lines(iterable: Iterable[str], num: int = 100) -> Iterator[str]:
    """Sort input file according to timestamps.

    Log files are written from multiple process and sometimes ordering can
    be violated. This method sorts inputs according to timestamps.
    """

    def _sort_key(line: str) -> list[str]:
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


def _timestamp(line: str) -> int:
    """Convert timestamp to nanoseconds.

    Timestamp looks like "2020-02-10 18:40:00,148".
    """
    ts = " ".join(line.split()[:2]).replace(",", ".")
    dt = datetime.fromisoformat(ts)
    dt = dt.replace(tzinfo=_tz)
    return int(round(dt.timestamp() * 1e3)) * 1000000


_re_tile1 = re.compile(r" tile=(\d+)x(\d+) ")
_re_tile2 = re.compile(r" tile \((\d+), (\d+)\)")


def _tile(line: str) -> str:
    """Extract tile Id from a line"""
    m = _re_tile1.search(line) or _re_tile2.search(line)
    if m:
        return "{}x{}".format(*m.group(1, 2))
    return "fov"


def _new_visit(line: str) -> None:
    """Initialize data structures for new visit."""
    _context.clear()
    _counters.clear()
    _timers_real.clear()
    _timers_cpu.clear()
    words = line.split()
    visit = int(words[-4])
    ts = _timestamp(line)
    _context["visit"] = visit
    # print(f"visit,tile='fov' start={visit} {ts}")
    print(f"visit start={visit} {ts}")


def _new_tile_visit(line: str) -> None:
    pass
    # words = line.split()
    # visit = int(words[-7])
    # ts = _timestamp(line)
    # tile = _tile(line)
    # print(f"visit,tile='{tile}' start={visit} {ts}")


def _parse_counts(line: str) -> None:
    """
    Parse line with table row counts.
    """
    words = line.split()
    count = int(words[-1])
    table_name = words[-4]
    ts = _timestamp(line)
    print(f"count,table={table_name} value={count} {ts}")


def _parse_timer(line: str) -> tuple[float, float]:
    """Parse timer info"""
    p = line.rfind("\x1b")
    if p > 0:
        line = line[:p]
    words = line.replace("=", " ").split()
    real = float(words[-5])
    cpu = float(words[-3]) + float(words[-1])
    return real, cpu


def _parse_timers(line: str) -> None:
    """Parse line with timer info."""
    real, cpu = _parse_timer(line)
    timer = None
    if "DiaObject select: " in line:
        timer = "obj_select"
    elif "DiaObject truncate: " in line:
        timer = "obj_trunc"
    elif "DiaObjectLast delete: " in line:
        timer = "obj_last_delete"
    elif "DiaObject insert: " in line or " DiaObjectNightly insert: " in line:
        timer = "obj_insert"
    elif "DiaObjectLast insert: " in line:
        timer = "obj_last_insert"
    elif "DiaObjectNightly copy: " in line:
        timer = "obj_daily_copy"
    elif "DiaObjectNightly delete: " in line:
        timer = "obj_daily_delete"
    elif "DiaSource select: " in line:
        timer = "src_select"
    elif "DiaSource insert: " in line:
        timer = "src_insert"
    elif "DiaForcedSource select: " in line:
        timer = "fsrc_select"
    elif "DiaForcedSource insert: " in line:
        timer = "fsrc_insert"
    elif " L1-store: " in line:
        timer = "store"
    elif " VisitProcessing: " in line:
        timer = "visit_proc"
    elif " Finished processing visit " in line and "tile" in line:
        timer = "visit"

    if timer:
        # ts = _timestamp(line)
        # tile = _tile(line)
        _timers_real[timer] += real
        _timers_cpu[timer] += cpu


def _parse_select_count(line: str) -> None:
    """Parse line with counter of selected rows"""
    words = line.split()
    key = None
    if "database found" in line:
        if "forced sources" in line:
            key = "fscr_selected"
            value = int(words[-3])
        elif "sources" in line:
            key = "src_selected"
            value = int(words[-2])
        else:
            key = "obj_selected"
            value = int(words[-2])
    elif "after filtering" in line:
        key = "obj_filtered"
        value = int(words[-2])
    if key:
        # ts = _timestamp(line)
        # tile = _tile(line)
        # print(f"counter,tile='{tile}',counter={key} value={value} {ts}")
        _counters[key] += value


def _parse_queries_count(line: str) -> None:
    """Parse line with counter of select queries"""
    words = line.split()
    key = None
    if "getDiaObjects" in line:
        key = "obj"
        value = int(words[-1])
    elif "_getSources DiaSource" in line:
        key = "src"
        value = int(words[-1])
    elif "_getSources DiaForcedSource" in line:
        key = "fsrc"
        value = int(words[-1])
    if key:
        if "#queries:" in line:
            key += "_queries"
        elif "#partitions:" in line:
            key += "_partitions"
        else:
            return
        # ts = _timestamp(line)
        # tile = _tile(line)
        # print(f"counter,tile='{tile}',counter={key} value={value} {ts}")
        _counters[key] += value


def _parse_store_count(line: str) -> None:
    """Parse line with counter of stored rows"""
    words = line.split()
    key = None
    if words[-1] == "ForcedSources":
        key = "fscr_stored"
        value = int(words[-2])
    elif words[-1] == "Sources":
        key = "src_stored"
        value = int(words[-2])
    elif words[-1] == "Objects":
        key = "obj_stored"
        value = int(words[-2])
    if key:
        # ts = _timestamp(line)
        # tile = _tile(line)
        # print(f"counter,tile='{tile}',counter={key} value={value} {ts}")
        _counters[key] += value


def _end_tile_visit(line: str) -> None:
    """Dump collected information"""
    # visit = _context['visit']
    # ts = _timestamp(line)
    # real, cpu = _parse_timer(line)
    # tile = _tile(line)
    # print(f"visit,tile='{tile}' end={visit},real={real},cpu={cpu} {ts}")


def _end_visit(line: str) -> None:
    """Dump collected information"""
    ts = _timestamp(line)
    visit = _context["visit"]
    _context["visit"] = None
    real, cpu = _parse_timer(line)
    # print(f"visit,tile='fov' end={visit},real={real},cpu={cpu} {ts}")
    print(f"visit end={visit},real={real},cpu={cpu} {ts}")
    for key, stat in _counters.items():
        print(f"counter,counter={key} sum={stat.sum},avg={stat.average} {ts}")
    for key, stat in _timers_real.items():
        print(f"timing,timer={key},kind=real sum={stat.sum},avg={stat.average} {ts}")
    for key, stat in _timers_cpu.items():
        print(f"timing,timer={key},kind=cpu sum={stat.sum},avg={stat.average} {ts}")
    sys.stdout.flush()


# Map line sibstring to method
_dispatch = [
    (re.compile(r"Start processing visit \d+ (?!tile)"), _new_visit),
    (re.compile(r"Start processing visit \d+ tile"), _new_tile_visit),
    (re.compile(" row count: "), _parse_counts),
    (re.compile(": real="), _parse_timers),
    (re.compile(": #partitions: "), _parse_queries_count),
    (re.compile(": #queries: "), _parse_queries_count),
    (re.compile(" database found "), _parse_select_count),
    (re.compile(" after filtering "), _parse_select_count),
    (re.compile(" will store "), _parse_store_count),
    (re.compile(r"Finished processing visit \d+ tile"), _end_tile_visit),  # must be last
    (re.compile(r"Finished processing visit \d+, (?!tile)"), _end_visit),  # must be last
]


def _follow(input: Any, stop_timeout_sec: int = 60) -> Iterator[str]:
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


def main() -> None:
    """Parse logs and generate influxdb line format."""
    descr = "Read ap_proto log and extract few numbers into csv."
    parser = ArgumentParser(description=descr)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="More verbose output, can use several times."
    )
    parser.add_argument("-D", "--database", default="ap_proto", help="Name of the InfluxDB database.")
    parser.add_argument(
        "-u", "--utc", default=False, action="store_true", help="Use UTC for timestamps in the log file."
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

    if args.utc:
        global _tz
        _tz = timezone.utc

    dispatch = _dispatch

    print("# DML")
    print("# CONTEXT-DATABASE: {}".format(args.database))

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
    main()
