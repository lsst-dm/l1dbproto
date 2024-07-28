#!/usr/bin/env python3

import re
import sys

_SPLIT_RE = re.compile(r"(?<!\\) ")

_CONFIG = [
    {
        "measurement_pattern": re.compile("cassandra_.*"),
        "tags": {"name"},
        "replace": "{measurement_name}_{name}",
        "remove_tags": {"name", "jolokia_agent_url"},
    }
]


def main():
    """Read metrics in influx line format, transform, and print."""
    for line in sys.stdin:
        try:
            line = process_line(line)
        except Exception:
            sys.stderr.write("Failed to parse line " + line)
        finally:
            # Print original one if parsing/replacing raises.
            sys.stdout.write(line)


def process_line(line):
    measurement, tags, rest = parse_influx(line)
    if measurement is None:
        return line
    measurement, tags = replace(measurement, tags)
    return measurement + "," + tags + " " + rest


def parse_influx(line):
    """Parse influx line and retyurn measurement name, tags and the rest of the
    line.
    """
    # Measurement and tags are comma-spearated and are separated from the
    # rest of the line by a space. Main problem here is that tag values can
    # contain spaces. There are even weirder cases when names of tags can
    # be quoted but I don't know how to handle that. For now assume that
    # we do not have any quoted stuff which should be true for cassandra
    # metrics.
    parts = _SPLIT_RE.split(line, maxsplit=1)
    if len(parts) != 2:
        return None, None, None
    measurement, _, tags_string = parts[0].partition(",")
    rest = parts[1]
    tags = [tag.split("=", 1) for tag in tags_string.split(",")]
    return measurement, tags, rest


def replace(measurement, tags):
    for config in _CONFIG:
        if config["measurement_pattern"].match(measurement):
            tag_names = {tag[0] for tag in tags}
            if config["tags"].issubset(tag_names):
                subs = dict(tags, measurement_name=measurement)
                measurement = config["replace"].format(**subs)
                tags = [tag for tag in tags if tag[0] not in config["remove_tags"]]
    tag_str = ",".join("=".join(tag) for tag in tags)
    return measurement, tag_str


if __name__ == "__main__":
    main()
