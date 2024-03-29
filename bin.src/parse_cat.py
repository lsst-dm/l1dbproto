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

"""Application to read and parse schema from cat package.

Reads and pasres schema and generates code that can be pasted into Python
code or saved as YAML file.
"""

from argparse import ArgumentParser
import re
import os
import sys

import yaml


# parsing states
STATE_TOP = 0
STATE_TABLE = 1
STATE_TABLE_SKIP = 2
STATE_DESCR = 3

# Tables that we need for APDB
TABLES = ['DiaObject', 'SSObject', 'DiaSource', 'DiaForcedSource', 'DiaObject_To_Object_Match']


def main():

    descr = 'Read and parse L1 schema in cat package.'
    parser = ArgumentParser(description=descr)
    parser.add_argument('-a', '--all', default=False, action="store_true",
                        help='Generate schema for all tables')
    parser.add_argument('-y', '--yaml', default=False, action="store_true",
                        help='Generate YAML schema file')
    parser.add_argument('file', help='Name of input file')
    args = parser.parse_args()

    state = STATE_TOP
    tables = []
    table = {}
    for line in open(args.file):
        line = line.strip()

        if state == STATE_TOP:
            # look for CREATE TABLE
            if line.startswith("CREATE TABLE"):
                name = line.split()[2]
                if args.all or name in TABLES:
                    table = dict(table=name)
                    state = STATE_TABLE
                else:
                    state = STATE_TABLE_SKIP
                continue

        if state == STATE_TABLE_SKIP:
            # look for closing paren, ignore anything else
            if line.startswith(')'):
                state = STATE_TOP
            continue

        if state == STATE_DESCR:
            # parsing multi-line <descr> tag in comments
            if line.startswith("--"):
                state = parse_comment(line, table, state)
                continue
            else:
                # expecting a comment, but got something else,
                # get back to table parsing in next 'if'
                state = STATE_TABLE

        if state == STATE_TABLE:
            if line.startswith(')'):
                # end of table definition
                tables.append(table)
                table = {}
                state = STATE_TOP
            elif line.startswith('('):
                # start of columns definitions
                pass
            elif line.startswith("--"):
                # comments, get metadata from it
                state = parse_comment(line, table, state)
            else:
                # either column or index definition
                index = parse_index(line)
                if index:
                    table.setdefault('indices', []).append(index)
                else:
                    column = parse_column(line)
                    if column:
                        table.setdefault('columns', []).append(column)
            continue

    # done with parsing, dump everything
    if args.yaml:
        dump_yaml(tables, args)
    else:
        for table in tables:
            dump_pytables(table)


def parse_comment(line, table, state):
    """Extract meta information from comment line.

    Parameters
    ----------
    line : str
        Line with comment, usually starts with '--' string
    table : `dict`
        Table definition for current table.
    state : `int`
        Current parser state.

    Returns
    -------
    New parser state.
    """
    # update either table itself or last column
    to_update = table
    if 'columns' in table:
        to_update = table['columns'][-1]

    line = line.lstrip('-')
    if state == STATE_DESCR:
        # continue parsing <descr> tag
        idx = line.find("</descr>")
        if idx >= 0:
            # closing tag is here
            state = STATE_TABLE
            line = line[:idx]
        line = line.strip()
        if line:
            to_update['description'] += ' ' + line
    else:
        # order is important here
        tests = [r"<descr>(?P<description>.*)</descr>",
                 r"<descr>(?P<description>.*)",
                 r"<unit>(?P<unit>.*)</unit>",
                 r"<ucd>(?P<ucd>.*)</ucd>"]
        for test in tests:
            match = re.search(test, line)
            if match:
                to_update.update(match.groupdict())
                if "<descr>" in test and "</descr>" not in test:
                    # continues on next line
                    state = STATE_DESCR
                break
    return state


def parse_column(line):
    """Try to parse column definition.

    Parameters
    ----------
    line : str
        Line with column definition.

    Returns
    -------
    None if line is not a column definition, otherwise returns a dict with
    column description.
    """
    line = line.rstrip(',')
    words = line.split()
    if len(words) < 2:
        return None
    column = words[0]
    col_type = words[1]
    nullable = "NOT NULL" not in line
    column = dict(name=column, type=col_type, nullable=nullable)
    try:
        idx = words.index('DEFAULT')
        column['default'] = words[idx + 1]
    except ValueError:
        pass
    return column


def parse_index(line):
    """Try to parse line as an index definition

    Returns None if it is not an index definition, otherwise returns dict.
    """
    pkey_re = re.compile(r"\s*PRIMARY\s+KEY\s+((?P<name>\w+)\s+)?\((?P<columns>.*)\).*")
    unique_re = re.compile(r"\s*UNIQUE\s+(KEY\s+|INDEX\s+)?((?P<name>\w+)\s+)?\((?P<columns>.*)\).*")
    index_re = re.compile(r"\s*((KEY|INDEX)\s+)((?P<name>\w+)\s+)?\((?P<columns>.*)\).*")

    tests = dict(PRIMARY=pkey_re,
                 UNIQUE=unique_re,
                 INDEX=index_re)
    index = None
    for idx_type, test in tests.items():
        match = test.match(line)
        if match:
            index = match.groupdict()
            index['type'] = idx_type
            break

    if index:
        columns = index['columns'].split(',')
        index['columns'] = [c.strip() for c in columns]

    return index


def dump_pytables(table):
    """Dump table definition as a Python code for SQLAlchemy.

    Parameters
    ----------
    table : `dict`
        Table definition.
    """

    print("\ntable = Table('{}', self._metadata,".format(table['table']))
    for column in table['columns']:
        name = column['name']
        col_type = column['type']
        nullable = column['nullable']
        default = column.get('default')
        args = [repr(name), col_type, 'nullable=' + str(nullable)]
        if default is not None:
            args += ['default=' + default]
        print("              Column({0}),".format(', '.join(args)))
    print("              )")

    # also dump full list of all non-nullable column names
    print("-- all columns for table {0}".format(table['table']))
    print(' '.join(c['name'] for c in table['columns']))
    print("-- non-nullable columns for table {0}".format(table['table']))
    print(' '.join(c['name'] for c in table['columns'] if not c['nullable']))


def dump_yaml(tables, args):
    """Dump list of tables in YAML format.

    Parameters
    ----------
    tables : `list` of `dict`
        List of table definitions
    args : `Namespace`
        Command line arguments
    """

    # this is to output dict keys in the same order they were added to the dicts
    # (assuming Python3 ordered dicts):
    # https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
    yaml.add_representer(dict, represent_ordereddict)

    print("# Generated from {}".format(os.path.abspath(args.file)))
    print("# by {} script".format(sys.argv[0]))
    yaml.dump_all(tables, stream=sys.stdout, default_flow_style=False)


def represent_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


#
#  run application when imported as a main module
#
if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
