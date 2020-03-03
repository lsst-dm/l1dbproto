#!/usr/bin/env jython

"""Script that dumps monitoring metrix from Cassandra server.

Note that this is Jython script and Jython only understands Python 2.7.
"""

from __future__ import print_function

import argparse
from contextlib import closing
import gzip
import logging
import math
import threading
import re
import socket
import sys
import time
import yaml

from javax.management.remote import JMXConnector, JMXConnectorFactory, JMXServiceURL
from javax.management import MBeanServerConnection, MBeanInfo, ObjectName, RuntimeMBeanException
from java.lang import String
from jarray import array


_lock = threading.Lock()  # Jython has no GIL

def _configLogger(verbosity):
    """ configure logging based on verbosity level """

    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)

 
def main():
    parser = argparse.ArgumentParser(description="Dump JMX metrics to InfluxDB DML file")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='More verbose output, can use several times.')
    parser.add_argument("-s", "--server", default="127.0.0.1", help="Server IP address")
    parser.add_argument("-p", "--port", type=int, default=7199, help="Port number")
    parser.add_argument("-u", "--user", default=None, help="User name")
    parser.add_argument("-P", "--password", default=None, help="Password")
    parser.add_argument("-r", "--period", default=60, type=int, help="Period in seconds.")
    parser.add_argument("-o", "--output", default="cassandra_jmx_mon-{host}-{ts}.dat", help="Password")
    parser.add_argument("-C", "--compress", default=False, action="store_true", help="Compress output file.")
    parser.add_argument("-c", "--config", default="cassandra_jmx_mon.yaml",
                        help="YAML configuration file.")
    parser.add_argument("-D", "--database", default="ap_proto", help="Name of the InfluxDB database.")
    parser.add_argument("-H", "--host", default=socket.gethostname(), help="Add host name as a tag.")
    parser.add_argument("-d", "--dump", default=False, action="store_true",
                        help="Dump full list of metrics and attributes")

    args = parser.parse_args()

    # configure logging
    _configLogger(args.verbose)

    # connect to server
    environment = {}
    if args.user or args.password:
        credentials = array([args.user, args.password], String)
        environment = {JMXConnector.CREDENTIALS: credentials}
    jmxServiceUrl = JMXServiceURL('service:jmx:rmi:///jndi/rmi://{}:{}/jmxrmi'.format(args.server, args.port));
    jmxConnector = JMXConnectorFactory.connect(jmxServiceUrl, environment);

    with closing(jmxConnector):
        conn = jmxConnector.getMBeanServerConnection()

        if args.dump:
            _dump(conn)
            return

        config = _read_config(args.config)
        output = args.output.format(ts=time.strftime("%Y%m%dT%H%M%S"), host=(args.host or ""))

        if args.compress:
            output += ".gz"
            out = gzip.open(output, "wb", 9)
        else:
            out = open(output, "w")
        with closing(out):
            logging.info("Writing output to file %s", output)
            print("# DML", file=out)
            print("# CONTEXT-DATABASE: {}".format(args.database), file=out)

            _run(conn, config, args.period, args.host, out)

def _dump(conn):
    """Dump full list of metrics,

    Parameters
    ----------
    conn :
        server connection instance
    """
    names = list(conn.queryNames(ObjectName("*:*"), None))
    names.sort(key=str)
    for oname in names:
        info = conn.getMBeanInfo(oname)
        print("{}".format(oname))
        for attr in info.attributes:
            print("    {}: {}".format(attr.name, attr.type))


def _read_config(yaml_path):
    """Read configuration from YAML file.

    YAML should look like:

        metrics:
          - object: domain:pattern    # this is in format accepted by ObjectName
            attributes: .*              # this is regular expression
          - object: domain2:*
            attributes: attr.*

    Parameters
    ----------
    yaml_path : `str`
        Path to configuration file
    """
    with open(yaml_path) as input:
        config = yaml.load(input, Loader=yaml.FullLoader)
    logging.debug("config: %r", config)
    if "metrics" not in config:
        config["metrics"] = [dict(object="*:*", attributes=".*")]
    for cfg in config["metrics"]:
        cfg["object"] = ObjectName(cfg["object"])
        cfg["attributes"] = re.compile(cfg.get("attributes", ".*"))
    return config

def _run(conn, config, period, host, out):
    """Run monitoring loop until killed.

    Parameters
    ----------
    conn :
        server connection instance
    config : `dict`
        Configuration dictionary
    period : `int`
        Loop period in seconds
    host : `str`
        If not empty then add host name as a tag
    out : `object`
        File object to write output to.
    """

    all_metrics = {}

    # get the list of objects and attributes to monitor
    for metrics in config["metrics"]:
        objectName = metrics["object"]
        names = conn.queryNames(objectName, None)
        for oname in names:
            info = conn.getMBeanInfo(oname)
            logging.debug("checking %s", oname)

            attributes = []
            for attr in info.attributes:
                if metrics["attributes"].match(attr.name):
                    attributes += [attr.name]
                    logging.debug("    %s matches", attr.name)
            if attributes:
                all_attr = all_metrics.get(str(oname), [])
                all_attr += attributes
                all_metrics[str(oname)] = all_attr

    while True:

        now = time.time()
        interval = int(now / period + 1) * period - now
        logging.debug("sleep for %f sec", interval)
        time.sleep(interval)

        for oname, attributes in all_metrics.items():
            now = time.time()
            oname = ObjectName(oname)
            values = conn.getAttributes(oname, attributes)
            line = _attr2influx(oname, values, host)
            if line:
                ts = int(now*1e9)
                with _lock:
                    print(line, ts, file=out)

def _attr2influx(oname, values, host):
    """Convert object name and attribute values to line protocol
    """
    line = None
    keys = dict(oname.keyPropertyList)
    t = keys.pop(u"type", None)
    if t is not None:
        line = t
        n = keys.pop(u"name", None)
        if n:
            line += u"." + n
        for k, v in keys.items():
            v = v.replace(" ", "_")
            line += u",{}={}".format(k, v)
        if host:
            line += u",host={}".format(host)
        vals = []
        for val in values:
            # only include basic numeric types for now
            value = val.value
            if isinstance(value, float) and math.isnan(value):
                continue
            if isinstance(value, (int, long, float)):
                vals += [u"{}={}".format(val.name, value)]
        if vals:
            line += u" " + u",".join(vals)
        else:
            # cannot use measurement without value
            line = None
    return line





if __name__=='__main__':
    main()
