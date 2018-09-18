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

"""Unit test for L1db class.
"""

import datetime
import logging
import os
import unittest

import lsst.afw.table as afwTable
from lsst.l1dbproto import (L1db, L1dbConfig, make_minimal_dia_object_schema,
                            make_minimal_dia_source_schema)
from lsst.sphgeom import Angle, Circle, HtmPixelization, Vector3d, UnitVector3d
from lsst.geom import SpherePoint
import lsst.utils.tests


# HTM indexing level used in the unit tests
HTM_LEVEL = 20


def _makePixelRanges():
    """Generate pixel ID ranges for some envelope region"""
    pointing_v = UnitVector3d(1., 1., -1.)
    fov = 0.05 # radians
    region = Circle(pointing_v, Angle(fov/2))
    pixelator = HtmPixelization(HTM_LEVEL)
    indices = pixelator.envelope(region, 128)
    return indices.ranges()


def _makeObjectCatalog(pixel_ranges):
    """make a catalog containing a bunch of DiaObjects inside pixel envelope.
    
    The number of created records will be equal number of ranges (one object
    per pixel range). Coodirnates of the created objects are not usable.
    """
    # make afw catalog
    schema = make_minimal_dia_object_schema()
    catalog = afwTable.SourceCatalog(schema)

    # make small bunch of records, one entry per one pixel range,
    # we do not care about coordinates here, in current implementation
    # they are not used in any query
    v3d = Vector3d(1., 1., -1.)
    sp = SpherePoint(v3d)
    for oid, (start, end) in enumerate(pixel_ranges):
        record = catalog.addNew()
        record.set("id", oid)
        record.set("pixelId", start)
        record.set("coord_ra", sp.getRa())
        record.set("coord_dec", sp.getDec())

    return catalog

class L1dbTestCase(unittest.TestCase):
    """A test case for L1db class
    """

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)

    def test_makeSchema(self):
        """Test for making an instance of L1db using in-memory sqlite engine.
        """
        # sqlite does not support default READ_COMMITTED, for in-memory
        # database have to use connection pool
        config = L1dbConfig(db_url="sqlite://",
                            isolation_level="READ_UNCOMMITTED")
        l1db = L1db(config)
        l1db.makeSchema()

    def test_emptyGetsBaseline0months(self):
        """Test for getting data from empty database.

        All get() methods should return empty results, only useful for
        checking that code is not broken.
        """
        # set read_sources_months to 0 so that Forced/Sources are None
        config = L1dbConfig(db_url="sqlite:///",
                            isolation_level="READ_UNCOMMITTED",
                            read_sources_months=0,
                            read_forced_sources_months=0)
        l1db = L1db(config)
        l1db.makeSchema()

        pixel_ranges = _makePixelRanges()
        visit_time = datetime.datetime.now()

        # get objects by region
        res = l1db.getDiaObjects(pixel_ranges)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), 0)

        # get sources by region
        res = l1db.getDiaSourcesInRegion(pixel_ranges, visit_time)
        self.assertIs(res, None)

        # get sources by object ID, empty object list
        res = l1db.getDiaSources([], visit_time)
        self.assertIs(res, None)

        # get forced sources by object ID, empty object list
        res = l1db.getDiaForcedSources([], visit_time)
        self.assertIs(res, None)

    def test_emptyGetsBaseline(self):
        """Test for getting data from empty database.

        All get() methods should return empty results, only useful for
        checking that code is not broken.
        """
        # use non-zero months for Forced/Source fetching
        config = L1dbConfig(db_url="sqlite:///",
                            isolation_level="READ_UNCOMMITTED",
                            read_sources_months=12,
                            read_forced_sources_months=12)
        l1db = L1db(config)
        l1db.makeSchema()

        pixel_ranges = _makePixelRanges()
        visit_time = datetime.datetime.now()

        # get objects by region
        res = l1db.getDiaObjects(pixel_ranges)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), 0)

        # get sources by region
        res = l1db.getDiaSourcesInRegion(pixel_ranges, visit_time)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), 0)

        # get sources by object ID, empty object list, should return None
        res = l1db.getDiaSources([], visit_time)
        self.assertIs(res, None)

        # get sources by object ID, non-empty object list
        res = l1db.getDiaSources([1, 2, 3], visit_time)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), 0)

        # get forced sources by object ID, empty object list
        res = l1db.getDiaForcedSources([], visit_time)
        self.assertIs(res, None)

        # get sources by object ID, non-empty object list
        res = l1db.getDiaForcedSources([1, 2, 3], visit_time)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), 0)

    def test_emptyGetsObjectLast(self):
        """Same as bove but using DiaObjectLast table.

        All get() methods should return empty results, only useful for
        checking that code is not broken.
        """
        # don't care about sources.
        config = L1dbConfig(db_url="sqlite:///",
                            isolation_level="READ_UNCOMMITTED",
                            dia_object_index="last_object_table")
        l1db = L1db(config)
        l1db.makeSchema()

        pixel_ranges = _makePixelRanges()

        # get objects by region
        res = l1db.getDiaObjects(pixel_ranges)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), 0)

    def test_storeObjectsBaseline(self):
        """Store and retrieve DiaObjects."""
        # don't care about sources.
        config = L1dbConfig(db_url="sqlite:///",
                            isolation_level="READ_UNCOMMITTED",
                            dia_object_index="baseline")
        l1db = L1db(config)
        l1db.makeSchema()

        pixel_ranges = _makePixelRanges()
        visit_time = datetime.datetime.now()

        # make afw catalog with Objects
        catalog = _makeObjectCatalog(pixel_ranges)

        # store catalog
        l1db.storeDiaObjects(catalog, visit_time)

        # read it back and check sizes
        res = l1db.getDiaObjects(pixel_ranges)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), len(catalog))

    def test_storeObjectsLast(self):
        """Store and retrieve DiaObjects using DiaObjectLast table."""
        # don't care about sources.
        config = L1dbConfig(db_url="sqlite:///",
                            isolation_level="READ_UNCOMMITTED",
                            dia_object_index="last_object_table",
                            object_last_replace=True)
        l1db = L1db(config)
        l1db.makeSchema()

        pixel_ranges = _makePixelRanges()
        visit_time = datetime.datetime.now()

        # make afw catalog with Objects
        catalog = _makeObjectCatalog(pixel_ranges)

        # store catalog
        l1db.storeDiaObjects(catalog, visit_time)

        # read it back and check sizes
        res = l1db.getDiaObjects(pixel_ranges)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), len(catalog))

    def test_storeSources(self):
        """Store and retrieve DiaSources."""
        # don't care about sources.
        config = L1dbConfig(db_url="sqlite:///",
                            isolation_level="READ_UNCOMMITTED",
                            read_sources_months=12,
                            read_forced_sources_months=12)
        l1db = L1db(config)
        l1db.makeSchema()

        pixel_ranges = _makePixelRanges()
        visit_time = datetime.datetime.now()

        # have to store Objects first
        objects = _makeObjectCatalog(pixel_ranges)
        l1db.storeDiaObjects(objects, visit_time)

        # make some sources
        schema = make_minimal_dia_source_schema()
        catalog = afwTable.BaseCatalog(schema)
        oids = []
        for sid, obj in enumerate(objects):
            record = catalog.addNew()
            record.set("id", sid)
            record.set("ccdVisitId", 1)
            record.set("diaObjectId", obj['id'])
            record.set("parent", 0)
            record.set("coord_ra", obj['coord_ra'])
            record.set("coord_dec", obj['coord_dec'])
            record.set("flags", 0)
            record.set("pixelId", obj['pixelId'])
            oids.append(obj['id'])

        # save them
        l1db.storeDiaSources(catalog)

        # read it back and check sizes
        res = l1db.getDiaSourcesInRegion(pixel_ranges, visit_time)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), len(catalog))

        # read it back using different method
        res = l1db.getDiaSources(oids, visit_time)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), len(catalog))

    def test_storeForcedSources(self):
        """Store and retrieve DiaForcedSources."""
        # don't care about sources.
        config = L1dbConfig(db_url="sqlite:///",
                            isolation_level="READ_UNCOMMITTED",
                            read_sources_months=12,
                            read_forced_sources_months=12)
        l1db = L1db(config)
        l1db.makeSchema()

        pixel_ranges = _makePixelRanges()
        visit_time = datetime.datetime.now()

        # have to store Objects first
        objects = _makeObjectCatalog(pixel_ranges)
        l1db.storeDiaObjects(objects, visit_time)

        # make some sources
        schema = afwTable.Schema()
        schema.addField("diaObjectId", "L")
        schema.addField("ccdVisitId", "L")
        schema.addField("flags", "L")
        catalog = afwTable.BaseCatalog(schema)
        oids = []
        for obj in objects:
            record = catalog.addNew()
            record.set("diaObjectId", obj['id'])
            record.set("ccdVisitId", 1)
            record.set("flags", 0)
            oids.append(obj['id'])

        # save them
        l1db.storeDiaForcedSources(catalog)

        # read it back and check sizes
        res = l1db.getDiaForcedSources(oids, visit_time)
        self.assertIsInstance(res, afwTable.SourceCatalog)
        self.assertEqual(len(res), len(catalog))


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
