#!/usr/bin/env python

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

"""Script to simulate different partitionings of PPDB.

It an generate different spatial partitionings based on input parameters
and runs a MC simulation of amny pointings to estimate how CCDs map to
partitions.
"""

from argparse import ArgumentParser
import logging
import math
import random
import sys

from lsst.l1dbproto import generators, geom
from lsst.sphgeom import HtmPixelization, Q3cPixelization, Mq3cPixelization, UnitVector3d


def _configLogger(verbosity):
    """ configure logging based on verbosity level """

    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logfmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=logfmt)


FOV = 3.5  # degrees
FOV_rad = FOV * math.pi / 180.


def main():

    descr = 'One-line application description.'
    parser = ArgumentParser(description=descr)
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='More verbose output, can use several times.')
    parser.add_argument('-m', '--mode', default='htm',
                        choices=["htm", "q3c", "mq3c"],
                        help='Partitioning mode, def: %(default)s')
    parser.add_argument('--level', type=int, default=8,
                        help='Pixelization level, def: %(default)s')
    parser.add_argument('-n', '--counts', type=int, default=1000,
                        help='Number of visits, def: %(default)s')
    parser.add_argument('--pixels-per-tile', default=None,
                        help='Output file name for pixels-per-tile')
    parser.add_argument('--tiles-per-pixel', default=None,
                        help='Output file name for pixels-per-tile')
    args = parser.parse_args()

    # configure logging
    _configLogger(args.verbose)

    if args.mode == 'htm':
        pixelator = HtmPixelization(args.level)
        lvlchk = pixelator.level
    elif args.mode == 'q3c':
        pixelator = Q3cPixelization(args.level)
        lvlchk = None
    elif args.mode == 'mq3c':
        pixelator = Mq3cPixelization(args.level)
        lvlchk = pixelator.level

    pixels_per_tile = []
    tiles_per_pixel = []

    for i in range(args.counts):

        pointing_xyz = generators.rand_sphere_xyz(1, -1)[0]
        pointing_v = UnitVector3d(pointing_xyz[0], pointing_xyz[1], pointing_xyz[2])

        rot_ang = random.uniform(0., 2*math.pi)

        tiles = geom.make_square_tiles(FOV_rad, 15, 15, pointing_v, rot_rad=rot_ang)

        # for each tile find all pixels that overlap it
        pixel_tiles = dict()
        for ix, iy, tile in tiles:
            ranges = pixelator.envelope(tile, 1000000)

            tile_pixels = 0
            for i0, i1 in ranges:
                for pixId in range(i0, i1):
                    if lvlchk is not None and lvlchk(pixId) != args.level:
                        logging.warning("tile %dx%d not fully pixelized: %d-%d", ix, iy, i0, i1)
                    else:
                        tile_pixels += 1
                        pixel_tiles.setdefault(pixId, 0)
                        pixel_tiles[pixId] += 1

            pixels_per_tile.append(tile_pixels)

        for count in pixel_tiles.values():
            tiles_per_pixel.append(count)

    print("pixels_per_tile: {:.2f}".format(sum(pixels_per_tile)/len(pixels_per_tile)))
    print("tiles_per_pixel: {:.2f}".format(sum(tiles_per_pixel)/len(tiles_per_pixel)))

    if args.pixels_per_tile:
        with open(args.pixels_per_tile, 'w') as f:
            for num in pixels_per_tile:
                print(num, file=f)

    if args.tiles_per_pixel:
        with open(args.tiles_per_pixel, 'w') as f:
            for num in tiles_per_pixel:
                print(num, file=f)


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
