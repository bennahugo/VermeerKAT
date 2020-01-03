from pyrap.tables import table as tbl
import os
import sys
import argparse
import shutil
import vermeerkat
import numpy as np

parser = argparse.ArgumentParser("MeerKAT BasicApplyTransfer (BAT) pipeline")
parser.add_argument('--input_dir', dest='input_directory', metavar='<input directory>',
                    action='store', default='input',
                    help='directory to read input data with e.g. antenna table and beam files')
parser.add_argument('--output_dir', dest='output_directory', metavar='<output directory>',
                    action='store', default='output', help='directory to write output data')
parser.add_argument('--msdir_dir', dest='msdir_directory', metavar='<msdir directory>',
                    action='store', default='msdir', help='directory to store measurement sets')
parser.add_argument('msprefix', metavar='<measurement set name prefix>',
                    help='Prefix of measurement set name as it appears relative to msdir. This must NOT be a '
                         'path prefix')

args = parser.parse_args(sys.argv[2:])

INPUT = args.input_directory
MSDIR = args.msdir_directory
OUTPUT = args.output_directory
vermeerkat.log.info("Directory '{0:s}' is used as input directory".format(INPUT))
vermeerkat.log.info("Directory '{0:s}' is used as output directory".format(OUTPUT))
vermeerkat.log.info("Directory '{0:s}' is used as msdir directory".format(MSDIR))

PREFIX = args.msprefix
ZEROGEN_DATA = PREFIX + ".ms"

vermeerkat.log.info("Dataset '{0:s}' to be used throughout".format(ZEROGEN_DATA))

with tbl(os.path.join(MSDIR, ZEROGEN_DATA)+"::ANTENNA", ack=False) as t:
    name = t.getcol("NAME")
    pos = t.getcol("POSITION")

vermeerkat.log.info("The following antennae are available:")
for ni, (n, p) in enumerate(zip(name, pos)):
    vermeerkat.log.info("\t {0:d}: '{1:s}' with position [{2:s}]".format(
                        ni, n, ",".join(map(str, p))))


vermeerkat.log.info("---End of listr---")
