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
parser.add_argument('-ce', '--compute_exposure', action="store_true",
                    help='Compute unflagged exposure (this may take a bit of time)')

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

with tbl(os.path.join(MSDIR, ZEROGEN_DATA)+"::SOURCE", ack=False) as t:
    codes = t.getcol("CODE")
    intents = t.getcol("CALIBRATION_GROUP")
    SCODE = {sni: c for sni, c in enumerate(codes)}
    SINTENT = {sni: intent for sni, intent in enumerate(intents)}

with tbl(os.path.join(MSDIR, ZEROGEN_DATA)+"::STATE", ack=False) as t:
    states = t.getcol("OBS_MODE")

with tbl(os.path.join(MSDIR, ZEROGEN_DATA)+"::FIELD", ack=False) as t:
    field_names = t.getcol("NAME")
    code = t.getcol("CODE")
    sid = t.getcol("SOURCE_ID")

    FDB = {fn: str(fni) for fni, fn in enumerate(field_names)}
    FCODE = {fn: SCODE[fni] for (fni, fn), s in zip(enumerate(field_names), sid)}

with tbl(os.path.join(MSDIR, ZEROGEN_DATA), ack=False) as t:
    rfields = t.getcol("FIELD_ID")
    rstate = t.getcol("STATE_ID")
    FSTATE = {fn: ", ".join([states[stid] for stid in np.unique(rstate[fni == rfields])])
              for fni, fn in enumerate(field_names)}

if not args.compute_exposure:
    for f in FDB:
        vermeerkat.log.info("\t '{0:s}' index {1:s} marked by observer as "
                            "'{2:s}'".format(
                            f, FDB[f], FSTATE[f]))
else:
    vermeerkat.log.info("Computing exposure... stand by")
    with tbl(os.path.join(MSDIR, ZEROGEN_DATA), ack=False) as t:
        flg = t.getcol("FLAG")
        exp = t.getcol("EXPOSURE")
        exp[flg] = np.nan
        FEXP = {fn: np.nansum(exp[rfields==fni]) for fni, fn in enumerate(field_names)}

    vermeerkat.log.info("The following fields are available:")
    for f in FDB:
        vermeerkat.log.info("\t '{0:s}' index {1:s} marked by observer as "
                            "'{2:s}' with unflagged exposure of {3:02.0f}:{4:02.0f}:{5:02.2f}".format(
                            f, FDB[f], FSTATE[f],
                            FEXP[f] // 3600, FEXP[f] % 3600 // 60, FEXP[f] % 3600 % 60))


vermeerkat.log.info("---End of listr---")
