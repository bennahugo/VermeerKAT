from __future__ import absolute_import

import stimela
from stimela.dismissable import dismissable as sdm
import os
from collections import OrderedDict
from . import diagnostics as dgn
from vermeerkat import log
import argparse
import sys

parser = argparse.ArgumentParser("MeerKAT boresight polarization calibration pipeline")
parser.add_argument('--input_dir', dest='input_directory', metavar='<input directory>', action='store', default='input', help='directory to read input data with e.g. antenna table and beam files; default: input)')
parser.add_argument('--output_dir', dest='output_directory', metavar='<output directory>', action='store', default='output', help='directory to write output data')
parser.add_argument('--msdir_dir', dest='msdir_directory', metavar='<msdir directory>', action='store', default='msdir', help='directory to store measurement sets')
parser.add_argument('--bp', dest='bandpass_field', metavar='<bandpass fields>', action='store', nargs='+', help='full list of unpolarized bandpass fields', default=[])
parser.add_argument('--polcal', dest='polcal_field', metavar='<polarization crosshand fields>', action='store', nargs='+', help='full list of fields (polarized) to use for crosshand delays and angles', default=[])
parser.add_argument('--refant', dest='refant', metavar='<reference antenna>', action='store', type=str, help='reference antenna')
parser.add_argument('--pol_solint', dest='polarization_solint', metavar='<polarization_solint>', action='store', type=str, default="inf", help='time-invariant polarization angle solution intervals (default infinite, combining scans)')
parser.add_argument('--timegain_solint', dest='timegain_solint', metavar='<timegain_solint>', action='store', type=str, default="inf", help='frequency-invarient gain solution intervals (default 1 per scan)')
parser.add_argument('--pol_solchanavg', dest='pol_solchanavg', metavar='<pol_solchanavg>', action='store', type=int, default=1, help='Channels to average to compute polarization solutions')
parser.add_argument('--pol_mstimeavg', dest='pol_mstimeavg', metavar='<pol_mstimeavg>', action='store', type=str, default="8s", help="Time average for polarization measurement set")
parser.add_argument('--uv_cutoff', dest='uv_cutoff', metavar='<uv_cutoff>', action='store', type=str, default="100~10000m", help="UVrange to use to compute solutions")
parser.add_argument('msprefix',
                    metavar='<measurement set name prefix>',
                    help='Prefix of measurement set name as it appears relative to msdir. This must NOT be a '
                         'path prefix')
parser.add_argument("--containerization", dest="containerization", default="docker",
                    help="Containerization technology to use. See your stimela installation for options")


args = parser.parse_args(sys.argv[2:])
INPUT = args.input_directory
MSDIR = args.msdir_directory
OUTPUT = args.output_directory


PREFIX = args.msprefix
COMB_MS = PREFIX + ".ms"
BP_CAL_MS = PREFIX + "PREPOL.PREAVG.ms"
POL_CAL_MS = PREFIX + ".POL.ms"
KX = PREFIX + ".KX"
Xref = PREFIX + ".Xref"
Xf = PREFIX + ".Xf"
Dref = PREFIX + ".Dref"
Df = PREFIX + ".Df"

REFANT = args.refant
pol_solchanavg = args.pol_solchanavg
pol_mstimeavg = args.pol_mstimeavg
phase_solint = args.polarization_solint
timegain_solint = args.timegain_solint
uv_cutoff = args.uv_cutoff

field_list = {}
for bp in args.bandpass_field:
    field_list[len(field_list.keys())] = bp
for pc in args.polcal_field:
    field_list[len(field_list.keys())] = pc

assert len(args.bandpass_field) >= 1, "Need one or more bandpass fields"
assert len(args.polcal_field) >= 1, "Need one ore more polarized fields"

delay_calibrators = [field_list[fk] for fk in field_list.keys()]

# for calibration of diagonal bandpass solutions
# as well as crosshand leakage solutions
bp_calibrators={"PKS1934-638":{"standard":"Perley-Butler 2010",
                               "fluxdensity":None,
                               "spix":None,
                               "reffreq":None,
                               "polindex":None,
                               "polangle":None}
               }
bp_calibrators["PKS1934-63"] = bp_calibrators["PKS1934-638"]
bp_calibrators["J1939-6342"] = bp_calibrators["PKS1934-638"]
bp_calibrators["J1938-6341"] = bp_calibrators["PKS1934-638"]

fexcl = [bp for bp in bp_calibrators.keys() if bp not in args.bandpass_field]
for f in fexcl:
    bp_calibrators.pop(f)

# for calibration of crosshand phases
# Perley Butler standards + NRAO polarization models
polarized_calibrators={"3C138":{"standard":"manual",
                                "fluxdensity":[8.4012,-1,-1,-1],
                                "spix":[-0.54890527955337987, -0.069418066176041668, -0.0018858519926001926],
                                "reffreq":"1.45GHz",
                                "polindex":[0.075],
                                "polangle":[-0.19199]},
                       "3C286":{"standard":"manual",
                                "fluxdensity":[14.918703,-1,-1,-1],
                                "spix":[-0.50593909976893958,-0.070580431627712076,0.0067337240268301466],
                                "reffreq":"1.45GHz",
                                "polindex":[0.095],
                                "polangle":[0.575959]},
                      }
polarized_calibrators["J1331+3030"] = polarized_calibrators["3C286"]
polarized_calibrators["J0521+1638"] = polarized_calibrators["3C138"]

fexcl = [pc for pc in polarized_calibrators.keys() if pc not in args.polcal_field]
for f in fexcl:
    polarized_calibrators.pop(f)

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z
all_cals = merge_two_dicts(bp_calibrators, polarized_calibrators)

field_ids = []
def __extract_fields(ms, field_list, field_ids):
    from pyrap.tables import table as tbl
    with tbl("%s/%s::FIELD" % (MSDIR, ms)) as fields:
        fnames = fields.getcol("NAME")
    if len(set([field_list[f] for f in field_list]).difference(set(fnames))) != 0:
        raise ValueError("Fields {0:s} selected but only {1:s} available.".format(", ".join(["'{0:s}'".format(field_list[f]) for f in field_list]),
                                                                                  ", ".join(["'{0:s}'".format(f) for f in fnames])))
    field_ids += [fnames.index(field_list[f]) for f in field_list]
__extract_fields(ms = COMB_MS, field_list = field_list, field_ids = field_ids)

log.info("Input directory: %s" % INPUT)
log.info("MS directory: %s" % MSDIR)
log.info("Output directory: %s" % OUTPUT)
log.info("Reference antenna for delay calibration: %s" % REFANT)
log.info("Solution interval for time-invarient crosshand phase solutions: %s" % phase_solint)
log.info("Solution interval for frequency-invarient gains: %s" % timegain_solint)


log.info("Fields being used for solving:")
for fi, f in enumerate([field_list[fk] for fk in field_list.keys()]):
    label = " (BP)" if f in args.bandpass_field else " (POL)" if f in args.polcal_field else ""
    log.info("\t %d: %s%s" % (fi, f, label))

try:
    input = raw_input
except NameError:
    pass

while True:
    r = input("Is this configuration correct? (Y/N) >> ").lower()
    if r == "y":
        break
    elif r == "n":
        sys.exit(0)
    else:
        continue
stimela.register_globals()

recipe = stimela.Recipe('MEERKAT FleetingPol: Interferometric boresight polarization calibration', ms_dir=MSDIR, JOB_TYPE=args.containerization)

recipe.add("cab/casa_oldsplit", "split_avg_data", {
    "vis": COMB_MS,
    "outputvis": BP_CAL_MS,
    "datacolumn": "corrected",
    "field": "",
    "timebin": pol_mstimeavg,
    "width": pol_solchanavg,
},
input=INPUT, output=OUTPUT, label="split_avg_data")

recipe.add("cab/casa_clearcal", "clearcal_avg", {
    "vis": BP_CAL_MS,
    "addmodel": True
},
input=INPUT, output=OUTPUT, label="clearcal_avg")

recipe.add(dgn.generate_leakage_report, "polarization_leakage_precal", {
          "ms": os.path.abspath(os.path.join(MSDIR, BP_CAL_MS)),
          "rep": os.path.join(OUTPUT, "precal_polleakage.ipynb.html"),
          "field": args.bandpass_field[0]
},
input=INPUT, output=OUTPUT, label="precal_polleak_rep")


# First solve for crosshand delays with respect to the refant
# A stronly polarized source is needed for SNR purposes

for icf, cf in enumerate(all_cals):
    recipe.add("cab/casa47_setjy", "set_model_calms_%d" % icf, {
        "msname": BP_CAL_MS,
        "field": cf,
        "standard": all_cals[cf]["standard"],
        "fluxdensity": sdm(all_cals[cf]["fluxdensity"]),
        "spix": sdm(all_cals[cf]["spix"]),
        "reffreq": sdm(all_cals[cf]["reffreq"]),
        "polindex": sdm(all_cals[cf]["polindex"]),
        "polangle": sdm(all_cals[cf]["polangle"]),
    },
    input=INPUT, output=OUTPUT, label="set_model_calms_%d" % icf)

def __correct_feed_convention(ms):
    from pyrap.tables import table as tbl
    import numpy as np
    with tbl("%s::FEED" % ms, readonly=False) as t:
        ang = t.getcol("RECEPTOR_ANGLE")
        ang[:,0] = np.pi/2
        ang[:,1] = np.pi/2
        t.putcol("RECEPTOR_ANGLE", ang)
        log.info("Receptor angle rotated by -90deg")

recipe.add(__correct_feed_convention, "correct_feed_convention", {
          "ms": os.path.abspath(os.path.join(MSDIR, BP_CAL_MS)),
},
input=INPUT, output=OUTPUT, label="correct_feed_convention")

recipe.add("cab/casa47_gaincal", "crosshand_delay", {
        "vis": BP_CAL_MS,
        "caltable": KX,
        "field": ",".join([bp for bp in polarized_calibrators]),
        "refant": REFANT,
        "solint": timegain_solint, #one per scan to track movement of the mean
        "combine": "",
        "parang": True,
        "gaintype": "KCROSS",
},
input=INPUT, output=OUTPUT, label="crosshand_delay")

# Solve for the absolute angle (phase) between the feeds
# (P Jones auto enabled)
# remove the DC of the frequency solutions before 
# possibly joining scans to solve for per-frequency solutions
# a strongly polarized source is needed with known properties
# to limit the amount of PA coverage needed

recipe.add("cab/casa47_polcal", "crosshand_phase_ref", {
        "vis": BP_CAL_MS,
        "caltable": Xref,
        "field": ",".join([bp for bp in polarized_calibrators]),
        "solint": timegain_solint, #one per scan to track movement of the mean
        "combine": "",
        "poltype": "Xf",
        "refant": REFANT,
        "uvrange": uv_cutoff, # EXCLUDE RFI INFESTATION!
        "gaintable": ["%s:output" % ct for ct in [KX]],
},
input=INPUT, output=OUTPUT, label="crosshand_phase_ref")

recipe.add("cab/casa47_polcal", "crosshand_phase_freq", {
        "vis": BP_CAL_MS,
        "caltable": Xf,
        "field": ",".join([bp for bp in polarized_calibrators]),
        "solint": phase_solint, #solint to obtain SNR on solutions
        "combine": "scan",
        "poltype": "Xf",
        "refant": REFANT,
        "uvrange": uv_cutoff, # EXCLUDE RFI INFESTATION!
        "gaintable": ["%s:output" % ct for ct in [KX, Xref]],
},
input=INPUT, output=OUTPUT, label="crosshand_phase_freq")

# Solve for leakages (off-diagonal terms) using the unpolarized source
# - first remove the DC of the frequency response and combine scans
# if necessary to achieve desired SNR

recipe.add("cab/casa47_polcal", "leakage_ref", {
        "vis": BP_CAL_MS,
        "caltable": Dref,
        "field": ",".join([bp for bp in bp_calibrators]),
        "solint": timegain_solint, #1 per scan to keep track of mean
        "combine": "",
        "poltype": "D",
        "uvrange": uv_cutoff, # EXCLUDE RFI INFESTATION!
        "refant": REFANT,
        #"spw": "0:1.0~1.1ghz",
        "gaintable": ["%s:output" % ct for ct in [KX, Xref, Xf]],
},
input=INPUT, output=OUTPUT, label="leakage_ref")

recipe.add("cab/casa47_polcal", "leakage_freq", {
        "vis": BP_CAL_MS,
        "caltable": Df,
        "field": ",".join([bp for bp in bp_calibrators]),
        "solint": phase_solint, # ensure SNR criterion is met
        "combine": "scan",
        "poltype": "Df",
        "refant": REFANT,
        "uvrange": uv_cutoff, # EXCLUDE RFI INFESTATION!
        "gaintable": ["%s:output" % ct for ct in [KX, Xref, Xf, Dref]],
},
input=INPUT, output=OUTPUT, label="leakage_freq")

recipe.add("cab/casa47_applycal", "apply_polcal_sols_avg", {
        "vis": BP_CAL_MS,
        "field": "",
        "parang": True, #P Jones is autoenabled in the polarization calibration, ensure it is enabled now
        "gaintable": ["%s:output" % ct for ct in [KX, Xref, Xf, Dref, Df]]
    },
    input=INPUT, output=OUTPUT, label="apply_polcal_solutions_avg")

# split prior bandpass corrected data into DATA then apply
recipe.add("cab/casa_oldsplit", "split_polcal_data", {
    "vis": COMB_MS,
    "outputvis": POL_CAL_MS,
    "datacolumn": "corrected",
    "field": "",
},
input=INPUT, output=OUTPUT, label="split_polcal_data")

recipe.add(__correct_feed_convention, "correct_feed_convention_polcaldb", {
          "ms": os.path.abspath(os.path.join(MSDIR, POL_CAL_MS)),
},
input=INPUT, output=OUTPUT, label="correct_feed_convention_polcaldb")


recipe.add("cab/casa47_applycal", "apply_polcal_sols_skyframe", {
        "vis": POL_CAL_MS,
        "field": "",
        "parang": True, #P Jones is autoenabled in the polarization calibration, ensure it is enabled now
        "gaintable": ["%s:output" % ct for ct in [KX, Xref, Xf, Dref, Df]]
    },
    input=INPUT, output=OUTPUT, label="apply_polcal_solutions_skyframe")

# Copy data to SKY frame column
recipe.add("cab/msutils", "move_skycorr", {
            "command": "copycol",
            "fromcol": "CORRECTED_DATA",
            "tocol": "SKYFRAME_CORRECTED_DATA",
            "msname": POL_CAL_MS,
}, input=INPUT, output=OUTPUT, label="move_corrected_to_skycorrected_column")

# now make a feed frame relative corrected column if further calibration is ever needed
recipe.add("cab/casa47_applycal", "apply_polcal_sols_feedframe", {
        "vis": POL_CAL_MS,
        "field": "",
        "parang": False, #Feed relative frame for further calibration
        "gaintable": ["%s:output" % ct for ct in [KX, Xref, Xf, Dref, Df]]
    },
    input=INPUT, output=OUTPUT, label="apply_polcal_solutions_feedframe")

recipe.add(dgn.generate_leakage_report, "polarization_leakage_postcal", {
          "ms": os.path.abspath(os.path.join(MSDIR, BP_CAL_MS)),
          "rep": os.path.join(OUTPUT, "postcal_polleakage.ipynb.html"),
          "field": args.bandpass_field[0]
},
input=INPUT, output=OUTPUT, label="postcal_polleak_rep")

recipe.add(dgn.generate_calsolutions_report, "calsolutions_rep", {
          "output": os.path.abspath(OUTPUT),
          "rep": os.path.join(OUTPUT, "calsolutions_report.html"),
},
input=INPUT, output=OUTPUT, label="calsolutions_rep")

def compile_and_run(steps):
    recipe.run([s for s in steps])

def define_steps():
    opts = (
        [
            "split_avg_data",
            "clearcal_avg",
            "precal_polleak_rep",
            ###################################
            ### Polarization calibration
            ###################################
        ] +
        [ "set_model_calms_%d" % cfi for cfi, cf in enumerate(all_cals) ] +
        [
            "correct_feed_convention",
            "crosshand_delay",
            "crosshand_phase_ref",
            "crosshand_phase_freq",
            "leakage_ref",
            "leakage_freq",
            "apply_polcal_solutions_avg",
            "split_polcal_data",
            "correct_feed_convention_polcaldb",
            "apply_polcal_solutions_skyframe",
            "move_corrected_to_skycorrected_column",
            "apply_polcal_solutions_feedframe",
            "postcal_polleak_rep",
            "calsolutions_rep"
        ])

    checked_opts = OrderedDict()
    disabled_opts = []
    for o in opts:
        checked_opts[o] = o not in disabled_opts
    return checked_opts

