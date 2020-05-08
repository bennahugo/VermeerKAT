from __future__ import print_function, absolute_import

import stimela
import stimela.dismissable as sdm
from pyrap.tables import table as tbl
import os
import sys
import argparse
import numpy as np
from collections import OrderedDict
import shutil
import vermeerkat

parser = argparse.ArgumentParser("MeerKAT BasicApplyTransfer (BAT) pipeline")
parser.add_argument('--input_dir', dest='input_directory', metavar='<input directory>',
                    action='store', default='input',
                    help='directory to read input data with e.g. antenna table and beam files')
parser.add_argument('--output_dir', dest='output_directory', metavar='<output directory>',
                    action='store', default='output', help='directory to write output data')
parser.add_argument('--msdir_dir', dest='msdir_directory', metavar='<msdir directory>',
                    action='store', default='msdir', help='directory to store measurement sets')
parser.add_argument('--bp', dest='bandpass_field', metavar='<bandpass field>',
                    help='Bandpass fields')
parser.add_argument('--gc', dest='gain_field', metavar='<gain calibrator fields>', default=[],
                    action="append", nargs="+",
                    help='Gain fields. This switch can be used multiple times for more than 1 field')
parser.add_argument('--altcal', dest='alt_cal_field', metavar='<alternative calibrator fields>', default=[],
                    action="append", nargs="+",
                    help='Alternative calibrator. Phase corrections will be applied to this field for further '
                         'diagnostic or calibration procedures. This switch can be used multiple times for '
                         'more than 1 field. This field has no impact on target field calibration.')
parser.add_argument('--tar', dest='target_field', metavar='<target fields>', type=str, default=[],
                    action="append", nargs="+",
                    help='Target fields. This switch can be used multiple times for more than 1 field')
parser.add_argument('--no_delay_with_gcal', dest='delay_with_gcal', action='store_false', default=True,
                    help='DON''t use gain calibrators for delay calibration')
parser.add_argument('--flag_antenna', dest='flag_antenna', action='append', default=[],
                    help="Flag antenna. Can be specified more than once to flag more than one antenna.")
parser.add_argument('--skip_prepdata', dest='skip_prepdata', action='store_true',
                    help="Skip prepdata")
parser.add_argument('--skip_prelim_flagging', dest='skip_prelim_flagging', action='store_true',
                    help="Skip preliminary flagging")
parser.add_argument('--skip_prelim_1GC', dest='skip_prelim_1GC', action='store_true',
                    help="Skip preliminary 1GC")
parser.add_argument('--skip_final_flagging', dest='skip_final_flagging', action='store_true',
                    help="Skip final flagging")
parser.add_argument('--skip_final_1GC', dest='skip_final_1GC', action='store_true',
                    help="Skip final 1GC")
parser.add_argument('--skip_flag_targets', dest='skip_flag_targets', action='store_true',
                    help="Skip flag targets")
parser.add_argument('--skip_transfer_to_targets', dest='skip_transfer_to_targets', action='store_true',
                    help="Skip transfer of solutions to target")
parser.add_argument('--skip_final_split', dest='skip_final_split', action='store_true',
                    help="Skip final split")
parser.add_argument('msprefix', metavar='<measurement set name prefix>',
                    help='Prefix of measurement set name as it appears relative to msdir. This must NOT be a '
                         'path prefix')
parser.add_argument('--time_sol_interval', dest='time_sol_interval', default="inf",
                    help="Time (gain) solutions interval (default one per scan)")
parser.add_argument('--freq_sol_interval', dest='freq_sol_interval', default="inf",
                    help="Frequency time-invariant solutions interval (default one per observation)")
parser.add_argument('--clip_delays', dest='clip_delays', default=1, type=float,
                    help="Clip delays above this absolute in nanoseconds")
parser.add_argument('--cal_model', dest='cal_model', default='pks1934-638.lsm',
                    help="Calibrator apparent sky model (tigger lsm format)")
parser.add_argument('--ref_ant', dest='ref_ant', default='m037',
                    help="Reference antenna to use throughout")
parser.add_argument("--containerization", dest="containerization", default="docker",
                    help="Containerization technology to use. See your stimela installation for options")
parser.add_argument("--image_gaincalibrators", dest="image_gaincalibrators", action="store_true",
                    help="Image gain calibrators")
parser.add_argument("--dont_prompt", dest="dont_prompt",
                    action="store_true",
                    help="Don't prompt the user for confirmation of parameters")

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

with tbl(os.path.join(MSDIR, ZEROGEN_DATA)+"::FIELD", ack=False) as t:
    field_names = t.getcol("NAME")
    FDB = {fn: str(fni) for fni, fn in enumerate(field_names)}

def flistr():
    vermeerkat.log.info("The following fields are available:")
    for f in FDB:
        vermeerkat.log.info("\t '{0:s}' index {1:s}".format(f, FDB[f]))
    sys.exit(0)

def __merge_input():
    mod_path = os.path.dirname(vermeerkat.__file__)
    data_dir = os.path.join(mod_path, "data", "input")
    shutil.copytree(data_dir, INPUT)

if not os.path.exists(INPUT):
    __merge_input()
elif os.path.isdir(INPUT):
    shutil.rmtree(INPUT)
    __merge_input()
else:
    raise RuntimeError("A file called {} already exists, but is not a input directory".format(INPUT))

vermeerkat.log.info("Time invariant solution time interval: {0:s}".format(args.time_sol_interval))
vermeerkat.log.info("Frequency invariant solution frequency interval: {0:s}".format(args.freq_sol_interval))
vermeerkat.log.info("Will clip absolute delays over {0:.2f}ns".format(args.clip_delays))
vermeerkat.log.info("Will use '{}' as flux calibrator full sky model".format(args.cal_model))

FLAGANT = [f[0] if isinstance(f, list) else f for f in args.flag_antenna]
if len(FLAGANT) != 0:
    vermeerkat.log.info("Will flag antenna {}".format(", ".join(["'{}'".format(a) for a in FLAGANT])))

BPCALIBRATOR = args.bandpass_field
if BPCALIBRATOR is None: raise ValueError("No bandpass calibrator specified")
GCALIBRATOR = [f[0] if isinstance(f, list) else f for f in args.gain_field]
ALTCAL = [f[0] if isinstance(f, list) else f for f in args.alt_cal_field]
TARGET = [f[0] if isinstance(f, list) else f for f in args.target_field]
if len(TARGET) < 1: raise ValueError("No target specified")

DO_USE_GAINCALIBRATOR = len(GCALIBRATOR) > 0
if not DO_USE_GAINCALIBRATOR:
    vermeerkat.log.info("*NO* gain calibrator specified")

DO_USE_GAINCALIBRATOR_DELAY = args.delay_with_gcal and DO_USE_GAINCALIBRATOR
if DO_USE_GAINCALIBRATOR_DELAY:
    vermeerkat.log.info("Will transfer rate calibraton using gain calibrator")
else:
    vermeerkat.log.info("Will *NOT* transfer rate calibraton from gain calibrator")

REFANT = args.ref_ant

vermeerkat.log.info("Reference antenna {0:s} to be used throughout".format(REFANT))

## DO NOT EDIT ANY OF THESE PREDEFINED WORKERS UNLESS YOU KNOW WHAT YOU'RE DOING

K0 = PREFIX + ".K0"
B0 = PREFIX + ".B0"
G0 = PREFIX + ".G0"
GA = PREFIX + ".GAlt"
G1 = PREFIX + ".G1"
F0 = PREFIX + ".F0"
B1 = PREFIX + ".B1"

MANUAL_FLAG_LIST = []
FIRSTGEN_DATA = ["{}.{}.1gc.ms".format(t, PREFIX) for t in TARGET]

vermeerkat.log.info("The following fields are available:")
for f in FDB:
    vermeerkat.log.info("\t '{0:s}' index {1:s}{2:s}".format(
        f, FDB[f],
        " selected as 'BP'" if f == BPCALIBRATOR else
        " selected as 'GC'" if f in GCALIBRATOR else
        " selected as 'ALTCAL'" if f in ALTCAL else
        " selected as 'TARGET'" if f in TARGET else
        " not selected"))

try:
    input = raw_input
except NameError:
    pass

while not args.dont_prompt:
    r = input("Is this configuration correct? (Y/N) >> ").lower()
    if r == "y":
        break
    elif r == "n":
        sys.exit(0)
    else:
        continue

stimela.register_globals()
recipe = stimela.Recipe('MEERKAT: basic transfer calibration',
                        ms_dir=MSDIR,
                        singularity_image_dir=os.environ.get("SINGULARITY_PULLFOLDER", ""),
                        JOB_TYPE=args.containerization)

def addmanualflags(recipe, reason, antenna="", spw="", scan="", uvrange="", field=""):
    """ Read CASA flagdata docs before using """
    recipe.add("cab/casa_flagdata", "handflags", {
        "vis": ZEROGEN_DATA,
        "mode": "manual",
        "antenna": antenna,
        "spw": spw,
        "scan": scan,
        "uvrange": uvrange,
        "field": field
    },
    input=INPUT, output=OUTPUT, label=reason)
    MANUAL_FLAG_LIST.append(reason)
    return [reason]

def prepare_data():
        recipe.add("cab/casa_flagmanager", "backup_CAM_SP_flags", {
                   "vis": ZEROGEN_DATA,
                   "mode": "save",
                   "versionname": "SP_ORIGINAL",
            },
            input=INPUT, output=OUTPUT, label="backup_CAM_SP_flags")

        recipe.add("cab/casa_listobs", "listobs", {
                   "vis": ZEROGEN_DATA,
            },
            input=INPUT, output=OUTPUT, label="listobs")


        recipe.add("cab/casa_flagdata", "flag_reset", {
                   "vis": ZEROGEN_DATA,
                   "mode": "unflag",
            },
            input=INPUT, output=OUTPUT, label="reset_flags")
        recipe.add("cab/politsiyakat_autocorr_amp", "flag_autopower", {
                   "msname": ZEROGEN_DATA,
                   "field": ",".join(str(FDB[f]) for f in FDB),
                   "cal_field": ",".join([FDB[BPCALIBRATOR]] + [FDB[f] for f in GCALIBRATOR + ALTCAL]),
                   "nrows_chunk": 15000,
                   "scan_to_scan_threshold": 1.5,
                   "antenna_to_group_threshold": 4,
                   "nio_threads": 1,
                   "nproc_threads": 32,
                  },input=INPUT, output=OUTPUT, label="flag_autopower")
        recipe.add("cab/casa_flagdata", "flag_autocorrelations", {
                   "vis": ZEROGEN_DATA,
                   "mode": "manual",
                   "autocorr": True,
            },
            input=INPUT, output=OUTPUT, label="flagging_auto_correlations")
        recipe.add("cab/casa_flagdata", "flag_rolloff", {
                   "vis": ZEROGEN_DATA,
                   "mode": "manual",
                   "spw": "*:850~980MHz,*:1658~1800MHz,*:1419.8~1421.3MHz", #band-rolloffs and Milkyway HI line
            },
            input=INPUT, output=OUTPUT, label="flagging_rolloff")

        recipe.add("cab/casa_clearcal", "clear_calibration", {
                    "vis": ZEROGEN_DATA,
                    "addmodel": True,
            },
            input=INPUT, output=OUTPUT, label="clear_calibration")

        return [
           "backup_CAM_SP_flags",
           "listobs",
           ####"reset_flags",
           "flag_autopower",
           "flagging_auto_correlations",
           "flagging_rolloff",
           "clear_calibration",
        ]

def rfiflag_data(do_flag_targets=False, steplabel="flagpass1", exec_strategy="mk_rfi_flagging_calibrator_fields_firstpass.yaml", on_corr_residuals=False, dc="DATA"):
        recipe.add('cab/tricolour', steplabel,
        {
                  "ms"                  : ZEROGEN_DATA,
                  "data-column"         : dc,
                  "window-backend"      : 'numpy',
                  "field-names"         : [FDB[BPCALIBRATOR]],
                  "flagging-strategy"   : "total_power" if not do_flag_targets else "polarisation",
                  "config"              : exec_strategy,
                  "subtract-model-column": sdm.dismissable("MODEL_DATA" if on_corr_residuals else None),
                  "dilate-masks": sdm.dismissable(None),
                  "ignore-flags": sdm.dismissable(None),
                  "scan-numbers": sdm.dismissable(None),
        },
        input=INPUT, output=OUTPUT, label=steplabel)
        recipe.add('cab/tricolour', steplabel + "_gc",
        {
                  "ms"                  : ZEROGEN_DATA,
                  "data-column"         : dc,
                  "window-backend"      : 'numpy',
                  "field-names"         : [FDB[t] for t in TARGET] if do_flag_targets else [FDB[t] for t in GCALIBRATOR + ALTCAL],
                  "flagging-strategy"   : "total_power" if not do_flag_targets else "polarisation",
                  "subtract-model-column": sdm.dismissable("MODEL_DATA" if on_corr_residuals else None),
                  "config"              : exec_strategy,
                  "dilate-masks": sdm.dismissable(None),
                  "ignore-flags": sdm.dismissable(None),
                  "scan-numbers": sdm.dismissable(None),
        },
        input=INPUT, output=OUTPUT, label=steplabel + ".gc" if not do_flag_targets else steplabel + ".targets")


        recipe.add("cab/casa_flagdata", "flag_summary_{}".format(steplabel), {
                   "vis": ZEROGEN_DATA,
                   "mode": "summary"
            },
            input=INPUT, output=OUTPUT, label="flagging_summary_{}".format(steplabel))

        return (([steplabel, steplabel + ".gc"] if len(ALTCAL) > 0 or DO_USE_GAINCALIBRATOR else [steplabel])
                if not do_flag_targets else [steplabel + ".targets"]) + \
        [
          "flagging_summary_{}".format(steplabel)
        ]

def image_calibrator(recipe, label="prelim"):
    imfields = [FDB[a] for a in ALTCAL] + \
               ([FDB[t] for t in GCALIBRATOR] if (DO_USE_GAINCALIBRATOR and
                                                  DO_USE_GAINCALIBRATOR_DELAY) else [])
    steps = []
    for f in imfields:
        imopts={
            "msname": ZEROGEN_DATA,
            "join-channels": True,
            "channels-out": 9,
            "size": 4096,
            "scale": "1.6asec",
            "mgain": 0.8,
            "gain": 0.1,
            "niter": 3000,
            "name": "calfield-{}-{}".format(f, label),
            "field": f,
            "fits-mask": sdm.dismissable(None),
            ###"save-source-list": True,
            "fit-spectral-pol": 3,
        }
        maskname = "MASK-{}-{}.fits".format(f, label)

        recipe.add("cab/wsclean", "image_{}_field{}".format(label, f),
                   imopts,
                   input=INPUT, output=OUTPUT, label="image_calfield_{}_{}".format(f, label))

        recipe.add("cab/cleanmask", "mask_{}_{}".format(label, f), {
           'image': "calfield-{}-{}-MFS-image.fits:output".format(f, label),
           'output': maskname,
           'sigma': 35,
           'boxes': 9,
           'iters': 20,
           'overlap': 0.3,
           'no-negative': True,
           'tolerance': 0.75,
        }, input=INPUT, output=OUTPUT, label='mask_{}_{}'.format(label, f))
        imopts2 = {k: v for k, v in list(imopts.items())}

        imopts2["fits-mask"] = maskname + ":output"
        imopts2["local-rms"] = True
        imopts2["auto-threshold"] = 5

        recipe.add("cab/wsclean", "image_{}_field{}_rnd2".format(label, f),
                   imopts2,
                   input=INPUT, output=OUTPUT, label="image_calfield_{}_{}_rnd2".format(f, label))

        steps += ["image_calfield_{}_{}".format(f, label),
                  'mask_{}_{}'.format(label, f),
                  "image_calfield_{}_{}_rnd2".format(f, label)]
    return steps

def do_1GC(recipe, label="prelim", do_apply_target=False, do_predict=True, applyonly=False):
    recipe.add("cab/casa_flagmanager", "backup_flags_prior_1gc_%s" % label, {
                   "vis": ZEROGEN_DATA,
                   "mode": "save",
                   "versionname": "prior_%s_1GC" % label,
            },
            input=INPUT, output=OUTPUT, label="backup_flags_prior_1gc_%s" % label)

    recipe.add("cab/simulator", "predict_fluxcalibrator_%s" % label, {
           "skymodel": args.cal_model, # we are using 1934-638 as flux scale reference
           "msname": ZEROGEN_DATA,
           "threads": 24,
           "mode": "simulate",
           "column": "MODEL_DATA",
           "Ejones": False, # first bandpass calibration is normal calibration then we absorb coefficients into another table
           "beam-files-pattern": "meerkat_pb_jones_cube_95channels_$(xy)_$(reim).fits",
           "beam-l-axis": "X",
           "beam-m-axis": "-Y", #[OMS] flipped in code: southern hemisphere
           "parallactic-angle-rotation": True,
           "field-id": int(FDB[BPCALIBRATOR]),
    },
    input=INPUT, output=OUTPUT, label="set_flux_reference_%s" % label)

    recipe.add("cab/casa47_gaincal", "delaycal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": K0,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "refant": REFANT,
            "solint": args.time_sol_interval,
            "combine": "",
            "minsnr": 3,
            "minblperant": 4,
            "gaintype": "K",
        },
        input=INPUT, output=OUTPUT, label="delay_calibration_bp_%s" % label)

    def clip_delays(vis, clipminmax):
        with tbl(os.path.join(OUTPUT, vis), readonly=False) as t:
            fl = t.getcol("FLAG")
            d = t.getcol("FPARAM")
            prevflagged = np.sum(fl) * 100.0 / fl.size
            fl = np.logical_or(fl,
                               np.logical_or(d.real > np.max(clipminmax),
                                             d.real < np.min(clipminmax)))
            t.putcol("FLAG", fl)
            currentflagged = np.sum(fl) * 100.0 / fl.size
            vermeerkat.log.info("Flagged {0:.2f}%, up from previous {1:.2f}%".format(currentflagged,
                                                                       prevflagged))
    recipe.add(clip_delays, "clipdel_%s" % label, {
            "vis": K0,
            "clipminmax": [-args.clip_delays, +args.clip_delays],
        },
        input=INPUT, output=OUTPUT, label="clip_delay_%s" % label)


    ##capture time drift of bandpass
    recipe.add("cab/casa47_gaincal", "bandpassgain_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": G0,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "solint": args.time_sol_interval,
            "combine": "",
            "gaintype": "G",
            "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
            ##"spw": "0:1.3~1.5GHz",
            "gaintable": ["%s:output" % ct for ct in [K0]],
            "gainfield": [FDB[BPCALIBRATOR]],
            "interp":["nearest"],
            "refant": REFANT,
        },
        input=INPUT, output=OUTPUT, label="remove_bp_average_%s" % label)

    # average as much as possible to get as good SNR on bandpass as possible
    recipe.add("cab/casa47_bandpass", "bandpasscal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": "%s:output" % B0,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "solint": args.freq_sol_interval,
            "combine": "scan",
            "minsnr": 3.0,
            "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
            #"fillgaps": 100000000, # LERP!
            "gaintable": ["%s:output" % ct for ct in [K0, G0]],
            "gainfield": [FDB[BPCALIBRATOR], FDB[BPCALIBRATOR]],
            "interp": ["nearest", "nearest"],
            "refant": REFANT,
        },
        input=INPUT, output=OUTPUT, label="bp_freq_calibration_%s" % label)

    recipe.add("cab/casa47_applycal", "apply_sols_bp_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]] +
                              [FDB[a] for a in ALTCAL] + 
                              ([FDB[t] for t in GCALIBRATOR] if (DO_USE_GAINCALIBRATOR and DO_USE_GAINCALIBRATOR_DELAY) else
                               [])),
            "gaintable": ["%s:output" % ct for ct in [K0,G0,B0]],
            "gainfield": [",".join([FDB[BPCALIBRATOR]]),
                          ",".join([FDB[BPCALIBRATOR]]),
                          ",".join([FDB[BPCALIBRATOR]])],
            "interp": ["nearest","nearest","linear,linear"]
        },
        input=INPUT, output=OUTPUT, label="apply_sols_bp_%s" % label)

    #create basic model for secondaries
    cal_im_steps = image_calibrator(recipe=recipe, label=label) if args.image_gaincalibrators else []

    ##capture time drift of gain and alternative calibrators
    recipe.add("cab/casa47_gaincal", "delaycal_gc_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": K0,
            "field": ",".join([FDB[a] for a in ALTCAL] +
                              [FDB[t] for t in GCALIBRATOR]) if (DO_USE_GAINCALIBRATOR and
                                                                 DO_USE_GAINCALIBRATOR_DELAY) else
                     ",".join([FDB[a] for a in ALTCAL]),
            "refant": REFANT,
            "solint": args.time_sol_interval,
            "combine": "",
            "minsnr": 3,
            "minblperant": 4,
            "gaintype": "K",
            "gaintable": ["%s:output" % ct for ct in [B0]],
            "gainfield": [",".join([FDB[BPCALIBRATOR]])],
            "interp": ["linear,linear"],
            ##"spw": "0:1.3~1.5GHz",
            "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
            "append": True,
            "refant": REFANT,
        },
        input=INPUT, output=OUTPUT, label="delay_calibration_gc_%s" % label)

    recipe.add(clip_delays, "clipdel_%s" % label, {
            "vis": K0,
            "clipminmax": [-args.clip_delays, +args.clip_delays],
        },
        input=INPUT, output=OUTPUT, label="clip_delay_gc_%s" % label)

    recipe.add("cab/msutils", "delaycal_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % K0,
            "tabtype": "delay",
            "plot_file": "{0:s}.{1:s}.K0.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180
        },
        input=INPUT, output=OUTPUT, label="plot_delays_%s" % label)

    if DO_USE_GAINCALIBRATOR:
        recipe.add("cab/casa47_gaincal", "apgain_%s" % label, {
                "vis": ZEROGEN_DATA,
                "caltable": G0,
                "field": ",".join([FDB[t] for t in GCALIBRATOR]),
                "solint": args.time_sol_interval,
                "combine": "",
                "gaintype": "G",
                "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
                ##"spw": "0:1.3~1.5GHz",
                "gaintable": ["%s:output" % ct for ct in [B0, K0]],
                "gainfield": [FDB[BPCALIBRATOR],
                              ",".join([FDB[a] for a in ALTCAL] +
                                       [FDB[t] for t in GCALIBRATOR]) if (DO_USE_GAINCALIBRATOR and
                                                                          DO_USE_GAINCALIBRATOR_DELAY) else
                              ",".join([FDB[a] for a in ALTCAL]),
                             ],
                "interp":["linear,linear", "nearest"],
                "append": True,
                "refant": REFANT,
            },
            input=INPUT, output=OUTPUT, label="apgain_%s" % label)

        recipe.add("cab/casa_fluxscale", "fluxscale_%s" % label, {
                "vis": ZEROGEN_DATA,
                "caltable": "%s:output" % G0,
                "fluxtable": "%s:output" % F0,
                "reference": ",".join([FDB[BPCALIBRATOR]]),
                "transfer": ",".join([FDB[t] for t in GCALIBRATOR]),
            },
            input=INPUT, output=OUTPUT, label="fluxscale_%s" % label)

    recipe.add("cab/msutils", "gain_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % (F0 if DO_USE_GAINCALIBRATOR else G0),
            "tabtype": "gain",
            "plot_file": "{0:s}.{1:s}.F0.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180

        },
        input=INPUT, output=OUTPUT, label="plot_gain_%s" % label)

    recipe.add("cab/msutils", "bpgain_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % B0,
            "tabtype": "bandpass",
            "plot_file": "{0:s}.{1:s}.B0.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180

        },
        input=INPUT, output=OUTPUT, label="plot_bpgain_%s" % label)

    if len(ALTCAL) > 0:
        # no model of alternatives, don't adjust amp
        recipe.add("cab/casa47_gaincal", "altcalgain_%s" % label, {
                "vis": ZEROGEN_DATA,
                "caltable": GA,
                "field": ",".join([FDB[a] for a in ALTCAL]),
                "solint": args.time_sol_interval,
                "combine": "",
                "gaintype": "G",
                "calmode": "p",
                "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
                ##"spw": "0:1.3~1.5GHz",
                "gaintable": ["%s:output" % ct for ct in [K0, G0, B0]],
                "gainfield": [
                    ",".join([FDB[a] for a in ALTCAL]),
                    FDB[BPCALIBRATOR],
                    FDB[BPCALIBRATOR]],
                "interp":["linear,linear","nearest"],
                "refant": REFANT,
            },
            input=INPUT, output=OUTPUT, label="remove_altcal_average_%s" % label)

    recipe.add("cab/msutils", "altgain_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % GA,
            "tabtype": "gain",
            "plot_file": "{0:s}.{1:s}.GA.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180
        },
        input=INPUT, output=OUTPUT, label="plot_altgains_%s" % label)


    for a in ALTCAL:
        recipe.add("cab/casa47_applycal", "apply_sols_ac_%s_%s" % (FDB[a], label), {
                "vis": ZEROGEN_DATA,
                "field": FDB[a],
                "gaintable": ["%s:output" % ct for ct in [K0,G0,B0,GA]],
                "gainfield": [
                              ",".join([FDB[a]]),
                              ",".join([FDB[BPCALIBRATOR]]),
                              ",".join([FDB[BPCALIBRATOR]]),
                              ",".join([FDB[a]]),
                             ],
                "interp": ["linear,linear","nearest","nearest"]
            },
            input=INPUT, output=OUTPUT, label="apply_sols_ac_%s_%s" % (FDB[a], label))

    if do_apply_target or DO_USE_GAINCALIBRATOR:
        recipe.add("cab/casa47_applycal", "apply_sols_%s" % label, {
                "vis": ZEROGEN_DATA,
                "field": ",".join([FDB[t] for t in GCALIBRATOR] +
                                  ([FDB[t] for t in TARGET] if do_apply_target else [])),
                "gaintable": ["%s:output" % ct for ct in [B0,K0,F0]] if DO_USE_GAINCALIBRATOR else ["%s:output" % ct for ct in [B0,K0,G0]],
                "gainfield": [FDB[BPCALIBRATOR],
                              ",".join([FDB[t] for t in GCALIBRATOR])
                              if (DO_USE_GAINCALIBRATOR and DO_USE_GAINCALIBRATOR_DELAY) else FDB[BPCALIBRATOR],
                              ",".join([FDB[t] for t in GCALIBRATOR])
                              if DO_USE_GAINCALIBRATOR else FDB[BPCALIBRATOR],
                             ],
                "interp": ["linear,linear","nearest","nearest"]
            },
            input=INPUT, output=OUTPUT, label="apply_1GC_solutions_%s" % label)

    recipe.add("cab/casa_plotms", "plot_pa_bp_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "correlation": "XX,YY",
            "xaxis": "amp",
            "xdatacolumn": "corrected/model_vector",
            "yaxis": "phase",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "overwrite": True,
            "showgui": False,
            "avgtime": "32",
            "avgchannel": "32",
            "plotfile": "{}.{}.bp.ampphase.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="phaseamp_plot_for_bandpass_%s" % label)

    recipe.add("cab/casa_plotms", "plot_pa_gcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t] for t in GCALIBRATOR]),
            "correlation": "XX,YY",
            "xaxis": "amp",
            "xdatacolumn": "corrected/model_vector",
            "yaxis": "phase",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "iteraxis": "field",
            "overwrite": True,
            "showgui": False,
            "avgtime": "32",
            "avgchannel": "32",
            "plotfile": "{}.{}.gc.ampphase.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="phaseamp_plot_for_gain_%s" % label)

    recipe.add("cab/casa_plotms", "plot_ri_gcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t] for t in GCALIBRATOR + ALTCAL]),
            "correlation": "XX,YY",
            "xaxis": "real",
            "xdatacolumn": "corrected/model_vector",
            "yaxis": "imag",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "iteraxis": "field",
            "overwrite": True,
            "showgui": False,
            "avgtime": "32",
            "avgchannel": "32",
            "plotfile": "{}.{}.gc.realimag.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="reim_plot_for_gain_%s" % label)

    recipe.add("cab/casa_plotms", "plot_ri_bpcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": FDB[BPCALIBRATOR],
            "correlation": "XX,YY",
            "xaxis": "real",
            "xdatacolumn": "corrected/model_vector",
            "yaxis": "imag",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "iteraxis": "field",
            "overwrite": True,
            "showgui": False,
            "avgtime": "32",
            "avgchannel": "32",
            "plotfile": "{}.{}.bp.realimag.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="reim_plot_for_bp_%s" % label)


    recipe.add("cab/casa_plotms", "plot_afreq_gcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t] for t in GCALIBRATOR + ALTCAL]),
            "correlation": "XX,YY",
            "xaxis": "freq",
            "yaxis": "amp",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "iteraxis": "field",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "plotfile": "{}.{}.gc.ampfreq.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="afreq_for_gain_%s" % label)


    recipe.add("cab/casa_plotms", "plot_pfreq_gcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t] for t in GCALIBRATOR + ALTCAL]),
            "correlation": "XX,YY",
            "xaxis": "freq",
            "yaxis": "phase",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "iteraxis": "field",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "plotfile": "{}.{}.gc.phasefreq.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="pfreq_for_gain_%s" % label)

    recipe.add("cab/casa_plotms", "plot_ascan_gcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t] for t in GCALIBRATOR + ALTCAL]),
            "correlation": "XX,YY",
            "xaxis": "scan",
            "yaxis": "amp",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "iteraxis": "field",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "avgchannel": "32",
            "plotfile": "{}.{}.gc.ampscan.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="ascan_for_gain_%s" % label)

    recipe.add("cab/casa_plotms", "plot_pscan_gcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t] for t in GCALIBRATOR + ALTCAL]),
            "correlation": "XX,YY",
            "xaxis": "scan",
            "yaxis": "phase",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "iteraxis": "field",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "avgchannel": "32",
            "plotfile": "{}.{}.gc.phasescan.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="pscan_for_gain_%s" % label)


    recipe.add("cab/casa_plotms", "plot_afreq_bpcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "correlation": "XX,YY",
            "xaxis": "freq",
            "yaxis": "amp",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "plotfile": "{}.{}.bp.ampfreq.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="afreq_for_bp_%s" % label)

    recipe.add("cab/casa_plotms", "plot_pfreq_bpcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "correlation": "XX,YY",
            "xaxis": "freq",
            "yaxis": "phase",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "plotfile": "{}.{}.bp.phasefreq.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="pfreq_for_bp_%s" % label)

    recipe.add("cab/casa_plotms", "plot_ascan_bpcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "correlation": "XX,YY",
            "xaxis": "scan",
            "yaxis": "amp",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "avgchannel": "32",
            "plotfile": "{}.{}.bp.ampscan.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="ascan_for_bp_%s" % label)

    recipe.add("cab/casa_plotms", "plot_pscan_bpcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "correlation": "XX,YY",
            "xaxis": "scan",
            "yaxis": "phase",
            "xdatacolumn": "corrected/model_vector",
            "ydatacolumn": "corrected/model_vector",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "avgchannel": "32",
            "plotfile": "{}.{}.bp.phasescan.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="pscan_for_bp_%s" % label)


    return ([
                "backup_flags_prior_1gc_{}".format(label)
            ] +
            [
                "set_flux_reference_{}".format(label)
            ] if do_predict else []) + ([
                    "delay_calibration_bp_{}".format(label),
                    "clip_delay_{}".format(label),
                    "remove_bp_average_{}".format(label),
                    "bp_freq_calibration_{}".format(label),
                    "apply_sols_bp_{}".format(label),
                ] + cal_im_steps +
                    ([
                        "delay_calibration_gc_{}".format(label),
                        "clip_delay_gc_{}".format(label),
                     ] if len(ALTCAL) > 0 or DO_USE_GAINCALIBRATOR_DELAY else []) + ([
                        "apgain_{}".format(label),
                        "fluxscale_{}".format(label),
                     ] if DO_USE_GAINCALIBRATOR else [])  + ([
                        "remove_altcal_average_{}".format(label),
                        "plot_altgains_{}".format(label),
                     ] if len(ALTCAL) > 0 else [])
                if not applyonly else [
                    "apply_sols_bp_{}".format(label)
                ]) +\
            [
                "plot_delays_{}".format(label),
                "plot_gain_{}".format(label),
                "plot_bpgain_{}".format(label),
                "apply_sols_bp_{}".format(label),
            ] + ([
                    "apply_1GC_solutions_{}".format(label)
                 ] if do_apply_target or DO_USE_GAINCALIBRATOR else []) +\
            [
                "apply_sols_ac_{0:s}_{1:s}".format(FDB[a], label) for a in ALTCAL
            ] +\
            [
                "phaseamp_plot_for_bandpass_{}".format(label),
                "reim_plot_for_bp_{}".format(label),
                "afreq_for_bp_{}".format(label),
                "pfreq_for_bp_{}".format(label),
                "ascan_for_bp_{}".format(label),
                "pscan_for_bp_{}".format(label)
            ] + ([
                    "afreq_for_gain_{}".format(label),
                    "pfreq_for_gain_{}".format(label),
                    "ascan_for_gain_{}".format(label),
                    "pscan_for_gain_{}".format(label),
                    "phaseamp_plot_for_gain_{}".format(label),
                    "reim_plot_for_gain_{}".format(label),
                 ] if len(ALTCAL) > 0 or DO_USE_GAINCALIBRATOR else [])

def finalize_and_split():
    for ti, t in enumerate(TARGET):
        recipe.add("cab/casa_split", "split_%d" % ti, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t]]),
            "outputvis": FIRSTGEN_DATA[ti]
        },
        input=INPUT, output=OUTPUT, label="split_%d" % ti)
        recipe.add("cab/casa_flagmanager", "backup_1GC_flags_%d" % ti, {
                   "vis": FIRSTGEN_DATA[ti],
                   "mode": "save",
                   "versionname": "1GC_LEGACY",
            },
            input=INPUT, output=OUTPUT, label="backup_1GC_flags_%d" % ti)

    return ["split_%d" % ti for ti, t in enumerate(TARGET)] + \
           ["backup_1GC_flags_%d" % ti for ti, t in enumerate(TARGET)]

def define_steps():
    STEPS = []
    if not args.skip_prepdata:
        STEPS += prepare_data()
    for a in FLAGANT:
        STEPS += addmanualflags(recipe, "Pointing issue {}".format(a), antenna=a, spw="", scan="", uvrange="", field="")
    if not args.skip_prelim_flagging:
        STEPS += rfiflag_data(do_flag_targets=False, steplabel="flagpass1", exec_strategy="mk_rfi_flagging_calibrator_fields_firstpass.yaml", on_corr_residuals=False, dc="DATA")
    if not args.skip_prelim_1GC:
        STEPS += do_1GC(recipe, label="prelim", do_predict=True)
    if not args.skip_final_flagging:
        STEPS += rfiflag_data(do_flag_targets=False, steplabel="flagpass2", exec_strategy="mk_rfi_flagging_calibrator_fields_secondpass.yaml", on_corr_residuals=True, dc="CORRECTED_DATA")
    if not args.skip_final_1GC:
        STEPS += do_1GC(recipe, label="second_round", do_predict=False, do_apply_target=False)
    if not args.skip_transfer_to_targets:
        STEPS += do_1GC(recipe, label="apply_only", do_predict=False, do_apply_target=True, applyonly=True)
    if not args.skip_flag_targets:
        STEPS += rfiflag_data(do_flag_targets=True, steplabel="flagfinal", exec_strategy="mk_rfi_flagging_target_fields_firstpass.yaml", on_corr_residuals=False, dc="CORRECTED_DATA")
    if not args.skip_final_split:
        STEPS += finalize_and_split()

    checked_opts = OrderedDict()
    for o in STEPS: checked_opts[o] = True
    return checked_opts

def compile_and_run(STEPS):
    if len(STEPS) != 0:
        recipe.run(STEPS)

def main():
    steps = define_steps()
    compile_and_run(list(steps.keys()))

if __name__ == "__main__":
    main()