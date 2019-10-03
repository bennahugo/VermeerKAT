import stimela
import stimela.dismissable as sdm
from pyrap.tables import table as tbl
import os
import sys
import argparse

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
parser.add_argument('--time_sol_interval', dest='time_sol_interval', default="64s",
                    help="Time (gain) solutions interval")
parser.add_argument('--freq_sol_interval', dest='freq_sol_interval', default="inf",
                    help="Frequency time-invariant solutions interval")
parser.add_argument('--clip_delays', dest='clip_delays', default=30,
                    help="Clip delays above this absolute in nanoseconds")
parser.add_argument('--cal_model', dest='cal_model', default='pks1934-638.lsm',
                    help="Calibrator apparent sky model (tigger lsm format)")

args = parser.parse_args()

INPUT = args.input_directory
MSDIR = args.msdir_directory
OUTPUT = args.output_directory
print "Directory '{0:s}' is used as input directory".format(INPUT)
print "Directory '{0:s}' is used as output directory".format(OUTPUT)
print "Directory '{0:s}' is used as msdir directory".format(MSDIR)

print "Frequency invariant solution time interval: {0:s}".format(args.time_sol_interval)
print "Time invariant solution frequency interval: {0:s}".format(args.freq_sol_interval)
print "Will clip absolute delays over {0:d}ns".format(args.clip_delays)
print "Will use '{}' as flux calibrator full sky model".format(args.cal_model)

FLAGANT = [f[0] if isinstance(f, list) else f for f in args.flag_antenna]
if len(FLAGANT) != 0:
    print "Will flag antenna {}".format(", ".join(["'{}'".format(a) for a in FLAGANT]))

BPCALIBRATOR = args.bandpass_field
if BPCALIBRATOR is None: raise ValueError("No bandpass calibrator specified")
GCALIBRATOR = [f[0] if isinstance(f, list) else f for f in args.gain_field]
ALTCAL = [f[0] if isinstance(f, list) else f for f in args.alt_cal_field]
TARGET = [f[0] if isinstance(f, list) else f for f in args.target_field]
if len(TARGET) < 1: raise ValueError("No target specified")

DO_USE_GAINCALIBRATOR = len(GCALIBRATOR) > 0
if not DO_USE_GAINCALIBRATOR:
    print "*NO* gain calibrator specified"

DO_USE_GAINCALIBRATOR_DELAY = args.delay_with_gcal and DO_USE_GAINCALIBRATOR
if DO_USE_GAINCALIBRATOR_DELAY:
    print "Will transfer rate calibraton using gain calibrator"
else:
    print "Will *NOT* transfer rate calibraton from gain calibrator"

PREFIX = args.msprefix
REFANT = "m037"

print "Reference antenna {0:s} to be used throughout".format(REFANT)

## DO NOT EDIT ANY OF THESE PREDEFINED WORKERS UNLESS YOU KNOW WHAT YOU'RE DOING

K0 = PREFIX + ".K0"
B0 = PREFIX + ".B0"
G0 = PREFIX + ".G0"
GA = PREFIX + ".GAlt"
G1 = PREFIX + ".G1"
F0 = PREFIX + ".F0"
B1 = PREFIX + ".B1"

ZEROGEN_DATA = PREFIX + ".ms"
FIRSTGEN_DATA = ["{}.{}.1gc.ms".format(t, PREFIX) for t in TARGET]
MANUAL_FLAG_LIST = []

print "Dataset '{0:s}' to be used throughout".format(ZEROGEN_DATA)

with tbl(os.path.join(MSDIR, ZEROGEN_DATA)+"::FIELD", ack=False) as t:
    field_names = t.getcol("NAME")
    FDB = {fn: str(fni) for fni, fn in enumerate(field_names)}

print "The following fields are available:"
for f in FDB:
    print "\t '{0:s}' index {1:s}{2:s}".format(
        f, FDB[f],
        " selected as 'BP'" if f == BPCALIBRATOR else
        " selected as 'GC'" if f in GCALIBRATOR else
        " selected as 'ALTCAL'" if f in ALTCAL else
        " selected as 'TARGET'" if f in TARGET else
        " not selected")

while True:
    r = raw_input("Is this configuration correct? (Y/N) >> ").lower()
    if r == "y":
        break
    elif r == "n":
        sys.exit(0)
    else:
		continue

stimela.register_globals()
recipe = stimela.Recipe('MEERKAT: basic calibration', ms_dir=MSDIR, JOB_TYPE="docker")

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
                   "nio_threads":60
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
                  "config"              : exec_strategy,
                  "dilate-masks": sdm.dismissable(None),
                  "ignore-flags": sdm.dismissable(None),
                  "scan-numbers": sdm.dismissable(None),
        },
        input=INPUT, output=OUTPUT, label=steplabel + ".gc")


        recipe.add("cab/casa_flagdata", "flag_summary_{}".format(steplabel), {
                   "vis": ZEROGEN_DATA,
                   "mode": "summary"
            },
            input=INPUT, output=OUTPUT, label="flagging_summary_{}".format(steplabel))

        return [
          steplabel,
          steplabel + ".gc",
          "flagging_summary_{}".format(steplabel)
        ]

def do_1GC(recipe, label="prelim", do_apply_target=False, do_predict=True, applyonly=False):
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

    recipe.add("cab/casa_gaincal", "delaycal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": K0,
            "field": ",".join([FDB[BPCALIBRATOR]] +
                              [FDB[a] for a in ALTCAL] +
                              [FDB[t] for t in GCALIBRATOR]) if (DO_USE_GAINCALIBRATOR and
                                                                 DO_USE_GAINCALIBRATOR_DELAY) else
                     ",".join([FDB[BPCALIBRATOR] + [FDB[a] for a in ALTCAL]]),
            "refant": REFANT,
            "solint": args.time_sol_interval,
            "combine": "",
            "minsnr": 3,
            "minblperant": 4,
            "gaintype": "K",
            ##"spw": "0:1.3~1.5GHz",
            "uvrange": "150~10000m" # EXCLUDE RFI INFESTATION!
        },
        input=INPUT, output=OUTPUT, label="delay_calibration_%s" % label)

    recipe.add("cab/casa_flagdata", "clipdel_%s" % label, {
            "vis": ZEROGEN_DATA,
            "mode": "clip",
            "clipminmax": [-args.clip_delays, +args.clip_delays],
            "datacolumn": "FPARAM",
            "clipoutside": True,
            "clipzeros": True,
        },
        input=INPUT, output=OUTPUT, label="clip_delay_%s" % label)

    recipe.add("cab/msutils", "delaycal_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % K0,
            "tabtype": "delay",
            "plot_file": "{0:s}.{1:s}.K0.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180
        },
        input=INPUT, output=OUTPUT, label="plot_delays_%s" % label)

    ##capture time drift of bandpass and alternative calibrators
    recipe.add("cab/casa_gaincal", "bandpassgain_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": G0,
            "field": ",".join([FDB[BPCALIBRATOR]] +
                              [FDB[a] for a in ALTCAL]),
            "solint": args.time_sol_interval,
            "combine": "",
            "gaintype": "G",
            "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
            ##"spw": "0:1.3~1.5GHz",
            "gaintable": ["%s:output" % ct for ct in [K0]],
            "gainfield": [FDB[BPCALIBRATOR]],
            "interp":["linear"]
        },
        input=INPUT, output=OUTPUT, label="remove_bp_average_%s" % label)

    recipe.add("cab/msutils", "bpgain_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % G0,
            "tabtype": "gain",
            "plot_file": "{0:s}.{1:s}.G0.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180
        },
        input=INPUT, output=OUTPUT, label="plot_bpgains_%s" % label)

    # no model of alternatives, don't adjust amp
    recipe.add("cab/casa_gaincal", "altcalgain_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": GA,
            "field": ",".join([FDB[a] for a in ALTCAL]),
            "solint": args.time_sol_interval,
            "combine": "",
            "gaintype": "G",
            "calmode": "p",
            "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
            ##"spw": "0:1.3~1.5GHz",
            "gaintable": ["%s:output" % ct for ct in [K0]],
            "gainfield": [FDB[a] for a in ALTCAL],
            "interp":["linear"]
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

    # average as much as possible to get as good SNR on bandpass as possible
    recipe.add("cab/casa_bandpass", "bandpasscal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": "%s:output" % B0,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "solint": args.freq_sol_interval,
            "combine": "scan",
            "minsnr": 3.0,
            "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
            "fillgaps": 100000000, # LERP!
            "gaintable": ["%s:output" % ct for ct in [K0, G0]],
            "gainfield": [FDB[BPCALIBRATOR], FDB[BPCALIBRATOR]],
            "interp": ["linear", "linear"]
        },
        input=INPUT, output=OUTPUT, label="bp_freq_calibration_%s" % label)

    recipe.add("cab/msutils", "bp_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % B0,
            "tabtype": "bandpass",
            "plot_file": "{0:s}.{1:s}.B0.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180
        },
        input=INPUT, output=OUTPUT, label="plot_bandpass_gain_%s" % label)

    recipe.add("cab/casa_gaincal", "gaincal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": "%s:output" % G1,
            "field": ",".join([FDB[BPCALIBRATOR]] + [FDB[t] for t in GCALIBRATOR]) if DO_USE_GAINCALIBRATOR else ",".join([FDB[BPCALIBRATOR]]),
            "solint": args.time_sol_interval,
            "gaintype": "G",
            "uvrange": "150~10000m", # EXCLUDE RFI INFESTATION!
            "gaintable": ["%s:output" % ct for ct in [K0, G0, B0]],
            "interp": ["nearest", "nearest", "linear,linear"],
            "gainfield": [FDB[BPCALIBRATOR],
                          FDB[BPCALIBRATOR],
                          FDB[BPCALIBRATOR]]
        },
        input=INPUT, output=OUTPUT, label="gain_calibration_%s" % label)

    recipe.add("cab/casa_fluxscale", "fluxscale_0252_712_%s" % label, {
            "vis": ZEROGEN_DATA,
            "caltable": "%s:output" % G1,
            "fluxtable": "%s:output" % F0,
            "reference": ",".join([FDB[BPCALIBRATOR]]),
            "transfer": ",".join([FDB[t] for t in GCALIBRATOR]),
        },
        input=INPUT, output=OUTPUT, label="fluxscale_%s" % label)

    recipe.add("cab/msutils", "gain_plot_%s" % label, {
            "command": "plot_gains",
            "ctable": "%s:output" % F0,
            "tabtype": "gain",
            "plot_file": "{0:s}.{1:s}.F0.png".format(PREFIX, label),
            "subplot_scale": 4,
            "plot_dpi": 180

        },
        input=INPUT, output=OUTPUT, label="plot_gain_%s" % label)

    recipe.add("cab/casa_applycal", "apply_sols_bp_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "gaintable": ["%s:output" % ct for ct in [K0,G0,B0]],
            "gainfield": [",".join([FDB[BPCALIBRATOR]]),
                          ",".join([FDB[BPCALIBRATOR]]),
                          ",".join([FDB[BPCALIBRATOR]])],
            "interp": ["nearest","nearest","linear,linear"]
        },
        input=INPUT, output=OUTPUT, label="apply_sols_bp_%s" % label)

    for a in ALTCAL:
        recipe.add("cab/casa_applycal", "apply_sols_ac_%s_%s" % (FDB[a], label), {
                "vis": ZEROGEN_DATA,
                "field": FDB[a],
                "gaintable": ["%s:output" % ct for ct in [K0,GA,B0]],
                "gainfield": [",".join([FDB[a]]),
                              ",".join([FDB[a]]),
                              ",".join([FDB[BPCALIBRATOR]])],
                "interp": ["nearest","nearest","linear,linear"]
            },
            input=INPUT, output=OUTPUT, label="apply_sols_ac_%s_%s" % (FDB[a], label))

    recipe.add("cab/casa_applycal", "apply_sols_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t] for t in GCALIBRATOR] +
                              ([FDB[t] for t in TARGET] if do_apply_target else [])),
            "gaintable": ["%s:output" % ct for ct in [K0,G0,B0,F0]] if DO_USE_GAINCALIBRATOR else ["%s:output" % ct for ct in [K0,G0,B0]],
            "gainfield": [FDB[BPCALIBRATOR],
                          FDB[BPCALIBRATOR],
                          FDB[BPCALIBRATOR]] + \
                          ([",".join([FDB[t] for t in GCALIBRATOR])] if DO_USE_GAINCALIBRATOR
                           else []),
            "interp": ["nearest","nearest","linear,linear"]
        },
        input=INPUT, output=OUTPUT, label="apply_1GC_solutions_%s" % label)

    recipe.add("cab/casa_plotms", "plot_pa_bp_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "correlation": "XX,YY",
            "xaxis": "amp",
            "xdatacolumn": "corrected",
            "yaxis": "phase",
            "ydatacolumn": "corrected",
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
            "xdatacolumn": "corrected",
            "yaxis": "phase",
            "ydatacolumn": "corrected",
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
            "field": ",".join([FDB[t] for t in GCALIBRATOR]),
            "correlation": "XX,YY",
            "xaxis": "real",
            "xdatacolumn": "corrected",
            "yaxis": "imag",
            "ydatacolumn": "corrected",
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
            "xdatacolumn": "corrected",
            "yaxis": "imag",
            "ydatacolumn": "corrected",
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
            "xdatacolumn": "corrected",
            "yaxis": "amp",
            "ydatacolumn": "corrected",
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
            "xdatacolumn": "corrected",
            "yaxis": "phase",
            "ydatacolumn": "corrected",
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


    recipe.add("cab/casa_plotms", "plot_afreq_bpcal_%s" % label, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[BPCALIBRATOR]]),
            "correlation": "XX,YY",
            "xaxis": "freq",
            "xdatacolumn": "corrected",
            "yaxis": "amp",
            "ydatacolumn": "corrected",
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
            "xdatacolumn": "corrected",
            "yaxis": "phase",
            "ydatacolumn": "corrected",
            "coloraxis": "baseline",
            "expformat": "png",
            "exprange": "all",
            "overwrite": True,
            "showgui": False,
            "avgtime": "64",
            "plotfile": "{}.{}.bp.phasefreq.png".format(PREFIX, label)
        },
        input=INPUT, output=OUTPUT, label="pfreq_for_bp_%s" % label)


    return ([
                "set_flux_reference_{}".format(label)
            ] if do_predict else []) + ([
                "delay_calibration_{}".format(label),
                "clip_delay_{}".format(label),
                "plot_delays_{}".format(label),
                "remove_bp_average_{}".format(label),
                "plot_bpgains_{}".format(label),
                "bp_freq_calibration_{}".format(label),
                "plot_bandpass_gain_{}".format(label),
                "gain_calibration_{}".format(label),
                "fluxscale_{}".format(label),
                "plot_gain_{}".format(label),
            ] + ([
                    "remove_altcal_average_{}".format(label),
                    "plot_altgains_{}".format(label),
                 ] if len(ALTCAL) > 0 else [])
                if not applyonly else []) + [
                "apply_sols_bp_{}".format(label)
            ] + [
                "apply_sols_ac_{0:s}_{1:s}".format(FDB[a], label) for a in ALTCAL
            ] + [
                "apply_1GC_solutions_{}".format(label),
                "phaseamp_plot_for_bandpass_{}".format(label),
                "phaseamp_plot_for_gain_{}".format(label),
                "reim_plot_for_gain_{}".format(label),
                "reim_plot_for_bp_{}".format(label),
                "afreq_for_gain_{}".format(label),
                "pfreq_for_gain_{}".format(label),
                "afreq_for_bp_{}".format(label),
                "pfreq_for_bp_{}".format(label)
            ]

def finalize_and_split():
    for ti, t in enumerate(TARGET):
        recipe.add("cab/casa_split", "split_%d" % ti, {
            "vis": ZEROGEN_DATA,
            "field": ",".join([FDB[t]]),
            "outputvis": FIRSTGEN_DATA[ti]
        },
        input=INPUT, output=OUTPUT, label="split_%s" % t)
        recipe.add("cab/casa_flagmanager", "backup_1GC_flags_%d" % ti, {
                   "vis": FIRSTGEN_DATA[ti],
                   "mode": "save",
                   "versionname": "1GC_LEGACY",
            },
            input=INPUT, output=OUTPUT, label="backup_1GC_flags_%s" % t)

    return ["split_%s" % t for t in TARGET] + \
           ["backup_1GC_flags_%s" % t]

## MODIFY STRATEGY FROM HERE ON END
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
if not args.skip_flag_targets:
    STEPS += rfiflag_data(do_flag_targets=True, steplabel="final", exec_strategy="mk_rfi_flagging_target_fields_firstpass.yaml", on_corr_residuals=False, dc="CORRECTED_DATA")
if not args.skip_transfer_to_targets:
    STEPS += do_1GC(recipe, label="apply_only", do_predict=False, do_apply_target=True, applyonly=True)
if not args.skip_final_split:
    STEPS += finalize_and_split()

## FINALLY RUN
if len(STEPS) != 0:
    recipe.run(STEPS)
else:
    print "Nothing to be done. Goodbye!"
