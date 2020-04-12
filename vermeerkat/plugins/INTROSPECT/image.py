from __future__ import print_function, absolute_import

import stimela
import stimela.dismissable as sdm
from stimela.pathformatter import pathformatter as spf
import copy
import os
import time
import numpy as np
from pyrap.tables import table as tbl
import sys
import argparse
from collections import OrderedDict
import shutil
import vermeerkat
import re

parser = argparse.ArgumentParser("MeerKAT Introspect Selfcal pipeline")
parser.add_argument('--input_dir', dest='input_directory', metavar='<input directory>',
                    action='store', default='input',
                    help='directory to read input data with e.g. antenna table and beam files')
parser.add_argument('--output_dir', dest='output_directory', metavar='<output directory>',
                    action='store', default='output', help='directory to write output data')
parser.add_argument('--msdir_dir', dest='msdir_directory', metavar='<msdir directory>',
                    action='store', default='msdir', help='directory to store measurement sets')
parser.add_argument('--tar', dest='target_field', metavar='<target fields>', type=str, default=[],
                    action="append", nargs="+",
                    help='Target fields. This switch can be used multiple times for more than 1 field')
parser.add_argument('msprefix', metavar='<measurement set name prefix>',
                    help='Prefix of measurement set name as it appears relative to msdir. This must NOT be a '
                         'path prefix')
parser.add_argument('--clip_delays', dest='clip_delays', default=1, type=float,
                    help="Clip delays above this absolute in nanoseconds")
parser.add_argument('--mfs_bands', dest='mfs_bands', default=4, type=int,
                    help="MFS bands to use during imaging (default 8)")
parser.add_argument('--mfs_predictbands', dest='mfs_predictbands', default=15, type=int,
                    help="Number of predict bands to use during imaging (default 10)")
parser.add_argument('--ref_ant', dest='ref_ant', default='m037',
                    help="Reference antenna to use throughout")
parser.add_argument("--containerization", dest="containerization", default="docker",
                    help="Containerization technology to use. See your stimela installation for options")
default_recipe = "p(35,256s),p(35,64s),dp(35,16s),dd(45,4.0,20,50,CORRECTED_DATA,DE_DATA),i(DE_DATA,0.0),s(DE_DATA),i(SUBTRACTED_DATA,0.0)"
parser.add_argument("--recipe", dest="recipe",
                    default=default_recipe,
                    help="Selfcal steps to perform of the format cal(mask sigma,solint) or im(imcol,briggs) or s(datacol) or "
                         "dd(mask_sigma, tag_sigma, timesolint, freqsolint, incol, outcol) "
                         "Available options for cal are p - phase, dp - delay+phase, ap - ampphase "
                         "Available options for im are currently only i with customization of image column and robust weighting. "
                         "s subtracts MODEL_DATA from named datacol to form SUBTRACTED_DATA for all fields. "
                         "Available options for 'dd' direction dependent cal are currently masking sigma of the model image, "
                         "tagging threshold, solution interval in time integraions (NOT seconds), frequency integration in channels, "
                         "input column and output column (need not exist) "
                         "(default: '{}')".format(default_recipe))
parser.add_argument("--npix", dest="npix", default=8192,
                    help="Number of pixels to use in imaging (default 8192)")
parser.add_argument("--cellsize", dest="cellsize", default=1.3,
                    help="Cellsize to use in imaging (default 1.3asec)")
parser.add_argument("--cal_briggs", dest="cal_briggs", default=-0.6,
                    help="Briggs robust to use during calibration (default -0.6)")
parser.add_argument("--imaging_data_chunk_hours", dest="imaging_data_chunk_hours", default=0.05, type=float,
                    help="Chunking hours (default: 0.05 hours)")
parser.add_argument("--ncubical_workers", dest="ncubical_workers", default=25, type=int,
                    help="Number of cubical workers (default: 25)")
parser.add_argument("--cubical_split_DI_bandwidth", dest="cubical_split_DI_bandwidth",
                    action="store_true",
                    help="Split the fractional bandwidth into chunks the size of the DD frequency solution bin "
                         "for the DI case as well. By default the DI phase gain is frequency independent. "
                         "This chunks up the data much finer and may help eleviate memory pressure.")

args = parser.parse_args(sys.argv[2:])

INPUT = args.input_directory
MSDIR = args.msdir_directory
OUTPUT = args.output_directory
vermeerkat.log.info("Directory '{0:s}' is used as input directory".format(INPUT))
vermeerkat.log.info("Directory '{0:s}' is used as output directory".format(OUTPUT))
vermeerkat.log.info("Directory '{0:s}' is used as msdir directory".format(MSDIR))

PREFIX = args.msprefix
DATASET = PREFIX + ".ms"
vermeerkat.log.info("Dataset '{0:s}' to be used throughout".format(DATASET))

with tbl(os.path.join(MSDIR, DATASET)+"::FIELD", ack=False) as t:
    field_names = t.getcol("NAME")
    FDB = {fn: str(fni) for fni, fn in enumerate(field_names)}

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

TARGET = [f[0] if isinstance(f, list) else f for f in args.target_field]
if len(TARGET) < 1: raise ValueError("No target specified")

vermeerkat.log.info("Will be self calibrating:")
for t in TARGET:
    vermeerkat.log.info("\t{} (field id {})".format(t, FDB[t]))

REFANT = args.ref_ant
vermeerkat.log.info("Reference antenna {0:s} to be used throughout".format(REFANT))

reg = r"(?:(?P<cal>p|dp|ap)\((?P<s>\d+),(?P<int>\d+s?)\)),?|"\
      r"(?:(?P<img>i)\((?P<imcol>[a-zA-Z_]+),(?P<briggs>[+-]?\d+(?:.\d+)?)),?|"\
      r"(?:(?P<sub>s)\((?P<subcol>[a-zA-Z_]+)\)),?|"\
      r"(?:(?P<ddcal>dd)\((?P<ddsig>\d+),(?P<ddtag>\d+(?:.\d+)?),"\
      r"(?P<ddtimeint>\d+?),(?P<ddfreqint>\d+?),(?P<ddincol>[a-zA-Z_]+),(?P<ddoutcol>[a-zA-Z_]+)\)),?"

reg_eval = re.compile(reg)
calrecipe = list(reg_eval.finditer(args.recipe))
if len(calrecipe) == 0:
    raise ValueError("Recipe specification is invalid, see help")

vermeerkat.log.info("Steps to take:")
for cmd in map(lambda x: x.groupdict(), calrecipe):
    if cmd['cal']:
        vermeerkat.log.info("\t DI calibrate {} with mask set at {} sigma, calibrated at interval of {}".format(
            "phase" if cmd['cal'] == "p" else
            "delay+phase" if cmd['cal'] == "dp" else
            "ampphase" if cmd['cal'] == "ap" else
            "UNDEFINED",
            cmd['s'], cmd['int']))
    elif cmd['img']:
        vermeerkat.log.info("\t Image {} of target fields at briggs weighting {}".format(cmd['imcol'], cmd['briggs']))
    elif cmd['sub']:
        vermeerkat.log.info("\t Subtract last model from {}".format(cmd['subcol']))
    elif cmd['ddcal']:
        vermeerkat.log.info("\t DD calibrate {} with masking set at {} sigma, de tagging at {} sigma with "
                            "solution interval of [{}, {}] into {}".format(cmd["ddincol"], cmd["ddsig"], cmd["ddtag"],
                                                                           cmd["ddtimeint"], cmd["ddfreqint"], cmd["ddoutcol"]))
    else:
        raise ValueError("Unknown step in recipe")

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
recipe = stimela.Recipe('MeerKAT: INTROSPECT selfcal pipeline',
                        ms_dir=MSDIR,
                        singularity_image_dir=os.environ.get("SINGULARITY_PULLFOLDER", ""),
                        JOB_TYPE=args.containerization)

def image(incol="DATA",
          label='initial',
          tmpimlabel="",
          nfacets=19,
          masksig=25,
          briggs=args.cal_briggs,
          do_mask=True,
          restore=None,
          do_taper=False,
          taper_inner_cut=100, #meters
          taper_outer_cut=1500, #meters
          taper_gamma=200,
          rime_forward=None,
          model_data="MODEL_DATA",
          weight_col="WEIGHT"):
    steps = []
    for ti, t in enumerate(TARGET):
        image_opts =  {
            "Data-MS": DATASET,
            "Data-ColName": incol,
            "Data-ChunkHours": args.imaging_data_chunk_hours,
            "Selection-Field": int(FDB[t]),
            "Output-Mode": "Clean" if not restore else "Predict",
            "Output-Name": t + tmpimlabel,
            "Output-Images": "all",
            "Output-Cubes": "all",
            "Image-NPix": args.npix,
            "Image-Cell": args.cellsize,
            "Facets-NFacets": nfacets,
            "Weight-ColName": weight_col,
            "Weight-Mode": "Briggs",
            "Weight-Robust": briggs,
            #"Beam-Model": "FITS",
            #"Beam-FITSFile": "'MeerKAT_VBeam_10MHz_53Chans_$(corr)_$(reim).fits':output",
            "Freq-NBand": args.mfs_bands,
            "Freq-NDegridBand": args.mfs_predictbands,
            "Deconv-RMSFactor": 0,
            "Deconv-PeakFactor": 0.25,
            "Deconv-Mode": "Hogbom",
            "Deconv-MaxMinorIter": 50000,
            "Hogbom-PolyFitOrder": 6,
            "Deconv-Gain": 0.1,
            "Deconv-FluxThreshold": 1.0e-6,
            "Deconv-AllowNegative": True,
            "Log-Boring": True,
            "Log-Memory": True,
            "RIME-ForwardMode": sdm.dismissable(rime_forward),
            "Predict-ColName": model_data,
            "Predict-FromImage": sdm.dismissable(t + restore + ":output" if restore is not None else restore),
        }
        if do_taper:
            def taper_weigh(ms):
                from pyrap.tables import table as tbl
                with tbl(ms, readonly=False) as t:
                    uvw = t.getcol("UVW")
                    max_uv = np.sqrt(np.max(uvw[:,0]**2 + uvw[:,1]**2))
                    taper = lambda u, v, a, b, gamma: (1.0 / (1 + np.exp((np.sqrt(u**2 + v**2) - b) / (2.0 * max_uv / gamma))) +
                                                       1.0 / (1 + np.exp((-np.sqrt(u**2 + v**2) + a) / (2.0 * max_uv / gamma))) +
                                                       1.0 / (1 + np.exp((-np.sqrt(u**2 + v**2) - b) / (2.0 * max_uv / gamma))) + 
                                                       1.0 / (1 + np.exp((np.sqrt(u**2 + v**2) + a) / (2.0 * max_uv / gamma)))) - 2.0

                    weight = t.getcol("WEIGHT")
                    weight_new = weight.copy()
                    tp_weight = taper(uvw[:, 0], uvw[:, 1],
                                      taper_inner_cut, # inner cut
                                      taper_outer_cut, # outer cut
                                      taper_gamma)
                    weight_new *= tp_weight[:, None]
                    import matplotlib
                    matplotlib.use('Agg')
                    from matplotlib import pyplot as plt
                    import os
                    from scipy.interpolate import griddata
                    plt.figure()
                    x = np.linspace(np.min(uvw), np.max(uvw), 1024)
                    xx, xy = np.meshgrid(x, x, sparse=False)
                    ###grid = griddata(uvw[:, 0:2], tp_weight, (xx, xy), method="linear")
                    grid = taper(xx, xy, taper_inner_cut, taper_outer_cut, taper_gamma) 
                    plt.imshow(grid, cmap="magma", extent=[np.min(xx), np.max(xx), np.max(xx), np.min(xx)])
                    plt.xlabel("u (m)"); plt.ylabel("v (m)")
                    plt.savefig(os.path.join(OUTPUT, "uvtaper_{0:d}.png".format(ti)))

                    t.putcol("WEIGHT", weight_new)
                    t.close()
            recipe.add(taper_weigh, "taper_target_%d" % ti, {
                "ms": "%s/%s.%s.1GC.ms" % (MSDIR, PREFIX, t)
            }, input=INPUT, output=OUTPUT, label="taper %s %s" % (label, t))
            steps.append("taper %s %s" % (label, t))

        recipe.add("cab/ddfacet", "image_target_%d" % ti, image_opts, 
                   input=INPUT, output=OUTPUT, label="image %s %s" % (label, t), 
                   shared_memory="500g")
        steps.append("image %s %s" % (label, t))

        if not restore:
            if do_mask:
                recipe.add("cab/cleanmask", "mask_target_%d" % ti, {
                    'image': "%s.app.restored.fits:output" % t,
                    'output': "%s.mask.fits" % t,
                    'sigma': masksig,
                    'boxes': 9,
                    'iters': 20,
                    'overlap': 0.3,
                    'no-negative': True,
                    'tolerance': 0.75,
                }, input=INPUT, output=OUTPUT, label='mask %s %s' % (label, t))
                steps.append('mask %s %s' % (label, t))

                maskimage_opts = copy.deepcopy(image_opts)
                maskimage_opts["Predict-ColName"] = "MODEL_DATA"
                maskimage_opts["Mask-External"] = "%s.mask.fits:output" % t
                maskimage_opts["Output-Name"] = t + "_" + label

                recipe.add("cab/ddfacet", "image_target_%d" % ti, maskimage_opts,
                           input=INPUT, output=OUTPUT, label="mask image %s %s" % (label, t),
                           shared_memory="500g")
                steps.append("mask image %s %s" % (label, t))
    return steps

previous_caltables = {f: [] for f in TARGET}

def calibrate(incol="DATA",
              label='initial',
              masksig=45,
              corrtype='p',
              interval='int',
              restore=None):
    steps = []
    steps += image(incol=incol, label=label, masksig=masksig, restore=restore)
    for ti, t in enumerate(TARGET):
        if corrtype == "p":
            caltable_label = "{}_{}_{}".format(label, t, "Gp")
            recipe.add("cab/casa47_gaincal", "calibrate_{}_{}_{}".format(label, "Gp", ti), {
                "vis": DATASET,
                "caltable": caltable_label,
                "field": ",".join([FDB[t]]),
                "refant": REFANT,
                "solint": interval,
                "combine": "",
                "minsnr": 3,
                "minblperant": 4,
                "gaintype": "G",
                "calmode": "p",
                "uvrange": "400~10000m",
                "gaintable": ["%s:output" % ct for ct in previous_caltables[t]],
            }, input=INPUT, output=OUTPUT, label="calibrate_{}_{}_{}".format(label, "Gp", ti), shared_memory="250g")
            steps.append("calibrate_{}_{}_{}".format(label, "Gp", ti))
            previous_caltables[t].append(caltable_label)
            recipe.add("cab/msutils", "gain_plot_{}_{}_{}".format(label, "Gp", ti), {
                "command": "plot_gains",
                "ctable": "%s:output" % caltable_label,
                "tabtype": "gain",
                "plot_file": "{0:s}.{1:s}.png".format(PREFIX, caltable_label),
                "subplot_scale": 4,
                "plot_dpi": 180
            }, input=INPUT, output=OUTPUT, label="plot_gain_{}_{}_{}".format(label, "Gp", ti))
            steps.append("plot_gain_{}_{}_{}".format(label, "Gp", ti))
        elif corrtype == "ap":
            caltable_label = "{}_{}_{}".format(label, t, "Gap")
            recipe.add("cab/casa47_gaincal", "calibrate_{}_{}_{}".format(label, "Gap", ti), {
                "vis": DATASET,
                "caltable": caltable_label,
                "field": ",".join([FDB[t]]),
                "refant": REFANT,
                "solint": interval,
                "combine": "",
                "minsnr": 3,
                "minblperant": 4,
                "gaintype": "G",
                "uvrange": "400~10000m",
                "calmode": "ap",
                "gaintable": ["%s:output" % ct for ct in previous_caltables[t]],
            }, input=INPUT, output=OUTPUT, label="calibrate_{}_{}_{}".format(label, "Gap", ti), shared_memory="250g")
            steps.append("calibrate_{}_{}_{}".format(label, "Gap", ti))
            previous_caltables[t].append(caltable_label)
            recipe.add("cab/msutils", "gain_plot_{}_{}_{}".format(label, "Gap", ti), {
                "command": "plot_gains",
                "ctable": "%s:output" % caltable_label,
                "tabtype": "gain",
                "plot_file": "{0:s}.{1:s}.png".format(PREFIX, caltable_label),
                "subplot_scale": 4,
                "plot_dpi": 180
            }, input=INPUT, output=OUTPUT, label="plot_gain_{}_{}_{}".format(label, "Gap", ti))
            steps.append("plot_gain_{}_{}_{}".format(label, "Gap", ti))
        elif corrtype == "dp":
            caltable_label = "{}_{}_{}".format(label, t, "K")
            recipe.add("cab/casa47_gaincal", "calibrate_{}_{}_{}".format(label, "K", ti), {
                "vis": DATASET,
                "caltable": caltable_label,
                "field": ",".join([FDB[t]]),
                "refant": REFANT,
                "solint": interval,
                "combine": "",
                "minsnr": 3,
                "minblperant": 4,
                "gaintype": "K",
                "gaintable": ["%s:output" % ct for ct in previous_caltables[t]],
            }, input=INPUT, output=OUTPUT, label="calibrate_{}_{}_{}".format(label, "K", ti), shared_memory="250g")
            steps.append("calibrate_{}_{}_{}".format(label, "K", ti))
            previous_caltables[t].append(caltable_label)

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

            recipe.add(clip_delays, "clipdel_{}_{}".format(label, ti), {
                "vis": caltable_label,
                "clipminmax": [-args.clip_delays, +args.clip_delays],
            }, input=INPUT, output=OUTPUT, label="clip_delay_{}_{}".format(label, ti))
            steps.append("clip_delay_{}_{}".format(label, ti))

            recipe.add("cab/msutils", "gain_plot_{}_{}_{}".format(label, "K", ti), {
                "command": "plot_gains",
                "ctable": "%s:output" % caltable_label,
                "tabtype": "delay",
                "plot_file": "{0:s}.{1:s}.png".format(PREFIX, caltable_label),
                "subplot_scale": 4,
                "plot_dpi": 180
            }, input=INPUT, output=OUTPUT, label="plot_gain_{}_{}_{}".format(label, "K", ti))
            steps.append("plot_gain_{}_{}_{}".format(label, "K", ti))

            caltable_label = "{}_{}_{}".format(label, t, "Gp")
            recipe.add("cab/casa47_gaincal", "calibrate_{}_{}_{}".format(label, "Gp", ti), {
                "vis": DATASET,
                "caltable": caltable_label,
                "field": ",".join([FDB[t]]),
                "refant": REFANT,
                "solint": interval,
                "combine": "",
                "minsnr": 3,
                "minblperant": 4,
                "gaintype": "G",
                "calmode": "ap",
                "uvrange": "400~10000m",
                "gaintable": ["%s:output" % ct for ct in previous_caltables[t]],
            }, input=INPUT, output=OUTPUT, label="calibrate_{}_{}_{}".format(label, "Gp", ti), shared_memory="250g")
            steps.append("calibrate_{}_{}_{}".format(label, "Gp", ti))
            previous_caltables[t].append(caltable_label)
            recipe.add("cab/msutils", "gain_plot_{}_{}_{}".format(label, "Gp", ti), {
                "command": "plot_gains",
                "ctable": "%s:output" % caltable_label,
                "tabtype": "gain",
                "plot_file": "{0:s}.{1:s}.png".format(PREFIX, caltable_label),
                "subplot_scale": 4,
                "plot_dpi": 180
            }, input=INPUT, output=OUTPUT, label="plot_gain_{}_{}_{}".format(label, "Gp", ti))
            steps.append("plot_gain_{}_{}_{}".format(label, "Gp", ti))
        else:
            raise ValueError("Unknown calibration mode {}".format(corrtype))

        recipe.add("cab/casa47_applycal", "apply_sols_{}_{}".format(label, ti), {
                "vis": DATASET,
                "field": ",".join([FDB[t]]),
                "gaintable": ["%s:output" % ct for ct in previous_caltables[t]],
         },
        input=INPUT, output=OUTPUT, label="apply_sols_{}_{}".format(label, ti))
        steps.append("apply_sols_{}_{}".format(label, ti))

    return steps

def finalize_and_split():
    for ti, t in enumerate(TARGET):
        recipe.add("cab/casa_split", "split_%d" % ti, {
            "vis": DATASET,
            "field": ",".join([FDB[t]]),
            "outputvis": "{}_{}_SELFCAL.ms".format(PREFIX, t)
        },
        input=INPUT, output=OUTPUT, label="split_%d" % ti)
        recipe.add("cab/casa_flagmanager", "backup_1GC_flags_%d" % ti, {
                   "vis": "{}_{}_SELFCAL.ms".format(PREFIX, t),
                   "mode": "save",
                   "versionname": "1GC_LEGACY",
            },
            input=INPUT, output=OUTPUT, label="backup_1GC_flags_%d" % ti)

    return ["split_%d" % ti for ti, t in enumerate(TARGET)] + \
           ["backup_selfcal_flags_%d" % ti for ti, t in enumerate(TARGET)]


#def blcalibrate(incol="CORRECTED_DATA", 
#              calincol="CORRECTED_DATA",
#              outcol="CORRECTED_DATA", 
#              model="MODEL_DATA",
#              label='blcal',
#              solvemode='complex-diag',
#              corrtype='sc',
#              freq_int=100000,
#              interval=[80, 80, 80],
#            ):
#    for ti, t in enumerate(TARGET):
#        f = t + label
#        recipe.add("cab/cubical", "calibrate_target_%d" % ti, {
#            'data-ms': "%s.%s.1GC.ms" % (PREFIX, t),
#            'data-column': calincol,
#            'data-time-chunk': 100,
#            'data-freq-chunk': freq_int,
#            'model-list': "MODEL_DATA",
#            'model-beam-pattern': "'MeerKAT_VBeam_10MHz_53Chans_$(corr)_$(reim).fits':output",
#            'model-beam-l-axis' : "X",
#            'model-beam-m-axis' : "Y",
#            'weight-column': "WEIGHT",
#            'flags-apply': "-cubical",
#            'flags-auto-init': "legacy",
#            'madmax-enable': False,
#            'sol-jones': 'bbc',
#            'out-name': t + str(time.time()),
#
#            'out-mode': corrtype,
#            'out-column': outcol,
#            'bbc-time-int': interval[ti],
#            'bbc-freq-int': 0,
#            'bbc-type': solvemode,
#            'bbc-max-iter': 1000,
#            'bbc-load-from': sdm.dismissable("{data[ms]}/BBC-field_{sel[field]}-ddid_{sel[ddid]}.parmdb" if corrtype == "ac" else None),
#            'dist-ncpu': 16,
#            'out-overwrite': True
#
#        }, input=INPUT, output=OUTPUT, label="calibrate %s %s" % (label, t), shared_memory="250g")

def decalibrate(incol="corrected_data",
              calincol="corrected_data",
              outcol="subtracted_data",
              model="model_data",
              label='decal',
              freq_int=8,
              masksig=45,
              tagsigma=3.0,
              solvemode='complex-2x2',
              corrtype='sc',
              interval=40,
              restore=None):
    steps = []
    steps += image(incol=incol, label=label, masksig=masksig, restore=restore)

    for ti, t in enumerate(TARGET):
        recipe.add("cab/catdagger", "auto_tagger_{}_{}".format(label, ti), {
            'ds9-reg-file': "{}.{}.dE.reg".format(label, ti),
            'ds9-tag-reg-file': "{}.{}.dE.clusterleads.reg".format(label, ti),
            'sigma': tagsigma,
            'min-distance-from-tracking-centre': 350,
            'noise-map': "{}.app.residual.fits:output".format(t + "_" + label),
        }, input=INPUT, output=OUTPUT, label="auto_tagger_{}_{}".format(label, ti), shared_memory="250g")

        diconame = "{}.DicoModel".format(t + "_" + label)
        tagregs = "{}.{}.dE.reg".format(label, ti)
        recipe.add("cab/cubical", "de_calibrate_{}_{}".format(label, ti), {
                'data-ms': DATASET,
                'data-column': calincol,
                'dist-nworker': args.ncubical_workers,
                'dist-nthread': args.ncubical_workers,
                'dist-max-chunks': args.ncubical_workers,
                'data-time-chunk': interval*int(min(1, args.ncubical_workers)) if not args.cubical_split_DI_bandwidth else \
                                   interval*int(min(1, np.sqrt(args.ncubical_workers))),
                'data-freq-chunk': 0 if not args.cubical_split_DI_bandwidth else \
                                   freq_int*int(min(1, np.sqrt(args.ncubical_workers))),
                'model-list': spf("MODEL_DATA+-{{}}{}@{{}}{}:{{}}{}@{{}}{}".format(diconame, tagregs, diconame, tagregs),
                                  "output", "output", "output", "output"),
                #'model-beam-pattern': "'MeerKAT_VBeam_10MHz_53Chans_$(corr)_$(reim).fits':output",
                #'model-beam-l-axis' : "X",
                #'model-beam-m-axis' : "Y",
                'weight-column': "WEIGHT",
                'flags-apply': "FLAG",
                'flags-auto-init': "legacy",
                'madmax-enable': False,
                'madmax-threshold': [0,0,10],
                'madmax-global-threshold': [0,0],
                'sol-jones': 'g,dd',
                'sol-min-bl': 110.0,
                'sol-max-bl': 0,
                'sol-stall-quorum': 0.95,
                'sol-term-iters': [50,90,50,90],
                'out-name': t + str(time.time()),

                'out-mode': corrtype,
                'out-column': outcol,
                'out-model-column': "MODEL_OUT",
                'log-verbose': "solver=2",
                'dd-time-int': interval,
                'dd-freq-int': freq_int,
                'dd-type': solvemode,
                'g-type': solvemode,
                'g-freq-int': 0,
                'g-time-int': interval,
                'g-max-iter': 100,
                'g-update-type': "phase-diag",
                'dd-max-iter': 200,
                'dd-clip-high': 0,
                'dd-clip-low': 0,
                'dd-fix-dirs': "0",
                'out-subtract-dirs': '1:',
                'dd-dd-term': True,
                'dd-max-prior-error': 0.35,
                'dd-max-post-error': 0.35,
                'g-clip-high': 0,
                'g-clip-low': 0,
                'g-max-prior-error': 0.35,
                'g-max-post-error': 0.35,
                'degridding-NDegridBand': args.mfs_predictbands,
                'degridding-MaxFacetSize': 0.15
        }, input=INPUT, output=OUTPUT, label="de_calibrate_{}_{}".format(label, ti), shared_memory="250g")
    steps += [
        "auto_tagger_{}_{}".format(label, ti),
        "de_calibrate_{}_{}".format(label, ti)
    ]
    return steps

def subtract_model_from_corrected(colname="CORRECTED_DATA"):
    steps = []
    recipe.add("cab/msutils", "subtract", {
                "command": 'sumcols',
                "msname":  DATASET,
                "colname":  "SUBTRACTED_DATA",
                "col1": colname,
                "col2": "MODEL_DATA",
                "subtract": True
    }, input=INPUT, output=OUTPUT, label="subtract_mod_from_corr")
    steps.append("subtract_mod_from_corr")
    return steps

def define_steps():
    STEPS = []
    initcal = True
    #for xi, (x, sig, intdur, xx, imcol, briggs, xxx) in enumerate(calrecipe):
    for xi, cmd in enumerate(map(lambda x: x.groupdict(), calrecipe)):
        if cmd["cal"]:
           STEPS += calibrate(incol="DATA" if initcal else "CORRECTED_DATA",
                              label='calcycle_{}'.format(xi),
                              masksig=int(cmd["s"]),
                              corrtype=cmd["cal"],
                              interval=cmd["int"],
                              restore=None)
           initcal = False
        elif cmd["img"]:
            STEPS += image(incol=cmd["imcol"].strip(),
                           label='image_{}'.format(xi),
                           tmpimlabel="_image_{}".format(xi),
                           briggs=float(cmd["briggs"]),
                           do_mask=False,
                           model_data="None",
                           weight_col="WEIGHT")
        elif cmd["sub"]:
            STEPS += subtract_model_from_corrected(colname=cmd["subcol"].strip())
        elif cmd['ddcal']:
            STEPS += decalibrate(incol=cmd["ddincol"].strip(),
                        calincol=cmd["ddincol"].strip(),
                        outcol=cmd["ddoutcol"].strip(),
                        model="MODEL_DATA",
                        label='decal_{}'.format(xi),
                        freq_int=int(cmd["ddfreqint"]),
                        masksig=int(cmd["ddsig"]),
                        tagsigma=float(cmd["ddtag"]),
                        solvemode='complex-diag',
                        corrtype='sr',
                        interval=int(cmd["ddtimeint"]),
                        restore=None)
        else:
            raise ValueError("Only accepts p, ap, dp, i or s in recipe")

    STEPS += finalize_and_split()
    steps_enabled = OrderedDict()
    for s in STEPS: steps_enabled[s] = True
    return steps_enabled

def compile_and_run(STEPS):
    if len(STEPS) != 0:
        recipe.run(STEPS)

