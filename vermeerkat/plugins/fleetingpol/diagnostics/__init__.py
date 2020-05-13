import os
import json
import copy
import re
import nbconvert
import nbformat
import vermeerkat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

LEAKAGE_TEMPLATE=os.path.join(os.path.dirname(__file__), "Polarization.ipynb")
SOLUTIONS_TEMPLATE=os.path.join(os.path.dirname(__file__),"Polcal solutions.ipynb")

def generate_calsolutions_report(rep, output="output", calprefix="COMBINED"):
    # read template
    with open(SOLUTIONS_TEMPLATE) as f:
        rep_template = f.read()

    vermeerkat.log.info("Creating a report of polarization solutions. "
                         "The report will be dumped here: '%s'." % (rep))

    # grab a fresh template
    sols_rep = nbformat.reads(rep_template, as_version=4)

    def __customize(s):
        s = re.sub(r'OUTPUT\s*=\s*\S*',
                   'OUTPUT = \'%s\'' % output,
                   s)
        s = re.sub(r"CALPREFIX\s*=\s*\S*",
                   'CALPREFIX = \'%s\'' % calprefix,
                   s)
        return s

    # modify template to add the output directory
    sols_rep.cells[0]['source'] = '\n'.join(map(__customize, sols_rep.cells[0]['source'].split('\n')))

    # roll
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    try:
        ep.preprocess(sols_rep, {'metadata': {'path': os.path.abspath(os.path.dirname(__file__))}})
    except CellExecutionError: # reporting error is non-fatal
        out = None
        msg = 'Error executing the solution notebook.\n\n'
        msg += 'See notebook "%s" for the traceback.' % rep
        vermeerkat.log.error(msg)
    finally:
        #export to static HTML
        html_exporter = HTMLExporter()
        #html_exporter.template_file = 'basic'
        (body, resources) = html_exporter.from_notebook_node(sols_rep)
        with open(rep, 'w+') as f:
            f.write(body)


def generate_leakage_report(ms, rep, field="PKS1934-638"):
    # read template
    with open(LEAKAGE_TEMPLATE) as f:
        rep_template = f.read()

    vermeerkat.log.info("Creating a report for dataset id '%s'. "
                        "The report will be dumped here: '%s'." % (ms, rep))

    # grab a fresh template
    ms_rep = nbformat.reads(rep_template, as_version=4)

    def __customize(s):
        s = re.sub(r'MSNAME\s*=\s*\S*',
                   'MSNAME = \'%s\'' % ms,
                   s)
        s = re.sub(r'UNPOL_SOURCE\s*=\s*\S*',
                   'UNPOL_SOURCE = \'%s\'' % field,
                   s)
        return s

    # modify template to add the ms name
    ms_rep.cells[0]['source'] = '\n'.join(map(__customize, ms_rep.cells[0]['source'].split('\n')))

    # roll
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    try:
        ep.preprocess(ms_rep, {'metadata': {'path': os.path.abspath(os.path.dirname(__file__))}})
    except CellExecutionError: # reporting error is non-fatal
        out = None
        msg = 'Error executing the notebook for "%s".\n\n' % ms
        msg += 'See notebook "%s" for the traceback.' % rep
        vermeerkat.log.error(msg)
    finally:
        #export to static HTML
        html_exporter = HTMLExporter()
        #html_exporter.template_file = 'basic'
        (body, resources) = html_exporter.from_notebook_node(ms_rep)
        with open(rep, 'w+') as f:
            f.write(body)

def testms(msname, reportname):
    generate_leakage_report(msname, reportname)

def testsols(reportname, output):
    generate_calsolutions_report(reportname, output)

if __name__ == "__main__":
    testms("/scratch/bhugo/modelling0408-65/msdir/COMBINED1GC.ms", "/tmp/polarization_report.ipynb.html")
    testsols("/tmp/polarization_solutions.ipynb.html", output="/scratch/bhugo/modelling0408-65/output")
