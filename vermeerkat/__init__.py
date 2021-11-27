#!/usr/bin/env python
from __future__ import print_function, absolute_import

import logging
import logging.handlers

import os
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree

import pkg_resources
try:
     __version__ = pkg_resources.require("VermeerKAT")[0].version
except pkg_resources.DistributionNotFound:
    __version__ = "dev"

PIPELINE_LOG = os.path.join(os.getcwd(), "VermeerKAT.log")

class DelayedFileHandler(logging.handlers.MemoryHandler):
    """A DelayedFileHandler is a variation on the MemoryHandler. It will buffer up log
    entries until told to stop delaying, then dumps everything into the target file
    and from then on logs continuously. This allows the log file to be switched at startup."""
    def __init__(self, filename, delay=True):
        logging.handlers.MemoryHandler.__init__(self, 100000, target=logging.FileHandler(filename))
        self._delay = delay

    def shouldFlush(self, record):
        return not self._delay

    def setFilename(self, filename, delay=False):
        self._delay = delay
        self.setTarget(logging.FileHandler(filename))
        if not delay:
           self.flush()

def create_logger():
    """ Create a console logger """
    log = logging.getLogger(__name__)
    cfmt = logging.Formatter(
        ('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)

    filehandler = logging.FileHandler(PIPELINE_LOG)
    filehandler.setFormatter(cfmt)

    log.addHandler(filehandler)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)
    return log, filehandler, console, cfmt

def prompt(prompt_string="Is this configuration correct?", dont_prompt=False):
    while not dont_prompt:
        r = input(f"{prompt_string} (Y/N) >> ").lower()
        if r == "y":
            break
        elif r == "n":
            return False
        else:
            continue
    return True

def init_inputdir(INPUT, dont_prompt=False, dont_clean=False):
    def __merge_input(INPUT, dont_clean=False):
        if dont_clean:
            log.warning(f"Attempting to merge pipeline input files into existing '{INPUT}' directory")
        else:
            log.info(f"Initializing input directory '{INPUT}' for pipelined run")
        mod_path = os.path.dirname(__file__)
        data_dir = os.path.join(mod_path, "data", "input")
        copy_tree(data_dir, INPUT)

    if not os.path.exists(INPUT):
        __merge_input(INPUT)
    elif os.path.isdir(INPUT):
        def __reinit(INPUT, dont_clean=dont_clean):
            if not dont_clean:
                shutil.rmtree(INPUT)
            __merge_input(INPUT, dont_clean=dont_clean)
        if dont_clean:
            __reinit(INPUT, dont_clean=dont_clean)
        elif prompt(prompt_string=f"Input directory '{INPUT}' already exists. Are you sure you want to DELETE and reinitialze?",
                    dont_prompt=dont_prompt):
            __reinit(INPUT, dont_clean=dont_clean)
        else:
            log.info("Aborted because the input directory could be initialized per user request")
            sys.exit(1)
    else:
        raise RuntimeError("A file called {} already exists, but is not a input directory".format(INPUT))

# Create the log object
log, log_filehandler, log_console_handler, log_formatter = create_logger()

def remove_log_handler(hndl):
    log.removeHandler(hndl)


def add_log_handler(hndl):
    log.addHandler(hndl)

NONCURSES = False