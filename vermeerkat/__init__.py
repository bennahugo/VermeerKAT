#!/usr/bin/env python
from __future__ import print_function, absolute_import

import logging
import logging.handlers

import os
import argparse
import sys

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

# Create the log object
log, log_filehandler, log_console_handler, log_formatter = create_logger()

def remove_log_handler(hndl):
    log.removeHandler(hndl)


def add_log_handler(hndl):
    log.addHandler(hndl)

