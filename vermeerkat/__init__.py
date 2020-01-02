#!/usr/bin/env python

import logging
import os
import argparse
import sys

import pkg_resources
try:
     __version__ = pkg_resources.require("VermeerKAT")[0].version
except pkg_resources.DistributionNotFound:
    __version__ = "dev"

PIPELINE_LOG = os.path.join(os.getcwd(), "VermeerKAT.log")

def create_logger():
    """ Create a console logger """
    log = logging.getLogger(__name__)
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(PIPELINE_LOG)
    filehandler.setFormatter(cfmt)
    log.addHandler(filehandler)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log

# Create the log object
log = create_logger()
