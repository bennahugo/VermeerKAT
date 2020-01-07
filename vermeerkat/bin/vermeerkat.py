#!/usr/bin/env python
from __future__ import print_function, absolute_import

import sys
import argparse
import cursesmenu
import vermeerkat
from cursesmenu import CursesMenu
from cursesmenu.items import FunctionItem, MenuItem, ExitItem, SubmenuItem

def create_tlparser():
    tlparser = argparse.ArgumentParser(description="MeerKAT VermeerKAT Pipeline")
    tlparser.add_argument("-v", "--version", dest="version", action="store_true")
    tlparser.add_argument("command", nargs="?", help="Subcommand to run", choices=["fieldlist", "antlist", "transfer", "poltransfer"])
    return tlparser

def task_flistr():
    # lazy initialize plugin
    import vermeerkat.plugins.FLISTR

def task_alistr():
    # lazy initialize plugin
    import vermeerkat.plugins.ALISTR

def task_transfer():
    # lazy initialize plugin
    import vermeerkat.plugins.BAT

def task_poltransfer():
    # lazy initialize plugin
    import vermeerkat.plugins.fleetingpol

def main():
    tlparser = create_tlparser()
    tlargs = tlparser.parse_args(sys.argv[1:2])
    if tlargs.version:
        fleetingpol.log.info("VermeerKAT version %s" % vermeerkat.__version__)
        sys.exit(0)
    if tlargs.command == "transfer":
        task_transfer()
    elif tlargs.command == "fieldlist":
        task_flistr()
    elif tlargs.command == "antlist":
        task_alistr()
    elif tlargs.command == "poltransfer":
        task_poltransfer()
    else:
        tlparser.print_help()

if __name__ == "__main__":
    main()
