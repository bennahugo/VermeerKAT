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
    tlparser.add_argument("command",
                          nargs="?",
                          help="Subcommand to run",
                          choices=["fieldlist", "antlist", "transfer", "poltransfer", "selfcal"])
    return tlparser

def task_flistr():
    # lazy initialize plugin
    import vermeerkat.plugins.FLISTR

def task_alistr():
    # lazy initialize plugin
    import vermeerkat.plugins.ALISTR

def task_transfer(noncurses=False):
    # lazy initialize plugin   
    import vermeerkat 
    if vermeerkat.NONCURSES:
        from vermeerkat.plugins.BAT.crosscal import main
        main()
    else:
        import vermeerkat.plugins.BAT
    
def task_poltransfer(noncurses=False):
    # lazy initialize plugin
    import vermeerkat 
    if vermeerkat.NONCURSES:
        from vermeerkat.plugins.fleetingpol.fleetingpol import main
        main()
    else:
        import vermeerkat.plugins.fleetingpol

def task_selfcal(noncurses=False):
    # lazy initialize plugin
    import vermeerkat 
    if vermeerkat.NONCURSES:
        from vermeerkat.plugins.INTROSPECT.image import main
        main()
    else:
        import vermeerkat.plugins.INTROSPECT

def main():
    tlparser = create_tlparser()
    tlargs = tlparser.parse_args(sys.argv[1:2])
    noncurses = "--noncurses" in sys.argv
    noncurses and sys.argv.pop(sys.argv.index("--noncurses"))
    vermeerkat.NONCURSES = noncurses
    if tlargs.version:
        vermeerkat.log.info("VermeerKAT version %s" % vermeerkat.__version__)
        sys.exit(0)
    if tlargs.command == "transfer":
        task_transfer()
    elif tlargs.command == "fieldlist":
        task_flistr()
    elif tlargs.command == "antlist":
        task_alistr()
    elif tlargs.command == "poltransfer":
        task_poltransfer()
    elif tlargs.command == "selfcal":
        task_selfcal()
    else:
        tlparser.print_help()

if __name__ == "__main__":
    main()
