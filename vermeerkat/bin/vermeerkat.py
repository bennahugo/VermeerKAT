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
    tlparser.add_argument("command", nargs="?", help="Subcommand to run", choices=["fieldlist", "transfer"])
    return tlparser

def task_flistr():
    # lazy initialize plugin
    import vermeerkat.plugins.FLISTR

def task_transfer_new_menu():
    # lazy initialize plugin
    import vermeerkat.plugins.BAT

def task_transfer():
    title = "MeerKAT VermeerKAT Pipeline"
    menu = CursesMenu(title, "Main menu")
    def init_menu(opts, menu):
        steps_menu = CursesMenu(title, "Main menu >> Select steps to run")
        def __update(menu):
           key = " ".join(menu.items[menu.selected_option].text.split()[1:])
           opts[key] = not opts[key]
           sel = "[X]" if opts[key] else "[.]"
           menu.items[menu.selected_option].text = "%s %s" % (sel, key)
        def __run_crosscal(opts, menu):
           crosscal.compile_and_run([k for k in list(opts.keys()) if opts[k]])
        def __invert_all(opts, menu):
           for k in menu.items:
               if k.text.find("[X]") < 0 and k.text.find("[.]") < 0:
                   continue
               key = " ".join(k.text.split()[1:])
               opts[key] = not opts[key]
               sel = "[X]" if opts[key] else "[.]"
               k.text = "%s %s" % (sel, key)

        steps_menu.append_item(FunctionItem("Invert selections",
                                            __invert_all,
                                            args=[opts, steps_menu]))
        for k in opts:
            sel = "[X]" if opts[k] else "[.]"
            chbox_item = FunctionItem("%s %s" % (sel, k),
                                      __update,
                                      args=[steps_menu])
            steps_menu.append_item(chbox_item)
        menu.append_item(SubmenuItem("Select pipeline steps", steps_menu, menu=menu))
        menu.append_item(FunctionItem("Run forest run!",
                                      __run_crosscal,
                                      args=[opts, menu]))

    init_menu(steps, menu)
    menu.show()

def main():
    tlparser = create_tlparser()
    tlargs = tlparser.parse_args(sys.argv[1:2])
    if tlargs.version:
        fleetingpol.log.info("VermeerKAT version %s" % vermeerkat.__version__)
        sys.exit(0)
    if tlargs.command == "transfer":
        task_transfer_new_menu()
    elif tlargs.command == "fieldlist":
        task_flistr()
    else:
        tlparser.print_help()

if __name__ == "__main__":
    main()
