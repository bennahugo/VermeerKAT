from __future__ import print_function

import vermeerkat
import curses
from vermeerkat.plugins.fleetingpol.viewcontrollers import event_loop

log = vermeerkat.log

def STEPS():
    #lazy load task along with own argument parser
    global __STEPS
    try:
        __STEPS
    except:
        from vermeerkat.plugins.fleetingpol import fleetingpol
        __STEPS = fleetingpol.define_steps()
    return __STEPS, fleetingpol.compile_and_run

try:
    event_loop.STEPS, event_loop.RUN = STEPS()
    event_loop().run()
except KeyboardInterrupt:
    vermeerkat.log.info("Caught CTRL-C. Breaking")

print("Goodbye!")

import atexit
def cleanupshell():
    curses.endwin()
atexit.register(cleanupshell)
