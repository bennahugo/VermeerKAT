from __future__ import print_function

import vermeerkat
import curses
from vermeerkat.plugins.BAT.viewcontrollers import event_loop

def STEPS():
    #lazy load task along with own argument parser
    global __STEPS
    try:
        __STEPS
    except:
        from vermeerkat.plugins.BAT import crosscal
        __STEPS = crosscal.define_steps()
    return __STEPS, crosscal.compile_and_run
NONCURSES = vermeerkat.NONCURSES
if not NONCURSES:
    try:
        event_loop.STEPS, event_loop.RUN = STEPS()
        event_loop().run()
    except KeyboardInterrupt:
        vermeerkat.log.info("Caught CTRL-C. Breaking")
    print("Goodbye!")

import atexit
def cleanupshell():
    curses.endwin()
NONCURSES or atexit.register(cleanupshell)
