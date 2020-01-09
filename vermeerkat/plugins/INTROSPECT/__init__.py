from __future__ import print_function

import vermeerkat
import curses
from vermeerkat.plugins.INTROSPECT.viewcontrollers import event_loop

def STEPS():
    #lazy load task along with own argument parser
    global __STEPS
    try:
        __STEPS
    except:
        from vermeerkat.plugins.INTROSPECT import image
        __STEPS = image.define_steps()
    return __STEPS, image.compile_and_run

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
