import vermeerkat
import npyscreen
import os
import types
import curses
import pickle
from vermeerkat.plugins.BAT.viewcontrollers.bat_theme import bat_theme
from vermeerkat.plugins.BAT.viewcontrollers.option_editor import options_form
from vermeerkat.utils.interruptable_process import interruptable_process

class entry_form(npyscreen.FormBaseNew):
    steps = None

    def __init__(self, *args, **kwargs):
        npyscreen.setTheme(bat_theme)
        npyscreen.FormBaseNew.__init__(self, *args, **kwargs)
        self.event_loop

    @property
    def event_loop(self):
        return self.parentApp

    @property
    def lastname(self):
        return "BAT.last"

    def on_run_pressed(self):
        #self.event_loop["EXECUTIONVIEW"].start_pipeline_next_draw = True
        #self.event_loop.switchForm("EXECUTIONVIEW")

        #overwrite lastfile
        with open(self.lastname, 'wb') as handle:
                pickle.dump(self.event_loop.STEPS, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.event_loop.switchForm(None)
        curses.endwin()

        entry_form.__pl_proc = interruptable_process(target=types.MethodType(self.event_loop.RUN.__func__,
                                                     [k for k in self.event_loop.STEPS.keys() if self.event_loop.STEPS[k]]))
        try:
            entry_form.__pl_proc.start()
            entry_form.__pl_proc.join(None)
        except KeyboardInterrupt:
            entry_form.__pl_proc.interrupt()
        try:
            input = raw_input
        except NameError:
            pass
        input("Press return to continue...")
        self.event_loop.switchForm("MAIN")


    def on_edit_pressed(self):
        self.event_loop["OPTIONVIEW"].start_pipeline_next_draw = True
        self.event_loop.switchForm("OPTIONVIEW")

    def on_quit_pressed(self):
        self.event_loop.switchForm(None)
        SystemExit(0)

    def create(self):
        #lazy load task along with own argument parser
        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="VermeerKAT Basic Apply Transfer (BAT) calibration pipeline")
        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="==========================================================")
        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="")
        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="")
        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="")

        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="Module installed at: {0:s} (version {1:s})".format(
					os.path.dirname(vermeerkat.__file__), str(vermeerkat.__version__)))
        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="A logfile will be dumped here: {0:s}".format(vermeerkat.PIPELINE_LOG))
        self.add(npyscreen.TitleText, editable=False, name="\t\t\t",
                 value="Current working directory: {0:s}".format(os.getcwd()))
        self.add(npyscreen.TitleText, editable=False, name="\t")
        self.add(npyscreen.TitleText, editable=False, name="\t")
        self.btn_run = self.add(npyscreen.ButtonPress, name="Alter execution flow",
                                when_pressed_function=self.on_edit_pressed)
        self.btn_edit = self.add(npyscreen.ButtonPress, name="Run forest run!",
                                 when_pressed_function=self.on_run_pressed)
        self.btn_quit = self.add(npyscreen.ButtonPress, name="Quit to command line",
								 when_pressed_function=self.on_quit_pressed)

