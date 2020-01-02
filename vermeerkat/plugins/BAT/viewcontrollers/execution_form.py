import npyscreen
from threading import Thread

import types
import vermeerkat
from vermeerkat.utils.stream_director import stream_director
from vermeerkat.utils.interruptable_process import interruptable_process
from vermeerkat.plugins.BAT.viewcontrollers.bat_theme import bat_theme
from vermeerkat.plugins.BAT.viewcontrollers.log_view import log_view


class execution_form(npyscreen.FormBaseNew):
    __pl_proc = None
    __executing = False
    def __init__(self, *args, **kwargs):
        npyscreen.setTheme(bat_theme)
        execution_form.__pl_proc = None
        self.__initial_display = False
        npyscreen.FormBaseNew.__init__(self, *args, **kwargs)
        execution_form.__executing = False

    @property
    def event_loop(self):
        return self.parentApp

    def on_pipeline_complete(self, proc):
        self.lbl_successfailure.value = "Execution {}".format("finished SUCCESSFULLY" if not proc.exitcode else
                                                              "FAILED! See log for details")
        self.lbl_successfailure.color = "STANDOUT" if proc.exitcode != 0 else "SAFE"
        self.btn_back.hidden = False
        self.btn_back.display()
        self.lbl_successfailure.display()
        self.__executing = False

    def on_back_pressed(self):
        self.event_loop.switchFormPrevious()

    def edit(self):
        npyscreen.FormBaseNew.edit(self)

    @property
    def start_pipeline_next_draw(self):
        return self.__initial_display

    @start_pipeline_next_draw.setter
    def start_pipeline_next_draw(self, val):
        if not self.start_pipeline_next_draw:
            self.btn_back.hidden = True
            self.lbl_successfailure.value = "Executing"
            self.lbl_successfailure.color = "DEFAULT"
        self.__initial_display = val

    def display(self, clear=False):
        if self.start_pipeline_next_draw:
            self.start_pipeline_next_draw = False
            self.start_pipeline()

        npyscreen.FormBaseNew.display(self, clear)

    def create(self):
        self.box_logger = self.add(
            npyscreen.BoxBasic, name="Log tail", max_width=-5, max_height=-3, editable=False)
        self.lvw_logger = self.add(
            log_view, editable=False, value="", max_width=-10, max_height=-5, relx=5, rely=4)
        self.lvw_logger.color = "GOOD"
        self.lvw_logger.widgets_inherit_color = True
        self.lbl_successfailure = self.add(npyscreen.Textfield, editable=False,
                                           value="Executing", rely=-4, max_width=80)

        self.btn_back = self.add(npyscreen.ButtonPress, name="Back to main screen", hidden=True, editable=True, relx=-31, rely=-4,
                                 when_pressed_function=self.on_back_pressed)

        # BUG in npyscreen last thing must be enabled and visible
        self.add(npyscreen.ButtonPress, name="", rely=-3, width=0, height=0)

    def start_pipeline(self):
        """ Run pipeline interactively """
        execution_form.__executing = True
        def __block_and_callback(p, callback_func):
            with stream_director(vermeerkat.log):
                try:
                    p.start()
                    p.join(None)
                except KeyboardInterrupt:
                    p.interrupt()
            callback_func(p)

        execution_form.__pl_proc = interruptable_process(target=types.MethodType(self.event_loop.RUN.__func__,
                                                         [k for k in self.event_loop.STEPS.keys() if self.event_loop.STEPS[k]]))

        Thread(target=__block_and_callback, args=(
	           execution_form.__pl_proc, self.on_pipeline_complete)).start()

    @classmethod
    def handle_CTRLC(cls):
        if execution_form.__executing:
            execution_form.__pl_proc.interrupt()
