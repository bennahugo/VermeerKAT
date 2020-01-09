import npyscreen
import os
import pickle
import vermeerkat
from vermeerkat.plugins.INTROSPECT.viewcontrollers.bat_theme import bat_theme
from vermeerkat.plugins.INTROSPECT.viewcontrollers.message_boxes import error_box

class options_form(npyscreen.FormBaseNew):
    def __init__(self, *args, **kwargs):
        npyscreen.setTheme(bat_theme)
        self.__wt_proc = None
        npyscreen.FormBaseNew.__init__(self, *args, **kwargs)

    @property
    def event_loop(self):
        return self.parentApp

    def display(self, clear=False):
                npyscreen.FormBaseNew.display(self, clear)

    @property
    def lastname(self):
        return "INTROSPECT.last"

    def create(self):
        self.add(npyscreen.TitleText, editable=False,
                 name="Edit runtime steps", rely=2)

        def __on_invert():
            for f in self.event_loop.STEPS:
                self.event_loop.STEPS[f] = not(self.event_loop.STEPS[f])
            self.rls_options.value=[fi for fi, f in enumerate(self.event_loop.STEPS.values()) if f]
            self.rls_options.display()

        def __on_restore_last():
            if not os.path.exists(self.lastname):
                instance = error_box(self.event_loop, "Last file not found!")
                self.event_loop.registerForm("MESSAGEBOX", instance)
                self.event_loop.switchForm("MESSAGEBOX")
                return

            with open(self.lastname, 'rb') as handle:
                unserialized_data = pickle.load(handle)
                if self.event_loop.STEPS.keys() == unserialized_data.keys():
                    for k in self.event_loop.STEPS:
                        self.event_loop.STEPS[k] = unserialized_data[k]
                    self.rls_options.value=[fi for fi, f in enumerate(self.event_loop.STEPS.values()) if f]
                    self.rls_options.display()
                else:
                    instance = error_box(self.event_loop, "Steps of last run does not match current configuration!")
                    self.event_loop.registerForm("MESSAGEBOX", instance)
                    self.event_loop.switchForm("MESSAGEBOX")
                    return

        self.btn_back = self.add(npyscreen.ButtonPress, name="Back", relx=-35, rely=4,
                                 when_pressed_function=self.event_loop.switchFormPrevious, hidden=False)
        self.btn_back = self.add(npyscreen.ButtonPress, name="Restore last options", relx=-35, rely=5,
                                 when_pressed_function=__on_restore_last, hidden=False)
        self.btn_invert = self.add(npyscreen.ButtonPress, name="Invert options", relx=-35, rely=6,
                                   when_pressed_function=__on_invert, hidden=False)

        self.rls_options = self.add(npyscreen.MultiSelect, name="OptionList",
                                    value=[fi for fi, f in enumerate(self.event_loop.STEPS.values()) if f],
                                    values=self.event_loop.STEPS.keys(),
                                    exit_left=True,
                                    exit_right=True,
                                    scroll_exit=True,
                                    rely=8)

        def __on_select():
            for f in self.event_loop.STEPS:
                self.event_loop.STEPS[f] = False

            for f in self.rls_options.get_selected_objects() if self.rls_options.get_selected_objects() is not None else []:
                self.event_loop.STEPS[f] = True

        self.rls_options.when_value_edited = __on_select


