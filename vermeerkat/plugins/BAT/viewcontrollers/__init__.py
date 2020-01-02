import npyscreen

from vermeerkat.plugins.BAT.viewcontrollers.entry_form import entry_form
from vermeerkat.plugins.BAT.viewcontrollers.option_editor import options_form
from vermeerkat.plugins.BAT.viewcontrollers.execution_form import execution_form
class event_loop(npyscreen.NPSAppManaged):
    STEPS = None

    def __getitem__(self, key):
        return self._Forms[key]

    def onStart(self):
        self.addForm("MAIN", entry_form, name="Welcome")
        self.addForm("OPTIONVIEW", options_form, name="BAT options")
        self.addForm("EXECUTIONVIEW", execution_form, name="Executing BAT")
