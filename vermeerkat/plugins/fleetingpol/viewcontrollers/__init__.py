import npyscreen

from vermeerkat.plugins.fleetingpol.viewcontrollers.entry_form import entry_form
from vermeerkat.plugins.fleetingpol.viewcontrollers.option_editor import options_form
class event_loop(npyscreen.NPSAppManaged):
    STEPS = None

    def __getitem__(self, key):
        return self._Forms[key]

    def onStart(self):
        self.addForm("MAIN", entry_form, name="Welcome")
        self.addForm("OPTIONVIEW", options_form, name="FleetingPol options")
