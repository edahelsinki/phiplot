from phiplot.modules.ui.modals.base_modal import BaseModal

class HelpModal(BaseModal):
    """
    Floating panel displaying help and usage instructions.

    Reads content from `phiplot/assets/help.md` and presents it
    inside a modal window with a toggle button.
    """

    def __init__(self):
        super().__init__(
            "phiplot/assets/docs/HELP.md",
            "❔ Help",
            styles=dict(overflow_y="auto", height="80vh", width="70vw"),
        )