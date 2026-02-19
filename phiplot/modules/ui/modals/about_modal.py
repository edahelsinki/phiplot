from phiplot.modules.ui.modals.base_modal import BaseModal

class AboutModal(BaseModal):
    """
    Floating panel displaying application information.

    Reads content from `phiplot/assets/about.md` and presents it
    inside a modal window with a toggle button.
    """

    def __init__(self):
        super().__init__("phiplot/assets/docs/ABOUT.md", "ℹ️ About", styles=dict(width="50vw"))