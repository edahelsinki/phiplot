import panel as pn
from phiplot.modules.ui.styling.styling import Styling

class BaseModal:
    """
    Wrapper for displaying a modal dialog containing Markdown content.
    Provides an associated toggle button for opening and closing the modal.

    Args:
        md_path (str): Path to a Markdown file to display.
        button_name (str): Label for the modal toggle button.
        styles (dict, optional): Custom style overrides. Defaults to {}.
    """

    def __init__(self, md_path: str, button_name: str, styles={}):
        self._styling = Styling()
        
        with open(md_path) as f:
            markdown_content = f.read()

        base = dict(color="#000000", background="#ffffff")
        styles = styles | base

        self.modal = pn.Modal(
            pn.Column(pn.pane.Markdown(markdown_content), styles=styles)
        )

        self.button = self.modal.create_button("toggle", name=button_name)

    def __panel__(self):
        return self.modal