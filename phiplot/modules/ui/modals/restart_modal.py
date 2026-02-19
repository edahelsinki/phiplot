from __future__ import annotations
from typing import TYPE_CHECKING
import panel as pn
from phiplot.modules.ui.styling.styling import Styling

if TYPE_CHECKING:
    from phiplot.main import App

class RestartModal:
    def __init__(self, app: App):
        self.app = app

        self.modal = self._build()

        self.button = self.modal.create_button(
            "toggle",
            name="🔄 Restart",
            button_type="danger"
        )

    def __panel__(self):
        return self.modal

    def _build(self) -> pn.Modal:
        confirm_text = pn.pane.HTML(
            """
            <p style="text-align: center;">
                <b>Are you sure you want to restart?</b><br>
                All session related data will be lost.
            </p>
            """
        )

        no_button = pn.widgets.Button(name="Cancel", button_type="primary")
        no_button.on_click(lambda event: self.modal.toggle())

        yes_button = pn.widgets.Button(name="🔄 Restart", button_type="danger")
        yes_button.on_click(lambda event: self.app.restart())
        
        return pn.Modal(
            pn.Column(
                confirm_text,
                pn.Row(
                    pn.HSpacer(),
                    no_button,
                    yes_button,
                    pn.HSpacer()
                )
            )
        )