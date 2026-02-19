from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Callable
import panel as pn

logger = logging.getLogger(__name__)


class DisplayPanel:
    """
    Wrapper for creating a standardized display panel that lists items
    and allows interactive removal or selection (MultiChoice widget).

    Args:
        name (str): Name of the display panel.
    """

    def __init__(self, name: str):
        self.name = name
        self.display = pn.widgets.MultiChoice(
            value=[], options=[], sizing_mode="stretch_width"
        )

    def __panel__(self) -> pn.widgets.MultiChoice:
        """
        Get the underlying MultiChoice widget.

        Returns:
            pn.widgets.MultiChoice: Configured multi-choice display.
        """

        return self.display

    @property
    def value(self):
        return self.display.value
    
    @value.setter
    def value(self, contents: list[str] | None) -> None:
        """
        Update the display with new contents. Skip if no new contents
        are provided.

        Args:
            contents (list[str] | None): List of values to display and preselect.
        """

        if contents is None:
            return

        self.display.options = contents
        self.display.value = contents

    def set_watcher(self, action: Callable) -> None:
        """
        Register a watcher to respond to changes in selection.

        Args:
            action (Callable): Function to call when the value changes.
        """

        self.display.param.watch(action, "value")