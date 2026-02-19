from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any
import panel as pn

if TYPE_CHECKING:
    from phiplot.modules.ui.web_ui import WebUI
    from phiplot.modules.ui.views import *

logger = logging.getLogger(__name__)

class BaseMenu:
    """
    Base panel class for panels launched via menu.

    Provides shared logic for:
    - Managing a menu item
    - Assigning a destination container
    - Handling callbacks and action mapping

    Args:
        app (App): Reference to the main application instance.
    """

    def __init__(
            self,
            parent_view: WebUI | DataSummaryView | ClusteringView | EmbeddingView, 
            name: str, 
            icon: str
        ):
        self._parent_view = parent_view
        self._name = name
        self._icon = icon

        self._window_destination = parent_view.window_destination
        self._data_handler = parent_view.data_handler
        self._embedding_handler = parent_view.embedding_handler
        self._styling = parent_view.styling

        self._menu_items: list = []
        self._callbacks: dict = {}
        self._widgets: dict = {}
        self._menu: pn.widgets.MenuButton | None = None

    def __panel__(self):
        return self.menu
    
    @property
    def state(self) -> dict[str, Any]:
        states = dict()
        for name, widget in self._widgets.items():
            states[name] = widget.value
        return states

    @state.setter
    def state(self, new_state: dict[dict, Any]) -> None:
        for name, val in new_state.items():
            if name in self._widgets:
                self._widgets[name].value = val
    
    @property
    def widgets(self) -> dict[str, Any]:
        return self._widgets
    
    @property
    def menu(self):
        menu = pn.widgets.MenuButton(
            name=self._name,
            icon=self._icon,
            items=self._menu_items,
            margin=(0, 2, 4, 2),
            button_type="default",
            sizing_mode="stretch_both",
        )
        menu.on_click(lambda event: self._menu_click(event.new))

        return menu
    
    @property
    def menu_items(self):
        return self._menu_items

    @menu_items.setter
    def menu_items(self, items):
        self._menu_items = items

    @property
    def callbacks(self):
        return self._callbacks is not None
    
    @callbacks.setter
    def callbacks(self, fn_map):
        self._callbacks = fn_map

    def _menu_click(self, option: str) -> None:
        """
        Dispatch menu events to the appropriate action.

        Args:
            options: Name of the pressed menu option.
        """

        try:
            self._callbacks[option]()
        except KeyError:
            logger.info("Unrecognized action encountered.")