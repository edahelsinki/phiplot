from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging
from typing import TYPE_CHECKING, Any

import panel as pn

if TYPE_CHECKING:
    from phiplot.modules.ui.web_ui import WebUI
    from phiplot.modules.data.handlers.data_handler import DataHandler
    from phiplot.modules.embedding.embedding_handler import EmbeddingHandler

logger = logging.getLogger(__name__)

class BaseView(ABC):
    def __init__(self, ui: WebUI):
        self._ui = ui

        self.styling = ui.styling
        self.data_handler: DataHandler = ui.data_handler
        self.embedding_handler: EmbeddingHandler = ui.embedding_handler
        self.window_destination = ui.window_destination

        self._base_menu = ui.base_menu
        self._base_info = ui.base_info
        self._session_id = ui.session_id

        self._menus = {}
        self._center_column_items = []
        self._left_column_items = []
        self._right_column_items = []
        self._widgets = {}

        self._plots = {}
        self._plot_color_scheme = "light"

        self._title_row = self._build_title_row()
        
    @property
    def view(self) -> list:
        return [
            *list(self._ui.modals.values()),
            self.window_destination,
            self._ui.select_view_row,
            self.menu_row,
            pn.Row(
                self.left_column,
                self.center_column,
                self.right_column,
                sizing_mode="stretch_both"
            ),
            self._ui.footer
        ]

    @property
    def state(self) -> dict[str, Any]:
        states = dict()
        for name, widget in self._widgets.items():
            states[name] = widget.value
        return states
    
    @state.setter
    def state(self, new_state: dict[str: Any]) -> None:
        for name, val in new_state.items():
            if name in self._widgets:
                self._widgets[name].value = val

    @property
    def menu_row(self) -> pn.Row:
        return pn.Row(
            *(self._base_menu + list(self._menus.values())),
            sizing_mode="stretch_width",
            height=self.styling.top_menu_height,
            styles=dict(
                background=self.styling.neutral_gray,
                padding_left="7px",
                padding_right="7px"
            )
        )
    
    @menu_row.setter
    def menu_row(self, items: list) -> None:
        if all([isinstance(item, pn.widgets.MenuButton) for item in items]):
            self._menu_items = items

    @property
    def center_column(self) -> pn.Column:
        return pn.Column(
            self._title_row,
            *self._center_column_items,
            sizing_mode="fixed",
            styles=dict(
                width=self.styling.middle_column_relative_width,
                height="100%",
                border_top=self.styling.border_style,
                border_bottom=self.styling.border_style,
            )
        )
    
    @center_column.setter
    def center_column(self, items: list) -> None:
        self._center_column_items = items

    @property
    def left_column(self) -> pn.Column:
        return pn.Column(
            pn.Accordion(
                *self._left_column_items,
                active=list(range(len(self._left_column_items))),
                styles=dict(width=self.styling.accordion_relative_width),
            ),
            sizing_mode="fixed",
            styles=dict(
                width=self.styling.side_column_relative_width,
                height="100%",
                border=self.styling.border_style,
                overflow="auto"
            )
        )
    
    @left_column.setter
    def left_column(self, items: list) -> None:
        if self._valid_accordion_items(items):
            self._left_column_items = items

    @property
    def right_column(self) -> pn.Column:
        return pn.Column(
            pn.Accordion(
                *(self._base_info + self._right_column_items),
                active=list(range(len(self._base_info) + len(self._right_column_items))),
                styles=dict(width=self.styling.accordion_relative_width),
            ),
            sizing_mode="fixed",
            styles=dict(
                width=self.styling.side_column_relative_width,
                height="100%",
                border=self.styling.border_style,
                overflow="auto"
            )
        )
    
    @right_column.setter
    def right_column(self, contents: list) -> None:
        self._right_column_items = contents

    @property
    def title(self) -> str:
        return self._title_pane.objects[0].object
    
    @title.setter
    def title(self, new_title):
        self._title_pane.objects = [
            pn.VSpacer(),
            pn.pane.Markdown(f"## {new_title}"),
            pn.VSpacer()
        ]

    def _build_title_row(self) -> pn.Row:
        self._title_pane = pn.Column()

        self._widgets["main_refresh_indicator"] = pn.indicators.LoadingSpinner(
            value=False, size=40, name="", color="primary"
        )

        self._widgets["search_button"] = pn.widgets.ButtonIcon(
            icon="search",
            active_icon="loader",
            toggle_duration=300,
            size="2.5em",
            margin=(5, 0, 5, 5),
        )
        self._widgets["search_button"].on_click(self._search_by_index)

        self._widgets["search_index_str"] = pn.widgets.TextInput(
            placeholder="Search Index", margin=(5, 5, 5, 0), sizing_mode="stretch_width"
        )
        self._widgets["search_index_str"].param.watch(self._search_by_index, "enter_pressed")

        self._search_section = pn.Row(
            self._widgets["search_button"],
            self._widgets["search_index_str"],
            sizing_mode="fixed",
            align="end",
            styles=dict(width="25%"),
        )

        return pn.Row(
            self._widgets["main_refresh_indicator"],
            self._title_pane,
            pn.HSpacer(),
            self._search_section,
            height=50,
            sizing_mode = "stretch_width"
        )

    def _valid_accordion_items(self, items: list) -> bool:
        return all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) 
            for item in items
        )
    
    @abstractmethod
    def _search_by_index(self, event=None):
        return