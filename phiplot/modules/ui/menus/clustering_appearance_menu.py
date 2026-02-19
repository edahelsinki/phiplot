from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import panel as pn
import colorcet as cc
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.views.clustering_view import ClusteringView

logger = logging.getLogger(__name__)

class ClusteringAppearanceMenu(BaseMenu):
    def __init__(self, parent_view: ClusteringView):
        super().__init__(
            parent_view=parent_view,
            name="Appearance",
            icon="edit"
        )

        # Select only Glasbey categorical palettes
        self._color_palette_options = {
            name: palette for name, palette in cc.palette.items() if "glasbey" in name
        }
        self._color_palette = "glasbey_cool"

        self._floating_panels = dict(
           color_settings = self._build_color_settings_panel(),
        )

        self.menu_items = [
            ("Color Settings", "color_settings"),
            ("Toggle Legend", "toggle_legends"),
        ]

        self.callbacks = dict(
            color_settings = lambda: self._floating_panels["color_settings"].open(),
            toggle_legends = lambda: self._parent_view.toggle_legend()
        )

    @property
    def color_palette(self):
        return cc.palette[self._color_palette]
    
    def _update_color_palette(self) -> None:
        self._color_palette = self._widgets["color_palette_selector"].value_name
        self._parent_view.update_color_palette()

    def _build_color_settings_panel(self) -> WindowPanel:
        self._widgets["color_palette_selector"] = pn.widgets.ColorMap(
            name="Plot Color Palette",
            options=self._color_palette_options,
            ncols=1,
            swatch_width=80,
            swatch_height=20,
            value_name=self._color_palette,
            sizing_mode="stretch_width"
        )

        self._widgets["apply_color_settings_button"] = pn.widgets.Button(
            name = "Apply Color Settings", **self._styling.default_button_style
        )
        self._widgets["apply_color_settings_button"].on_click(
            lambda event: self._update_color_palette()
        )

        window = WindowPanel("Color Settings", self._window_destination)
        window.contents = [
            self._widgets["color_palette_selector"],
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["apply_color_settings_button"]
        ]
        return window
    
