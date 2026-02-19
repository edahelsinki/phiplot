from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import panel as pn
import colorcet as cc
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.views.embedding_view import EmbeddingView

logger = logging.getLogger(__name__)

class EmbeddingAppearanceMenu(BaseMenu):
    def __init__(self, parent_view: EmbeddingView):
        super().__init__(
            parent_view=parent_view,
            name="Appearance",
            icon="edit"
        )

        self._embedding_data_handler = parent_view.embedding_data_handler

        self._color_palette = "CET_D9"

        self._floating_panels = dict(
            colorby_feature = self._build_colorby_feature_panel(),
            color_settings = self._build_color_settings_panel(),
        )

        self.menu_items = [
            ("Color by Feature", "colorby_feature"),
            ("Color Settings", "color_settings"),
            ("Toggle Legend", "toggle_legends"),
        ]

        self.callbacks = dict(
            colorby_feature = lambda: self._floating_panels["colorby_feature"].open(),
            color_settings = lambda: self._floating_panels["color_settings"].open(),
            toggle_legends = lambda: self._parent_view.toggle_legend()
        )

    @property
    def color_palette(self):
        return cc.palette[self._color_palette]
    
    def update_available_features(self) -> None:
        self._widgets["colorby_feature_selector"].options = self._embedding_data_handler.colorby_columns
    
    def _build_colorby_feature_panel(self) -> WindowPanel:
        """
        Create the panel for coloring datapoints by feature.

        Returns:
            WindowPanel: A floating panel with selectors for
            feature-based coloring and log scale toggle.
        """

        window = WindowPanel("Color by Feature", self._window_destination)

        self._widgets["update_coloring_spinner"] = pn.indicators.LoadingSpinner(
            **self._styling.default_spinner_style
        )

        self._widgets["colorby_feature_selector"] = pn.widgets.Select(
            name="Color Datapoints by",
            options=[None] + self._embedding_data_handler.columns,
            value=None,
            sizing_mode="stretch_width",
        )

        self._widgets["color_log_scale_toggle"] = pn.widgets.Checkbox(name="Use log-scale")

        self._widgets["apply_colorby_button"] = pn.widgets.Button(
            name = "Apply Color by Feature", **self._styling.default_button_style
        )
        self._widgets["apply_colorby_button"].on_click(self._on_colorby_feature_selection)

        window.contents = [
            self._widgets["colorby_feature_selector"],
            self._widgets["color_log_scale_toggle"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["apply_colorby_button"],
                self._widgets["update_coloring_spinner"]
            )
        ]
        return window

    def _build_color_settings_panel(self) -> WindowPanel:
        self._widgets["color_palette_selector"] = pn.widgets.ColorMap(
            name="Plot Color Palette",
            options=cc.palette,
            ncols=1,
            swatch_width=80,
            swatch_height=20,
            value_name=self._color_palette,
            sizing_mode="stretch_width"
        )

        self._widgets["apply_color_settings_button"] = pn.widgets.Button(
            name = "Apply Color Settings", **self._styling.default_button_style
        )
        self._widgets["apply_color_settings_button"].on_click(self._on_color_palette_selection)

        window = WindowPanel("Color Settings", self._window_destination)
        window.contents = [
            self._widgets["color_palette_selector"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["apply_color_settings_button"],
                self._widgets["update_coloring_spinner"]
            )
        ]
        return window
    
    def _on_color_palette_selection(self, event=None) -> None:
        self._color_palette = self._widgets["color_palette_selector"].value_name
        with toggle_spinner(self._widgets["update_coloring_spinner"]):
            self._parent_view.plot.set_palette(self._color_palette)

    def _on_colorby_feature_selection(self, event=None) -> None:
        feature = self.widgets["colorby_feature_selector"].value
        with toggle_spinner(self._widgets["update_coloring_spinner"]):
            use_log_scale = self.widgets["color_log_scale_toggle"].value
            value_after = self._parent_view.plot.update_coloring(
                feature, use_log_scale
            )
            self.widgets["color_log_scale_toggle"].value = value_after
                