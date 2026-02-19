from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import panel as pn
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.views.data_summary_view import DataSummaryView

logger = logging.getLogger(__name__)

class DataSummaryAppearanceMenu(BaseMenu):
    def __init__(self, parent_view: DataSummaryView):
        super().__init__(
            parent_view=parent_view,
            name="Appearance",
            icon="edit"
        )

        self._construct_widgets()
        
        self._floating_panels = dict(
           color_settings = self._build_color_settings_panel(),
        )

        self.menu_items = [
            ("Color Settings", "color_settings"),
            ("Toggle Legends", "toggle_legends"),
        ]

        self.callbacks = dict(
            color_settings = lambda: self._floating_panels["color_settings"].open(),
            toggle_legends = lambda: self._parent_view.toggle_legends()
        )

    def _construct_widgets(self) -> None:
        self._widgets["box_plot_iqr_fill_color_selector"] = pn.widgets.ColorPicker(
            name="IQR Fill",
            value=self._styling.plot_blue
        )

        self._widgets["box_plot_iqr_line_color_selector"] = pn.widgets.ColorPicker(
            name="IQR Line",
            value=self._styling.neutral_gray
        )

        self._widgets["box_plot_whisker_color_selector"] = pn.widgets.ColorPicker(
            name="Whiskers",
            value=self._styling.neutral_gray
        )

        self._widgets["box_plot_median_color_selector"] = pn.widgets.ColorPicker(
            name="Median",
            value="lightgreen"
        )

        self._widgets["box_plot_std_color_selector"] = pn.widgets.ColorPicker(
            name="STD",
            value="red"
        )

        self._widgets["histogram_fill_color_selector"] = pn.widgets.ColorPicker(
            name="Bins Fill",
            value=self._styling.plot_blue
        )

        self._widgets["histogram_line_color_selector"] = pn.widgets.ColorPicker(
            name="Bins Line",
            value=self._styling.neutral_gray
        )

        self._widgets["kde_color_selector"] = pn.widgets.ColorPicker(
            name="KDE Curve",
            value="red"
        )

        self._widgets["plot_color_scheme_selector"] = pn.widgets.Select(
            name = "Plot Color Scheme",
            options = ["Light", "Dark"],
            value = "Light"
        )

        self._widgets["apply_color_settings_button"] = pn.widgets.Button(
            name = "Apply Color Settings", **self._styling.default_button_style
        )
        self._widgets["apply_color_settings_button"].on_click(
            lambda event: self._parent_view.update_colors()
        )

    def get_color_settings(self) -> dict[str, dict[str, str]]:
        return dict(
            plot = dict(
                color_scheme = self._widgets["plot_color_scheme_selector"].value.lower()
            ),
            histogram = dict(
                fill = self._widgets["histogram_fill_color_selector"].value,
                line = self._widgets["histogram_line_color_selector"].value,
                kde = self._widgets["kde_color_selector"].value
            ),
            box_plot = dict(
                iqr_fill = self._widgets["box_plot_iqr_fill_color_selector"].value,
                iqr_line = self._widgets["box_plot_iqr_line_color_selector"].value,
                whiskers = self._widgets["box_plot_whisker_color_selector"].value,
                median = self._widgets["box_plot_median_color_selector"].value,
                std = self._widgets["box_plot_std_color_selector"].value
            )
        )

    def _build_color_settings_panel(self) -> WindowPanel:
        window = WindowPanel("Color Settings", self._window_destination)
        window.contents = [
            pn.pane.Markdown("### Box Plot"),
            pn.Row(
                self.widgets["box_plot_iqr_fill_color_selector"],
                pn.layout.Spacer(width=25),
                self.widgets["box_plot_iqr_line_color_selector"],
                pn.layout.Spacer(width=25),
                self.widgets["box_plot_whisker_color_selector"],
                pn.layout.Spacer(width=25),
                self.widgets["box_plot_median_color_selector"],
                pn.layout.Spacer(width=25),
                self.widgets["box_plot_std_color_selector"],
                sizing_mode="stretch_width"
            ),
            pn.pane.Markdown("### Histograms"),
            pn.Row(
                self.widgets["histogram_fill_color_selector"],
                pn.layout.Spacer(width=25),
                self.widgets["histogram_line_color_selector"],
                pn.layout.Spacer(width=25),
                self.widgets["kde_color_selector"],
                sizing_mode="stretch_width"
            ),
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["apply_color_settings_button"]
        ]
        return window