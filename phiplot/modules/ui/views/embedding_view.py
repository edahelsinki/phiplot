from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import holoviews as hv
import panel as pn
from jinja2 import Environment, FileSystemLoader
from .base_view import BaseView
from phiplot.modules.ui.widgets import *
from phiplot.modules.ui.menus import *
from phiplot.modules.plotting.interactive_plot import InteractivePlot
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.web_ui import WebUI

logger = logging.getLogger(__name__)

# Load HTML templates
env = Environment(loader=FileSystemLoader("phiplot/assets/templates"))


class EmbeddingView(BaseView):
    def __init__(self, ui: WebUI):
        super().__init__(ui)
        self.title = "Embedding"

        self.embedding_data_handler = self.data_handler.embedding_data_handler

        self.plot_pane = pn.pane.Bokeh(sizing_mode="stretch_both")
        self.plot = InteractivePlot(self)
        self.plot.render()

        self._menus = dict(
            embedding = EmbeddingMenu(self),
            constraints = EmbeddingConstraintsMenu(self),
            appearance = EmbeddingAppearanceMenu(self)
        )

        self._display_panes = dict(
            metrics = self._create_metrics_section(),
            control_points = DisplayPanel("Control Points"),
            must_links = DisplayPanel("Must-Links"),
            cannot_links = DisplayPanel("Cannot-Links")
        )

        self._attach_display_actions(
            cp_action=self._menus["constraints"].remove_control_point,
            ml_action=self._menus["constraints"].remove_must_link,
            cl_action=self._menus["constraints"].remove_cannot_link
        )

        self._create_link_strength_adjuster()

        self.center_column = [self.plot_pane]

        self.left_column = [
            ("Embedding Metrics", self._display_panes["metrics"]),
            ("Link Strength", self._widgets["link_strength_slider"]),
            ("Control Points", self._display_panes["control_points"]),
            ("Must-Links", self._display_panes["must_links"]),
            ("Cannot-Links", self._display_panes["cannot_links"])
        ]

        self.right_column = []

    @property
    def control_points(self) -> list[str]:
        return self._display_panes["control_points"].value
    
    @control_points.setter
    def control_points(self, points: list[str]) -> None:
        self._display_panes["control_points"].value = points

    @property
    def must_links(self) -> list[str]:
        return self._display_panes["must_links"].value
    
    @must_links.setter
    def must_links(self, links: list[str]) -> None:
        self._display_panes["must_links"].value = links
    
    @property
    def cannot_links(self) -> list[str]:
        return self._display_panes["cannot_links"].value
    
    @cannot_links.setter
    def cannot_links(self, links: list[str]) -> None:
        self._display_panes["cannot_links"].value = links

    def update_available_features(self):
        self._menus["appearance"].update_available_features()

    def embed(self, recompute: bool = True) -> None:
        with toggle_spinner(self._widgets["main_refresh_indicator"]):
            algo = self._menus["embedding"].algo
            params = self._menus["embedding"].params

            init_success = self.embedding_handler.init_embedding(algo, params)
            self.embedding_data_handler.create_plot_df()
            plot_success = self.plot.build()

            if init_success and plot_success:
                pn.state.notifications.success("Embedding ready!")
            elif not init_success:
                pn.state.notifications.error("Error in initializing the model.")
            elif not plot_success:
                pn.state.notifications.error("Error in building the embedding.")

    def recompute_kernel_heuristics(self) -> None:
        self._menus["embedding"].recompute_kernel_heuristics()

    def flush_kernel_mpds(self) -> None:
        self._menus["embedding"].flush_mpds()

    def toggle_legend(self) -> None:
        """
        Show/hide the plot legend
        """

        legend = self.plot_pane.object.legend[0]
        current = legend.visible
        legend.visible = not current

    def sync_displays(self, control: list, must_link: list, cannot_link: list) -> None:
        """
        Synchronize the UI displays with updated constraint data,
        refresh embedding metrics and constraint displays.

        Args:
            control (list): Updated control point constraints.
            must_link (list): Updated must-link constraints.
            cannot_link (list): Updated cannot-link constraints.
        """

        self._display_panes["control_points"].value = control
        self._display_panes["must_links"].value = must_link
        self._display_panes["cannot_links"].value = cannot_link
        self.update_metrics_display()

    def update_point_colors(self, event=None):
        return
    
    def update_color_palette(self, event=None):
        return

    def update_metrics_display(self, event=None):
        """
        Update the embedding metrics display.

        Renders metrics in HTML with with threshold-based color coding.
        """

        if not self.embedding_handler.model_initialized:
            pn.state.notifications.warning("The embedding has not been initialized...")
            return

        with toggle_spinner(self._widgets["computing_metrics_spinner"]):
            template = env.get_template("metrics_display.html")

            embedding_metrics = self.embedding_handler.get_embedding_metrics(
                self._widgets["metrics_distance_selector"].value
            )

            threshold_color_map = {
                "Trustworthiness": [0.7, 0.8, 0.95, 1],
                "KNN Preservation": [0.7, 0.8, 0.9, 1],
                "Shepard Correlation": [0.5, 0.7, 0.9, 1],
                "Stress": [0.2, 0.1, 0.05, 0],
            }
            palette = ["#440154", "#3b528b", "#21908d", "#5dc963"]
            palette_labels = ["Poor", "Fair", "Good", "Excellent"]

            metrics = []
            for metric, value in embedding_metrics.items():
                thresholds = threshold_color_map[metric]
                reverse = thresholds[0] > thresholds[-1]
                try:
                    if not reverse:
                        idx = next(i for i, x in enumerate(thresholds) if value < x)
                    else:
                        idx = next(i for i, x in enumerate(thresholds) if x < value)
                    color = palette[idx]
                except StopIteration:
                    color = "white"

                font_color = "white" if idx < 3 else "black"
                metrics.append((metric, value, color, font_color))

            legend = list(zip(palette, palette_labels))

            distance_measure = self._widgets["metrics_distance_selector"].value
            html = template.render(metrics=metrics, legend=legend, distance_measure=distance_measure)
            self._widgets["metrics_display"].object = html
    
    def _attach_display_actions(self, cp_action, ml_action, cl_action) -> None:
        self._display_panes["control_points"].set_watcher(action=cp_action)
        self._display_panes["must_links"].set_watcher(action=ml_action)
        self._display_panes["cannot_links"].set_watcher(action=cl_action)

    def _create_metrics_section(self) -> pn.Column:
        """
        Create and configure the metrics evaluation section.
        """

        self._widgets["recompute_metrics_button"] = pn.widgets.Button(
            name="Recompute", **self.styling.default_button_style
        )

        self._widgets["computing_metrics_spinner"] = pn.indicators.LoadingSpinner(
            **self.styling.default_spinner_style
        )

        self._widgets["metrics_distance_selector"] = pn.widgets.Select(
            name="Distance Measure",
            options=sorted(["cityblock", "cosine", "euclidean", "sqeuclidean"]),
            value="cosine",
            sizing_mode="stretch_width",
        )

        self._widgets["recompute_metrics_button"].on_click(self.update_metrics_display)

        self._widgets["metrics_display"] = pn.pane.HTML(sizing_mode="stretch_width")

        return pn.Column(
            self._widgets["metrics_display"],
            self._widgets["metrics_distance_selector"],
            pn.Row(
                self._widgets["recompute_metrics_button"],
                self._widgets["computing_metrics_spinner"]
            )
        )

    def _search_by_index(self, event=None):
        try:
            search_index = str(self._widgets["search_index_str"].value)
        except ValueError:
            logger.warning("Invalid search index")
        self.plot.highlight(search_index)

    def _create_link_strength_adjuster(self) -> None:
        """
        Create a slider for adjusting link strength in interactive embeddings.
        """

        def adjust_link_strength(event=None):
            self.embedding_handler.link_strengh_multiplier = (
                self._widgets["link_strength_slider"].value
            )
            self.plot.update()

        self._widgets["link_strength_slider"] = pn.widgets.FloatSlider(
            name="Link Strength Multiplier",
            start=0,
            end=2,
            value=1,
            step=0.1,
            sizing_mode="stretch_width",
        )
        self._widgets["link_strength_slider"].param.watch(adjust_link_strength, "value")
        