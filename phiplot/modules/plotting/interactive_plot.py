from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Callable, Any
from bokeh.themes import built_in_themes
import colorcet as cc
import pandas as pd
import param
import holoviews as hv
from jinja2 import Environment, FileSystemLoader
from .highlighter import Highlighter
from .line_segments import LineSegments
from .tracked_point_scatter import TrackedPointScatter
from phiplot.modules.ui.styling.styling import Styling
from phiplot.modules.ui.utils import *
from .debouncer import DebouncedCallback

if TYPE_CHECKING:
    from phiplot.modules.ui.views.embedding_view import EmbeddingView

# Load HTML templates
env = Environment(loader=FileSystemLoader("phiplot/assets/templates"))

logger = logging.getLogger(__name__)


class InteractivePlot:
    """
    Manages the handling of the Bokeh plot including interactivity with the embedding points.

    Acts as an abstraction layer between the user interface and Bokeh, providing methods to build
    and combine plot elements (e.g. point scatter, line segments and highlighters), render the
    plot and set the necessary references between plotting components.

    Args:
        app (App): The main application instance.
    """

    def __init__(self, parent_view: EmbeddingView):
        self._parent_view = parent_view
        self._embedding_handler = parent_view.embedding_handler
        self._embedding_data_handler = parent_view.embedding_data_handler

        self._pane = parent_view.plot_pane

        self._theme = None
        self._base_plot = self._build_base_plot()
        self._plot = self._base_plot

        self._styling = Styling()
        self._colors = self._styling.default_plot_colors
        self._color_feature = None
        self._color_use_log_scale = False

        self.is_initialized = False

        self._components = None
        self._plot_elements = []
        self._state = None

        self._molecule_tooltips_temp = env.get_template("molecule_tooltips.html")

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @property
    def theme(self) -> str:
        return self._theme

    @theme.setter
    def theme(self, new_theme: str) -> None:
        if new_theme.lower() in built_in_themes:
            self._theme = built_in_themes[new_theme]
        else:
            logger.error("Invalid theme provided. Defaulting to `light_minimal`.")
            self._theme = built_in_themes["light_minimal"]

    def get_state(self):
        return
    
    def set_state(self, state):
        return

    def set_palette(self, palette: str) -> None:
        """
        Set the color palette for the plot.

        Args:
            palette (str): The name of a valid `colorcet` color palette
        """

        if palette in list(cc.palette.keys()):
            self._colors["palette"] = palette
        else:
            palette = self._styling.default_plot_colors["palette"]
            log_process(
                ProcessResult(
                    False,
                    f"Invalid color palette provided. Defaulting to {palette}.",
                    True,
                    "error",
                )
            )
            self._colors["palette"] = palette

        if self.is_initialized:
            self.build()

    def update_coloring(self, feature: str, use_log_scale: bool = False) -> bool:
        """
        Set a new feature to color points by.

        Args:
            feature (str): The name of the feature to color by.
            use_log_scale (bool): Optional. If True, use logarithmic color-scale, otherwise use
                linear color-scale. Defaults to False.

        Returns:
            bool: True if log color-scale was used succesfully, False otherwise.
        """

        if feature in self._embedding_data_handler.columns:
            self._color_feature = feature
            self._color_use_log_scale = use_log_scale
        else:
            log_process(
                ProcessResult(
                    False,
                    f"Could not color by {feature} as the column is missing from the data.",
                    True,
                    "error",
                )
            )

        if self.is_initialized:
            self.build()
            return self._components["main_scatter"].color_use_log_scale
        else:
            return use_log_scale

    def highlight(self, search_index: str) -> None:
        """
        Highlight a point within in the plot.

        Args:
            search_index (str): The index of the point to highlight.
        """

        if self.is_initialized:
            pos_index = self._embedding_data_handler.search_by_index(search_index)
            if pos_index is not None:
                x = self._state["embedding"]["x"][pos_index]
                y = self._state["embedding"]["y"][pos_index]
                logger.info(f"Point {search_index} is located at ({x:.3f}, {y:.3f})")
                self._components["highlighter"].highlight(x, y)

    def apply_filters(self) -> None:
        """
        Fade the points within the plot tha have been filtered out.
        """

        if self.is_initialized:
            indices = self._embedding_data_handler.filtered_indices
            self._components["main_scatter"].fade_filtered(indices)

    def build(self) -> ProcessResult:
        """
        Build and combine the plot elements, set references, and render.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        try:
            self._update_state()
            self._clear_references()

            self._components = self._build_elements()

            self.is_initialized = False
            self._plot_elements = [self._build_base_plot()] + [
                component.get_object() for component in self._components.values()
            ]
            self.is_initialized = True

            self.render()
            self._set_references()
            self._init_main_scatter()
            self._set_watchers(DebouncedCallback(self.update, interval=0.25))
            self.update()

            res = ProcessResult(
                True, "The plot has been built succesfully!", False, "debug"
            )
        except Exception as e:
            logger.exception(f"Error in building the plot:")
            res = ProcessResult(False, "Could not build the plot", True, "error")

        log_process(res)
        return res

    def update(self) -> None:
        """
        Update the plot to reflect new point scatter positions, updated coloring or
        added plot elements.
        """

        if not self._embedding_handler:
            log_process(
                ProcessResult(
                    False,
                    "Cannot update the plot as no embedding handler has been provided.",
                    False,
                    "error",
                )
            )
        else:
            with toggle_spinner(self._parent_view._widgets["main_refresh_indicator"]):
                self._update_state()

                indices = self._state["indices"]
                embedding = self._state["embedding"]

                # update constraints
                self._components["main_scatter"].update_line_color(
                    indices["control"], indices["added"]
                )
                self._components["must_link_segments"].update_links(
                    self._state["must_links"]
                )
                self._components["cannot_link_segments"].update_links(
                    self._state["cannot_links"]
                )

                # update coordinates
                x_new, y_new = embedding["x"], embedding["y"]
                self._components["main_scatter"].update_coords(x_new, y_new)
                self._components["must_link_segments"].update_coords(x_new, y_new)
                self._components["cannot_link_segments"].update_coords(x_new, y_new)

                # sync displays unless dragging
                if not self._components["main_scatter"].constraint_tracker.dragging:
                    self._sync_ui_displays()
                    self._embedding_handler.update_existing = False

    def render(self) -> None:
        """
        Render the plot within the plot pane in the user interface.
        Uses Bokeh as the backend.
        """

        if not self.is_initialized:
            self._pane.object = hv.render(
                self._base_plot, backend="bokeh", theme=self.theme
            )
            return

        combined = hv.Overlay(self._plot_elements).collate()
        self._plot = hv.render(combined, backend="bokeh", theme=self.theme)
        self._pane.object = self._plot

    def _build_base_plot(self) -> hv.Points:
        """
        Build the minimal base plot onto which the other
        plot elements will be combined into.

        Returns:
            hv.Points: The constructed base plot.
        """

        return hv.Points(pd.DataFrame(columns=["x", "y"]), kdims=["x", "y"]).opts(
            show_grid=True, responsive=True
        )

    def _init_main_scatter(self) -> None:
        """
        Initialize the scatter-plot element for plotting points.
        """

        self._components["main_scatter"].update_line_color(
            self._state["indices"]["control"], self._state["indices"]["added"]
        )
        self._components["main_scatter"].fade_filtered(
            self._state["indices"]["filtered"]
        )
        self._components["main_scatter"].init_tracker(
            self._embedding_data_handler, self._embedding_handler
        )

    def _sync_ui_displays(self) -> None:
        """
        Synchronize the user interface displays for control-points and
        link constraints to reflect the current state of the plot.
        """

        formatted_control = self._embedding_data_handler.format_control_points()
        formatted_must_links = self._embedding_data_handler.format_links(
            self._state["must_links"]
        )
        formatted_cannot_links = self._embedding_data_handler.format_links(
            self._state["cannot_links"]
        )

        self._parent_view.sync_displays(
            formatted_control, formatted_must_links, formatted_cannot_links
        )

    def _watch(self, attr, update_func: Callable) -> param.parameterized.Watcher:
        """
        Add a watcher to a `param` attribute.

        Args:
            attr (str): Name of the attribute to watch.
            update_func (Callable): The update function to invoke when the attribute changes.

        Returns:
            param.parameterized.Watcher: The watcher object for the attribute
        """

        return self._embedding_handler.param.watch(lambda *_: update_func(), attr)

    def _set_watchers(self, update_func: Callable) -> None:
        """
        Set the watchers for the constraint attributes within the `param.Parameterized`
        embedding handler and unwatch the previous watchers.

        Args:
            update_func: The update function to invoke when one of the constraints change.
        """

        if hasattr(self, "control_watcher"):
            self._embedding_handler.param.unwatch(self._control_watcher)
        if hasattr(self, "must_link_watcher"):
            self._embedding_handler.param.unwatch(self._must_link_watcher)
        if hasattr(self, "cannot_link_watcher"):
            self._embedding_handler.param.unwatch(self._cannot_link_watcher)

        self._control_watcher = self._watch("control_points", update_func)
        self._must_link_watcher = self._watch("must_links", update_func)
        self._cannot_link_watcher = self._watch("cannot_links", update_func)

    def _set_tooltips(self) -> None:
        """
        Set the tool-tip box to show when hovering over points in the plot.
        """

        # Use the 'self._molecule_tooltips_temp' template for molecular data.
        cols = self._embedding_data_handler.tooltip_columns

        properties = {}

        index_col = self._embedding_data_handler.index_column
        index_styles = "font-size: 14px; color: #000080; font-weight: bold; margin-bottom: 3px;"

        properties["@" + index_col] = {
            "title": index_col.replace("_", " ").title(),
            "styles": index_styles,
        }

        if index_col in cols:
            cols.remove(index_col)

        smiles_like_cols = self._embedding_data_handler.smiles_like_cols
        for col in smiles_like_cols:
            smiles_styles = "font-size: 12p; color: #228B22; font-family: Monospace; margin-bottom: 2px;"

            properties["@" + col] = {
                "title": col.replace("_", " ").title(),
                "styles": smiles_styles,
            }

            if col in cols:
                cols.remove(col)

        base_styles = "font-size: 12px; margin-bottom: 2px;"

        for col in cols:
            properties["@" + col] = {
                "title": col.replace("_", " ").title(),
                "styles": base_styles,
            }

        self.tooltips = self._molecule_tooltips_temp.render(properties=properties)
        # Otherwise just list all the feature columns.

    def _build_elements(self) -> None:
        """
        Build all the plot element classes.
        """

        self.plot_df = self._embedding_data_handler.plot_df
        self._set_tooltips()

        return dict(
            main_scatter=TrackedPointScatter(
                label="Original",
                vdims=self._embedding_data_handler.vdims,
                idx_col=self._embedding_data_handler.index_column,
                hover_tooltips=self.tooltips,
                data=self.plot_df,
                colors=self._colors,
                color_feature=self._color_feature,
                color_use_log_scale=self._color_use_log_scale
            ),
            must_link_segments=LineSegments(
                label="Must-link",
                data=self.plot_df[["x", "y"]],
                links=[],
                color=self._colors["must_link"],
                line_dash="solid",
            ),
            cannot_link_segments=LineSegments(
                label="Cannot-link",
                data=self.plot_df[["x", "y"]],
                links=[],
                color=self._colors["cannot_link"],
                line_dash="dashed",
            ),
            highlighter=Highlighter(radius=25, color="Fuchsia"),
        )

    def _set_references(self) -> None:
        """
        Set the references to the right data sources within the rendered plot
        for the scatter plot and the line segments.
        """

        for r in self._plot.renderers:
            if r.name == "Original":
                self._components["main_scatter"].set_references(
                    r.data_source, self._plot
                )
            elif r.name == "Must-link":
                self._components["must_link_segments"].set_references(r.data_source)
            elif r.name == "Cannot-link":
                self._components["cannot_link_segments"].set_references(r.data_source)

    def _clear_references(self) -> None:
        """
        Clear the references to the old data sources in the previously rendered plot.
        """

        for component in getattr(self, "components", {}).values():
            component.clear_references()

    def _update_state(self):
        """
        Update the sate of the object including:
            -"indices" (dict): All the named inidices including control point indices
            -"must_links" (list[tuple]): The must link indices.
            -"cannot_links" (list[tuple]): The cannot link indices.
            -"embedding" (dict): The embedding coordinates of the points.
        """

        self._state = dict(
            indices = dict(
                control=self._embedding_handler.cp_indices,
                added=[],
                filtered=self._embedding_data_handler.filtered_indices
            ),
            must_links=self._embedding_handler.must_links,
            cannot_links=self._embedding_handler.cannot_links,
            embedding=self._embedding_handler.embedding,
        )

    def set_color(self, element: str, color: str) -> None:
        """
        Set a new color for given element.

        Args:
            element (str): The name of the element to set the color for.
            color (str): The new color for the element.
        """

        if element in self._colors.keys():
            self._colors[element] = color
            
        if self.is_initialized:
            if element in ["control_point", "added_point"]:
                self.update()
            else:
                self.build()