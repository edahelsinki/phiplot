from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any
from bokeh.colors import RGB
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
    LogColorMapper,
    CategoricalColorMapper,
    LogTicker,
    PrintfTickFormatter,
)
from bokeh.plotting import figure
import colorcet as cc
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import streams
from .event_handlers import ConstraintTracker

if TYPE_CHECKING:
    from phiplot.modules.data.handlers.data_handler import DataHandler
    from phiplot.modules.embedding.embedding_handler import EmbeddingHandler

logger = logging.getLogger(__name__)


class TrackedPointScatter:
    """
    Manages the handling of the point scatter object.

    Acts as an abstraction layer between the `hv.Points` object and the InteractivePlot
    providing methods to build and update the `hv.Points` object as well as attaching hooks
    to the underlying Bokeh object to make visual changes.

    Args:
        label (str): Label for the `hv.Points` object (used e.g. in the legend)
        vdims (list[str]): Names of the value dimensions.
        idx_col (str): Name of the index column.
        hover_tooltips (str | list[tuple[str, str]]): Template for the hover tooltips.
        data (pd.DataFrame): All the necessary data for the plotting.
        colors (dict): The coloring settings to use.
        **kwargs: Optional arguments for setting up rest of the styling:
            -"color_feature" (str): Which feature to color by
            -"color_use_log_scale" (bool): If True use logarithmic color scale,
                otherwise use linear color scale.
            -"size" (int): Size of the points in default units.
            -"line_width" (int). The width of the border around the points in default units.
    """

    def __init__(
        self,
        label: str,
        vdims: list[str],
        idx_col: str,
        hover_tooltips: str | list[tuple[str, str]],
        data: pd.DataFrame,
        colors: dict, 
        **kwargs: dict,
    ) -> None:

        self._label = label
        self._vdims = vdims
        self._idx_col = idx_col
        self._hover_tooltips = hover_tooltips
        self._data = data

        self._constraint_tracker = None
        self._source = None
        self._bokeh_plot = None

        self._colors = colors
        self._default_point_size = 25
        self._default_line_width = 5
        self._fade_fraction = 0.1

        self._styles = self._resolve_kwargs(kwargs)

        self._data["fill_alpha"] = [1.0] * len(self._data)
        self._data["line_color"] = [None] * len(self._data)
        self._vdims += ["fill_alpha", "line_color"]

        self._set_object()
        self._set_draw_tool()

    @property
    def color_use_log_scale(self) -> bool:
        return self._color_use_log_scale

    def set_references(self, source: ColumnDataSource, bokeh_plot: figure) -> None:
        """
        Set the references to the right datasource within the rendered
        Bokeh plot and the Bokeh plot instance itself.

        Args:
            source (ColumnDataSource): The datasource for the points within the rendered Bokeh plot.
            bokeh_plot (figure): The rendered Bokeh plot instance.
        """

        self._source = source
        self._bokeh_plot = bokeh_plot

    def clear_references(self) -> None:
        """
        Clear the references to the previous datasource and Bokeh plot.
        """

        self._source = None
        self._bokeh_plot = None

    def init_tracker(
        self, data_handler: DataHandler, embedding_handler: EmbeddingHandler
    ) -> None:
        """
        Initialize the constraint tracker for the scatter.

        Args:
            data_handler (DataHandler): The data handler instance of the main application.
            embedding_handler (EmbeddingHandler): The mbedding handler instance of the main application.
        """

        self.constraint_tracker = ConstraintTracker(
            self._source, self._bokeh_plot, data_handler, embedding_handler
        )

    def get_object(self) -> hv.Points:
        """
        Get the underlying `hv.Points` object.

        Returns:
            hv.Points: The points to render.
        """

        return self._hv_points

    def update_coords(self, x_new: list[float], y_new: list[float]) -> None:
        """
        Update the coordinates in the underlying ColumnDataSource of the points.

        Args:
            x_new (list[float]): The new x-coordinates.
            y_new (list[float]): The new y-coordinates.
        """

        self._source.data["x"][:] = x_new
        self._source.data["y"][:] = y_new
        self._source.trigger("data", self._source.data, self._source.data)

    def update_line_color(self, control_indices: list, added_indices: list) -> None:
        """
        Update the color of the border around the points to reflect their status.

        Args:
            control_indices (list): Indices of the control points.
            added_indices (list): Indices of the points added after the initial embedding.
        """

        control_set, added_set = set(control_indices), set(added_indices)
        self._source.data["line_color"] = [
            (
                self._colors["control_point"]
                if i in control_set
                else self._colors["added_point"] if i in added_set else "transparent"
            )
            for i in range(len(self._data))
        ]

    def fade_filtered(self, filtered_indices: list) -> None:
        """
        Fade the points that have been filtered out.

        Args:
            filtered_indices (list): Indices of the filtered points.
        """

        filtered_set = set(filtered_indices)
        self._source.data["fill_alpha"] = [
            self._fade_fraction if i in filtered_set else 1.0
            for i in range(len(self._data))
        ]

    def apply_point_appearance_mapper(self, plot: Any, element: Any) -> None:
        """
        Create the hook to use for attaching point appearance rules.

        Args:
            plot (Any): The HoloViews plot object that is rendering the plot.
            element (Any): The element being rendered.
        """

        glyph = plot.handles["glyph"]
        plot_handle = plot.state

        glyph.line_color = {"field": "line_color"}
        glyph.fill_alpha = {"field": "fill_alpha"}

        if (
            self._color_feature is not None
            and self._color_feature in self._data.columns
        ):
            series = self._data[self._color_feature]
            
            is_categorical = (
                isinstance(series.dtype, pd.CategoricalDtype) or 
                pd.api.types.is_object_dtype(series) or 
                pd.api.types.is_string_dtype(series)
            )

            if not hasattr(self, "_color_mapper"):
                palette = self._get_hex_palette(self._color_palette)
                
                if is_categorical:
                    factors = series.unique().astype(str).tolist()
                    self._color_mapper = CategoricalColorMapper(
                        palette=palette, 
                        factors=factors
                    )
                    ticker = None
                else:
                    low, high = series.min(), series.max()
                    if self._color_use_log_scale and low > 0:
                        self._color_mapper = LogColorMapper(palette=palette, low=low, high=high)
                        ticker = LogTicker(desired_num_ticks=15)
                    else:
                        self._color_mapper = LinearColorMapper(palette=palette, low=low, high=high)
                        ticker = BasicTicker(desired_num_ticks=15)

            glyph.fill_color = {
                "field": self._color_feature,
                "transform": self._color_mapper,
            }

            if not any(isinstance(renderer, ColorBar) for renderer in plot_handle.renderers):
                if not is_categorical:
                    color_bar = ColorBar(
                        color_mapper=self._color_mapper,
                        label_standoff=12,
                        ticker=ticker,
                        formatter=PrintfTickFormatter(format="%.3g"),
                        location=(0, 0),
                        title=self._color_feature,
                    )
                    plot_handle.add_layout(color_bar, "right")
                else:
                    pass
            return

        glyph.fill_color = self._colors["fill"]

    def _resolve_kwargs(self, kwargs: dict) -> dict[str, int]:
        """
        Resolve the optional keyword arguments to set up the styling for the points.
        Uses predefined defaults for missing arguments.

        Args:
            kwargs (dict): The keyword arguments described in the class-level doc.

        Returns:
            dict: The point size and border width arguments.
        """

        self._color_feature = kwargs.get("color_feature", None)
        self._color_use_log_scale = kwargs.get("color_use_log_scale", False)
        self._color_palette = kwargs.get("color_palette", self._colors["palette"])

        styles = {}
        styles["size"] = kwargs.get("size", self._default_point_size)
        styles["line_width"] = kwargs.get("line_width", self._default_line_width)
        return styles

    def _set_object(self) -> None:
        """
        Create the `hv.Points` object.
        """

        self._hv_points = hv.Points(
            self._data, kdims=["x", "y"], vdims=self._vdims, label=self._label
        ).opts(
            **self._styles,
            hooks=[self.apply_point_appearance_mapper],
            hover_tooltips=self._hover_tooltips,
            active_tools=["point_draw"],
            responsive=True,
        )

    def _set_draw_tool(self) -> None:
        """
        Set the `streams.PointDraw` object used for moving points.
        """

        self._point_draw = streams.PointDraw(source=self._hv_points, add=False)

    def _get_hex_palette(self, name: str) -> list[str]:
        """
        Convert a palette name into a list of hex color codes.

        Args:
            name (str): Palette name from `colorcet`.

        Returns:
            list[str]: List of hex color codes.
        """

        palette = getattr(cc, name)
        if isinstance(palette[0], str):
            return palette
        return [
            RGB(int(r * 255), int(g * 255), int(b * 255)).to_hex()
            for r, g, b in palette
        ]
