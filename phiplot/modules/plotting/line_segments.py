import logging
from typing import Any
from bokeh.models import ColumnDataSource
import holoviews as hv

logger = logging.getLogger(__name__)


class LineSegments:
    """
    Manages the handling of the line segment objects.

    Acts as an abstraction layer between the `hv.Segments` object and the InteractivePlot
    providing methods to build and update the `hv.Segments` object.

    Args:
        label (str): Label for the `hv.Segments` object (used e.g. in the legend)
        data (pd.DataFrame): All the necessary data for the plotting.
        links (list[tuple]): The pairs of indices for the points to connect.
        **kwargs: Optional arguments for setting up the styling:
            -"color" (str): The color of the line
            -"line_width" (int): The width of the line in default units.
            -"line_dash" (str): The style of the line (e.g. "solid" or "dashed").
    """

    def __init__(self, label, data, links=[], **kwargs):
        self._label = label
        self._data = data
        self._links = links

        self._styles = self._resolve_kwargs(kwargs)

        self._source = None
        self._hv_segments = None

        self._set_object()

    def set_references(self, source: ColumnDataSource) -> None:
        """
        Set the reference to the right datasource within the rendered Bokeh plot.

        Args:
            source (ColumnDataSource): The datasource for the line segments within the rendered Bokeh plot.
        """

        self._source = source

    def clear_references(self) -> None:
        """
        Clear the reference to the previous datasource.
        """

        self._source = None

    def get_object(self) -> hv.Segments:
        """
        Get the underlying `hv.Segments` object.

        Returns:
            hv.Segments: The segments to render.
        """

        return self._hv_segments

    def update_coords(self, x_new: list[float], y_new: list[float]) -> None:
        """
        Update the coordinates in the underlying ColumnDataSource of the line segments.

        Args:
            x_new (list[float]): The new x-coordinates.
            y_new (list[float]): The new y-coordinates.
        """

        xs0, ys0, xs1, ys1 = (
            zip(*[(x_new[i], y_new[i], x_new[j], y_new[j]) for i, j in self._links])
            if len(self._links) > 0
            else ([], [], [], [])
        )
        self._source.data.update({"x0": xs0, "y0": ys0, "x1": xs1, "y1": ys1})

    def update_links(self, new_links: list[tuple]) -> None:
        """
        Update the index pairs for the points to link.

        Args:
            new_links (list[tuple]): The new pairs of indices.

        """

        self._links = new_links

    def _resolve_kwargs(self, kwargs: dict) -> dict[str, Any]:
        """
        Resolve the optional keyword arguments to set up the styling for the line segments.
        Uses predefined defaults for missing arguments.

        Args:
            kwargs (dict): The keyword arguments described in the class-level doc.

        Returns:
            dict: The resolved styling.
        """

        styles = {}
        styles["color"] = kwargs.get("color", "black")
        styles["line_width"] = kwargs.get("line_width", 8)
        styles["line_dash"] = kwargs.get("line_dash", "solid")
        return styles

    def _set_object(self) -> None:
        """
        Create the `hv.Segments` object.
        """

        segments = [
            (
                self._data.iloc[i]["x"],
                self._data.iloc[i]["y"],
                self._data.iloc[j]["x"],
                self._data.iloc[j]["y"],
            )
            for i, j in self._links
        ]
        self._hv_segments = hv.Segments(segments, label=self._label).opts(
            **self._styles
        )
