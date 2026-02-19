import logging
import panel as pn
import holoviews as hv
from holoviews import streams

logger = logging.getLogger(__name__)


class Highlighter:
    """
    Manages the higlighting of points.

    Uses a `hv.Points` object with a single value to create a fading colored
    disk around a point within the InteractivePlot when invoked with the
    `highlight` method.

    Args:
        radius (int): Optional. Radius of the disk in default units. Defaults to 10.
        color (str): Optional. The color of the disk. Defaults to "red".
        initial_alpha (float). Optional. The initial opacity of the disk. Defaults to 0.6.
        fade_duration (int). Optional. The amount of time in ms the disk stays visible. Defaults to 2000.
    """

    def __init__(self, radius=20, color="fuchsia", initial_alpha=0.75, fade_duration=3000):
        self._radius = radius
        self._color = color
        self._initial_alpha = initial_alpha
        self._fade_duration = fade_duration

        self._alpha = initial_alpha
        self._reduce_by = 0.025
        self._period = int(fade_duration * self._reduce_by / self._initial_alpha)

        self._data = {"x": [], "y": [], "fill_alpha": []}
        self._stream = streams.Stream.define("HighlightStream", **self._data)()
        self._dmap = hv.DynamicMap(self._plot, streams=[self._stream])

        self._fade_callback = None

    def get_object(self) -> hv.DynamicMap:
        """
        Get the underlying `hv.Points` object set up via the dynamic map.

        Returs:
            hv.DynamicMap: The disk to render when the underlying data stream updates.
        """

        return self._dmap

    def clear_references(self) -> None:
        """
        Clear the data stream for the disk.
        """

        self._stream.event(x=[], y=[], fill_alpha=[])
        if self._fade_callback:
            self._fade_callback.stop()
            self._fade_callback = None

    def highlight(self, x: float, y: float) -> None:
        """
        Highlight a point with a fading disk around it.

        Args:
            x (float): The x-coordinate of the point to highlight
            y (float): The y-coordinate of the point to highlight.
        """

        self._alpha = self._initial_alpha
        self._data = {"x": [x], "y": [y], "fill_alpha": [self._alpha]}
        self._stream.event(**self._data)

        if self._fade_callback:
            self._fade_callback.stop()

        self._fade_callback = pn.state.add_periodic_callback(
            self._fade_step, period=self._period
        )

    def _plot(self, x: float, y: float, fill_alpha: float) -> None:
        """
        Create the single point `hv.Points` object for the disk.

        Args:
            x (float): The x-coordinate of the disk.
            y (float): The y-coordinate of the disk.
            fill_alpha (float): The opacity of the disk initially.
        """

        if not x or not y:
            return hv.Points([])

        return hv.Points(
            {"x": x, "y": y, "fill_alpha": fill_alpha},
            kdims=["x", "y"],
            vdims=["fill_alpha"],
        ).opts(
            fill_color=self._color,
            size=2 * self._radius,
            fill_alpha="fill_alpha",
            line_alpha=0,
            responsive=True,
        )

    def _fade_step(self) -> None:
        """
        Incrementally reduce the opacity of the disk until its invisible.
        """

        self._alpha -= self._reduce_by
        if self._alpha <= 0:
            self._data["fill_alpha"] = [0]
            self._fade_callback.stop()
            self._fade_callback = None
        else:
            self._data["fill_alpha"] = [self._alpha]

        self._stream.event(**self._data)
