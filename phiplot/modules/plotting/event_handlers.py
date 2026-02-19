from collections import deque
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Plot
from bokeh.events import Pan, PanStart, PanEnd, Tap
from .debouncer import DebouncedCallback


class ConstraintTracker:
    """
    Track and update embedding constraints on a Bokeh plot object
    using browser-side mouse and keyboard interactions.

    This class uses Bokeh's JavaScript event system (e.g., pan, tap, selection) to
    monitor user interactions with glyphs on a plot. It enables continuous tracking and
    real-time updating of control point positions as well as must-link and cannot-link
    pairs from the browser client.

    Args:
        source (ColumnDataSource): The data source used by the Bokeh plot glyphs.
        plot (Plot): The Bokeh plot object containing the glyphs.
        embedding_handler: Custom class handling the embedding variables.
    """

    def __init__(
        self,
        source: ColumnDataSource,
        plot: Plot,
        data_handler,
        embedding_handler,
    ) -> None:
        self.source = source
        self.plot = plot
        self.embedding_handler = embedding_handler
        self.data_handler = data_handler

        self._js_keydown_handler = None
        self._js_attached = False

        self.hover_tool = self._find_hover_tool()
        self.original_tooltips = self.hover_tool.tooltips if self.hover_tool else None

        self.last_two = deque(maxlen=2)
        self.selected = {"active": False, "idx": None, "x": None, "y": None}
        self.mouse_start = {"x": None, "y": None}

        self.mouse_source = ColumnDataSource(data=dict(x=[None], y=[None]))
        self.start_source = ColumnDataSource(data=dict(x=[None], y=[None]))
        self.keyboard_source = ColumnDataSource(data=dict(keys=[]))
        self.tap_source = ColumnDataSource(data=dict(x=[None], y=[None]))
        self.end_source = ColumnDataSource(data=dict(trigger=[0]))

        self.dragging = False

        self.listener_attached = "false"
        self._setup_callbacks()
        self.debounced_update = DebouncedCallback(
            self._update_control_point_position, interval=0.125
        )
        self._last_update_time = 0

    def _setup_callbacks(self) -> None:
        """
        Set up all necessary JavaScript and Python callbacks to handle
        user interactions with the plot.

        This includes:
            - Tap events to select glyphs and listen for keyboard input.
            - Pan events to track dragging of control points.
            - Selection changes to update the currently selected glyph.
        """

        # Tap -> select point
        self.plot.js_on_event(
            Tap,
            CustomJS(
                args=dict(ksource=self.keyboard_source, tsource=self.tap_source),
                code=f"""
                tsource.data = {{x: [cb_obj.x], y: [cb_obj.y]}};
                tsource.change.emit();

                if (window._bokehKeyHandler) {{
                    document.removeEventListener('keydown', window._bokehKeyHandler);
                    window._bokehKeyHandler = null;
                }}

                window._bokehKeyListenerAttached = {self.listener_attached}
                if (!window._bokehKeyListenerAttached) {{
                    window._bokehKeyHandler = function(event) {{
                        if (!ksource.data.count) {{
                            ksource.data.count = [0];
                        }}
                        let counter = ksource.data.count[0] + 1;
                        ksource.data = {{keys: [event.key], count: [counter]}};
                        ksource.change.emit();
                    }};
                    document.addEventListener('keydown', window._bokehKeyHandler);
                }}
                """,
            ),
        )
        self.listener_attached = "true"
        self.tap_source.on_change("data", self._on_tap)
        self.keyboard_source.on_change("data", self._on_keypress)

        # Pan -> update control point position continuously
        self.plot.js_on_event(
            Pan,
            CustomJS(
                args=dict(source=self.mouse_source),
                code="""
            source.data = {x: [cb_obj.x], y: [cb_obj.y]};
            source.change.emit();
        """,
            ),
        )
        self.mouse_source.on_change("data", self._on_move)

        # PanStart -> determine mouse position at start
        self.plot.js_on_event(
            PanStart,
            CustomJS(
                args=dict(source=self.start_source),
                code="""
            source.data = {x: [cb_obj.x], y: [cb_obj.y]};
            source.change.emit();
        """,
            ),
        )
        self.start_source.on_change("data", self._on_start)

        # PanEnd -> sync final control point
        self.plot.js_on_event(
            PanEnd,
            CustomJS(
                args=dict(source=self.end_source),
                code="""
            source.data = {trigger: [Date.now()]};
            source.change.emit();
        """,
            ),
        )
        self.end_source.on_change("data", self._on_end)

        # Currently selected point
        self.source.selected.on_change("indices", self._on_selection_change)

    def _on_tap(self, attr, old, new) -> None:
        """
        Track the indices of the two most recently tapped glyphs.

        This maintains a rolling record of recent selections for
        use in actions like adding constraints or marking control points.
        """

        if self.selected["idx"] is None:
            return
        idx = self.selected["idx"]
        self.last_two.append(idx)

    def _on_start(self, attr, old, new) -> None:
        """
        Initialize dragging state and record the mouse position at the start of a pan event.
        Also disables hover tooltips during dragging for clarity.
        """

        self.dragging = True
        self.mouse_start = {"x": new["x"][0], "y": new["y"][0]}
        if self.hover_tool:
            self.hover_tool.tooltips = ""

    def _on_move(self, attr, old, new) -> None:
        """
        Continuously track the mouse position during a pan event to update the position
        of the currently dragged control point.

        The control point position is updated based on the delta movement from the drag start.
        """
        if not self.selected["active"] or self.mouse_start["x"] is None:
            return
        self.debounced_update(new)

    def _update_control_point_position(self, new):
        idx = self.selected["idx"]
        x, y = new["x"][0], new["y"][0]
        dx, dy = x - self.mouse_start["x"], y - self.mouse_start["y"]
        sx, sy = self.selected["x"], self.selected["y"]

        self.embedding_handler.add_control_point(idx, sx + dx, sy + dy)

    def _update_control_points(self, idx: int) -> None:
        """
        Add a new control point or update the position of an existing one.

        Args:
            idx (int): Index of the glyph to add or update as a control point.
        """

        x = self.source.data["x"][idx]
        y = self.source.data["y"][idx]
        self.embedding_handler.add_control_point(idx, x, y)

    def _add_must_link(self) -> None:
        """
        Add a must-link between the two most recently selected glyph indices.
        """

        if len(self.last_two) == 2:
            link = tuple(sorted(self.last_two))
            self.embedding_handler.add_must_link(link)

    def _add_cannot_link(self) -> None:
        """
        Add a cannot-link between the two most recently selected glyph indices.
        """
        if len(self.last_two) == 2:
            link = tuple(sorted(self.last_two))
            self.embedding_handler.add_cannot_link(link)

    def _on_keypress(self, attr, old, new) -> None:
        """
        Handle keyboard input after a glyph is tapped:

        - `p`: Mark the most recently selected glyph as a control point.
        - `m`: Add a must-link constraint between the two most recently selected glyphs.
        - `c`: Add a cannot-link constraint between the two most recently selected glyphs.
        """

        if self.keyboard_source.data["keys"][0] == "p":
            idx = self.last_two[-1]
            self._update_control_points(idx)
        elif self.keyboard_source.data["keys"][0] == "m":
            self._add_must_link()
        elif self.keyboard_source.data["keys"][0] == "c":
            self._add_cannot_link()

    def _on_end(self, attr, old, new) -> None:
        """
        Finalize the position of a control point after dragging ends.

        This method:
            - Marks the drag as complete.
            - Updates the position of the dragged point if a valid index is selected.
            - Resets the current selection.
            - Flags that an existing control point was updated.
            - Restores the original tooltips if they were hidden during dragging.
        """

        self.dragging = False
        idx = self.selected["idx"]
        if idx is not None:
            self._update_control_points(idx)
        self.selected = {"active": False, "idx": None, "x": None, "y": None}

        if self.hover_tool and self.original_tooltips:
            self.hover_tool.tooltips = self.original_tooltips

    def _on_selection_change(self, attr, old, new) -> None:
        """
        Store the index and coordinates of the most recently selected glyph.
        Triggered when the user selects a glyph (e.g., by tapping or clicking).
        """

        if not new:
            return

        idx = new[0]
        self.selected = {
            "active": True,
            "idx": idx,
            "x": self.source.data["x"][idx],
            "y": self.source.data["y"][idx],
        }

    def _find_hover_tool(self) -> HoverTool | None:
        """
        Locate the HoverTool in the current Bokeh plot.

        Returns:
            HoverTool if found, otherwise None.
        Used to temporarily disable tooltips during control point dragging.
        """

        for tool in self.plot.tools:
            if isinstance(tool, HoverTool):
                return tool
        return None
