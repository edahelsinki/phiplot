from collections import deque
from typing import Any, Dict, List, Optional, Union
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Plot
from bokeh.events import Pan, PanStart, PanEnd, Tap
from .throttler import ThrottledCallback

class ConstraintTracker:
    """Handles interactive embedding constraints via Bokeh browser events.

    This class synchronizes low-latency browser-side dragging with server-side
    embedding recalculations. It manages control points, must-link, and
    cannot-link constraints using mouse and keyboard inputs.

    Args:
        scatter: The abstraction layer managing the scatter object (e.g., TrackedPointScatter).
        source: The Bokeh ColumnDataSource containing point coordinates.
        plot: The Bokeh figure/plot object where events are registered.
        embedding_handler: The logic handler for the embedding algorithm and constraints.
    """

    def __init__(
        self,
        scatter: Any,
        source: ColumnDataSource,
        plot: Plot,
        embedding_handler: Any
    ) -> None:

        self.scatter = scatter
        self.source = source
        self.plot = plot
        self.embedding_handler = embedding_handler

        self.ui_bridge = ColumnDataSource(data=dict(
            idx=[-1], x=[0.0], y=[0.0], dragging=[0], key=[""], key_count=[0]
        ))

        self.last_two: deque = deque(maxlen=2)
        self.dragging: bool = False
        self.target_coords: Dict[str, Union[float, int]] = {"x": 0.0, "y": 0.0, "idx": -1}

        self.hover_tool = self._find_hover_tool()
        self.original_tooltips = self.hover_tool.tooltips if self.hover_tool else None

        self._setup_callbacks()

        # Only listen to the bridge and selection changes. No more data-echoing.
        self.ui_bridge.on_change("data", self._handle_ui_signal)
        self.source.selected.on_change("indices", self._on_selection_change)

    def _setup_callbacks(self) -> None:
        args = dict(source=self.source, bridge=self.ui_bridge)

        # PanStart: Record initial positions and initialize throttle timer
        self.plot.js_on_event(PanStart, CustomJS(args=args, code="""
            const indices = source.selected.indices;
            if (indices.length > 0) {
                const idx = indices[0];
                window._dragIdx = idx;
                window._mouseX0 = cb_obj.x;
                window._mouseY0 = cb_obj.y;
                window._ptX0 = source.data['x'][idx];
                window._ptY0 = source.data['y'][idx];
                window._lastEmit = Date.now(); // Throttle timer init
                
                bridge.data = {...bridge.data, idx: [idx], dragging: [1]};
                bridge.change.emit();
            }
        """))

        # Pan: Update visual instantly, but THROTTLE websocket messages to 333ms
        self.plot.js_on_event(Pan, CustomJS(args=args, code="""
            if (window._dragIdx !== undefined) {
                const dx = cb_obj.x - window._mouseX0;
                const dy = cb_obj.y - window._mouseY0;
                const nx = window._ptX0 + dx;
                const ny = window._ptY0 + dy;
                
                // 1. Instant visual update (No network traffic)
                source.data['x'][window._dragIdx] = nx;
                source.data['y'][window._dragIdx] = ny;
                source.change.emit();

                // 2. Throttled network update to Python (Max 20 times a second)
                const now = Date.now();
                if (now - window._lastEmit > 50) {
                    bridge.data = {...bridge.data, x: [nx], y: [ny]};
                    bridge.change.emit();
                    window._lastEmit = now;
                }
            }
        """))

        self.plot.js_on_event(PanEnd, CustomJS(args=args, code="""
            if (window._dragIdx !== undefined) {
                // Ensure the absolute final position is sent
                const finalX = source.data['x'][window._dragIdx];
                const finalY = source.data['y'][window._dragIdx];
                
                window._dragIdx = undefined;
                bridge.data = {...bridge.data, x: [finalX], y: [finalY], dragging: [0], idx: [-1]};
                bridge.change.emit();
            }
        """))

        self.plot.js_on_event(Tap, CustomJS(args=args, code="""
            if (window._bokehKeyHandler) {
                document.removeEventListener('keydown', window._bokehKeyHandler);
            }
            window._bokehKeyHandler = (e) => {
                if (['p', 'm', 'c'].includes(e.key)) {
                    bridge.data = {...bridge.data, key: [e.key], key_count: [bridge.data.key_count[0] + 1]};
                    bridge.change.emit();
                }
            };
            document.addEventListener('keydown', window._bokehKeyHandler);
        """))

    def _handle_ui_signal(self, attr: str, old: Dict[str, Any], new: Dict[str, Any]) -> None:
        # Drag Logic
        if new['dragging'][0] == 1:
            self.dragging = True
            if self.hover_tool:
                self.hover_tool.tooltips = ""
            
            # Since JS is now throttling the network, Python can safely execute this directly
            self.embedding_handler.add_control_point(new['idx'][0], new['x'][0], new['y'][0])
            
        elif self.dragging and new['dragging'][0] == 0:
            self.target_coords = {"idx": new['idx'][0], "x": new['x'][0], "y": new['y'][0]}
            self._finalize_drag()

        # Keystroke Logic
        if new['key_count'][0] != old['key_count'][0]:
            self._on_keypress(new['key'][0])

    def _on_keypress(self, key: str) -> None:
        if not self.last_two:
            return
        idx = self.last_two[-1]

        try:
            if key == "p":
                self.embedding_handler.add_control_point(idx, self.source.data['x'][idx], self.source.data['y'][idx])
            elif key == "m" and len(self.last_two) == 2:
                self.embedding_handler.add_must_link(tuple(sorted(self.last_two)))
            elif key == "c" and len(self.last_two) == 2:
                self.embedding_handler.add_cannot_link(tuple(sorted(self.last_two)))

            e = self.embedding_handler.embedding
            self.scatter.update_coords(e["x"], e["y"])
            self.embedding_handler.refresh_plot = True

        except Exception as err:
            print(f"Error updating constraints: {err}")

    def _finalize_drag(self) -> None:
        self.dragging = False
        idx = int(self.target_coords['idx'])
        if idx != -1:
            self.embedding_handler.add_control_point(idx, self.target_coords['x'], self.target_coords['y'])
            self.embedding_handler.refresh_cp_display = True

        if self.hover_tool:
            self.hover_tool.tooltips = self.original_tooltips

    def _on_selection_change(self, attr: str, old: List[int], new: List[int]) -> None:
        if new:
            self.last_two.append(new[0])

    def _find_hover_tool(self) -> Optional[HoverTool]:
        return next((t for t in self.plot.tools if isinstance(t, HoverTool)), None)