from __future__ import annotations
import logging
import time
import panel as pn

logger = logging.getLogger(__name__)


class ProgressBar:
    """
    Monospaced text + progress bar component for tracking task progress.

    Displays:
    - Elapsed time (ms)
    - Optional counted units (e.g., items processed)

    Args:
        desc (str): Description of the tracked process.
        unit (str, optional): Unit label for counted progress.
        counted (bool, optional): If True, display count + unit.
        width (int, optional): Width of the progress bar in pixels.
        color (str, optional): Text color. Defaults to gray.
    """

    def __init__(
        self,
        desc: str,
        unit: str | None = None,
        counted: bool = False,
        width: int = 480,
        color: str = "#666666",
    ):
        self.desc = desc
        self.unit = unit
        self.color = color
        self.counted = counted

        self.start_time = 0
        self.elapsed = 0
        self.count = 0
        self.rate = 0

        styles = dict(
            color=self.color, font_family="monospace", margin_bottom="0", padding_bottom="0"
        )

        self.text_pane = pn.pane.HTML("<p>Waiting fetching to start...", styles=styles)

        self.progress_bar = pn.widgets.Progress(active=False, value=0, width=width, styles=dict(margin_top="0", padding_top="0"))

    def build(self) -> pn.Column:
        """
        Build and return the progress bar layout.

        Returns:
            pn.Column: Layout containing text + progress bar widget.
        """

        return pn.Column(self.text_pane, self.progress_bar)
    
    def empty(self) -> None:
        self.text_pane.object = "Waiting fetching to start..."
        self.progress_bar.active = False

    def start(self) -> None:
        """
        Start the progress timer and activate the progress bar.
        """

        self.count = 0
        self.start_time = time.perf_counter()
        self.progress_bar.value = -1
        self.progress_bar.active = True

    def update(self) -> None:
        """
        Increment the counter (if counted) and refresh display
        every 100 iterations.
        """

        self.count += 1
        if self.count == 1 or self.count % 10 == 0:
            self.elapsed = int(1000 * (time.perf_counter() - self.start_time))
            self._update_text_pane()

    def finished(self) -> None:
        """
        Mark the progress as finished, stop the bar animation,
        and update the final elapsed time.
        """

        self.progress_bar.active = False
        self.elapsed = int(1000 * (time.perf_counter() - self.start_time))
        self._update_text_pane()

    def _update_text_pane(self) -> None:
        """
        Internal method to refresh the text display with latest
        elapsed time and optional count.
        """

        unit = self.unit
        if self.counted:
            if self.count > 1 and unit:
                unit += "s"
            self.text_pane.object = (
                f"<p>{self.desc}: {self.count} {unit} [{self.elapsed} ms elapsed]</p>"
            )
        else:
            self.text_pane.object = f"<p>{self.desc} [{self.elapsed} ms elapsed]</p>"