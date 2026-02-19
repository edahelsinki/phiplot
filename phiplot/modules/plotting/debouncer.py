import time
from typing import Callable, Any
import panel as pn

class DebouncedCallback:
    """
    A debounced wrapper for a callback function using Panel's event loop.

    Ensures the callback is not called more often than a specified interval.
    If invoked repeatedly, it delays execution until the interval has passed,
    only triggering the most recent call.

    Parameters:
        callback (callable): The function to debounce.
        interval (float): Minimum interval (in seconds) between executions.
            Defaults to 0.1
    """

    def __init__(self, callback: Callable[..., Any], interval: float = 0.1) -> None:
        self.callback = callback
        self.interval = interval
        self._last_call = 0.0
        self._pending = False
        self._args = ()
        self._kwargs = {}

    def __call__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        pn.state.curdoc.add_next_tick_callback(self._schedule)

    def _schedule(self) -> None:
        now = time.time()
        elapsed = now - self._last_call

        if elapsed >= self.interval:
            self._last_call = now
            self._pending = False
            self.callback(*self._args, **self._kwargs)
        elif not self._pending:
            self._pending = True
            wait_ms = int((self.interval - elapsed) * 1000)
            pn.state.curdoc.add_timeout_callback(self._schedule, wait_ms)