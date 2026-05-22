import time
from typing import Any, Callable, Dict, Optional, Tuple
import panel as pn

class ThrottledCallback:
    """A rate-limiting wrapper for callbacks using Panel's event loop.

    Ensures that a callback is executed at most once per specified interval. 
    This implementation uses a "leading-edge + trailing-edge" approach: 
    it executes immediately if the interval has passed, or schedules a 
    delayed execution if called during the quiet period.

    Args:
        callback: The function to be executed.
        interval: The minimum time between executions in seconds.
    """

    def __init__(self, callback: Callable[..., Any], interval: float = 0.1) -> None:
        self.callback = callback
        self.interval = interval
        self._last_run_time: float = 0.0
        self._timeout_handle: Optional[Any] = None
        self._args: Tuple[Any, ...] = ()
        self._kwargs: Dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Invoke the throttled callback.

        Updates the internal arguments to the latest values. If the interval 
        has passed, it executes immediately. Otherwise, it schedules a single 
        future execution if one is not already pending.

        Args:
            *args: Positional arguments for the callback.
            **kwargs: Keyword arguments for the callback.
        """
        self._args = args
        self._kwargs = kwargs
        
        now = time.time()
        elapsed = now - self._last_run_time

        # Case 1: Execute immediately if time has elapsed and no timer is pending
        if elapsed >= self.interval and self._timeout_handle is None:
            self._run()
        
        # Case 2: Schedule a trailing execution if no timer is already active
        elif self._timeout_handle is None:
            wait_ms = int(max(0, (self.interval - elapsed) * 1000))
            self._timeout_handle = pn.state.curdoc.add_timeout_callback(
                self._run, wait_ms
            )

    def _run(self) -> None:
        """The actual execution wrapper triggered by the throttler or timer.

        Updates the last run timestamp and executes the callback with the 
        most recent arguments. Ensures the timeout handle is cleared even 
        if the callback fails.
        """
        try:
            self._last_run_time = time.time()
            self.callback(*self._args, **self._kwargs)
        finally:
            # Clear the handle to allow subsequent calls to schedule new timers
            self._timeout_handle = None