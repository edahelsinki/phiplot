from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Callable, Any
import panel as pn
from scipy.spatial import distance

logger = logging.getLogger(__name__)

@contextmanager
def toggle_spinner(spinner: pn.widgets.LoadingSpinner):
    """
    Context manager for toggling a loading spinner indicator.

    Args:
        spinner(pn.widgets.LoadingSpinner): The panel indicator widget to toggle.
    """

    spinner.value = True
    try:
        yield
    finally:
        spinner.value = False


@contextmanager
def toggle_indicator(indicator: pn.widgets.BooleanStatus, color_conds: Callable = None):
    """
    Context manager for toggling a boolean status indicator and setting its color.

    Args:
        indicator(pn.widgets.BooleanStatus): The panel indicator widget to toggle.
        color_conds (Callable): A function that returns the indicator color.
    """

    indicator.value = False
    try:
        yield
    finally:
        indicator.color = color_conds()
        indicator.value = True


@dataclass
class ProcessResult:
    """
    An interface for unified process results.

    Attributes:
        success (bool): Whether the operation was successful.
        message (str): A human-readable status message.
        notify (bool): Whether the result should trigger a user notification.
        level (str): Log level associated with the result
            (e.g. "info", "warning", "error", "debug").
    """

    success: bool
    message: str = ""
    notify: bool = False
    level: str = "info"  # "info", "warning", "error"


def log_process(result: ProcessResult, logger: logging.Logger | None = None) -> None:
    """
    Handle logging and Panel notifications in a unified fashion.

    Args:
        result (ProcessResult): The result of a process.
        logger (Logger, optional): Logger instance to use. Defaults to root logger.
    """
    logger = logger or logging.getLogger(__name__)

    # Map levels to logging calls
    log_methods = {
        "info": logger.info,
        "debug": logger.debug,
        "warning": logger.warning,
        "error": logger.error,
    }
    log_fn = log_methods.get(result.level, logger.info)

    if result.success:
        log_fn(result.message)
    else:
        logger.error(result.message)

    # Optional Panel notifications
    if result.notify and hasattr(pn.state, "notifications"):
        notif_methods = {
            "info": pn.state.notifications.success,
            "warning": pn.state.notifications.warning,
            "error": pn.state.notifications.error,
        }
        notif_fn = notif_methods.get(result.level, pn.state.notifications.success)
        notif_fn(result.message)

def doc_link_widget(config, module_path) -> pn.pane.HTML:
    module, cls = module_path.split('.')
    url = config['doc_format'].format(
        base_url=config['base_url'],
        module=module,
        cls=cls
    )
    return pn.pane.Markdown(f"## 📄<a href='{url}' target='_blank'>Docs</a>")

def info_link_widget(url) -> pn.pane.HTML:
    return pn.pane.Markdown(f"## ℹ️<a href='{url}' target='_blank'>Info</a>")