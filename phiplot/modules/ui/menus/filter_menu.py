from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import panel as pn
from .base_menu import BaseMenu
from phiplot.modules.ui.widgets import *
from phiplot.modules.ui.floating import *

if TYPE_CHECKING:
    from phiplot.modules.ui.web_ui import WebUI

logger = logging.getLogger(__name__)


class FilterMenu(BaseMenu):
    """
    UI Panel group for managing dataset filters.

    Provides an funtionality for applying, clearing, and removing
    filters on the dataset. Changes propagate to the interactive plot.

    Args:
        app (App): Reference to the main application instance.
        window_destination (pn.Column): Destination container for rendering floating panels.
    """

    def __init__(self, parent_view: WebUI):
        super().__init__(
            parent_view=parent_view,
            name="Filter",
            icon="filter"
        )

        self._floating_panels = dict(
            add_filter_panel=self._build_add_filter_panel()
        )

        self._widgets["filter_display"] = DisplayPanel("Applied Filters")
        self._widgets["filter_display"].set_watcher(action=self._remove_filters)

        self.menu_items = [
            ("Add Filter", "add_filter"),
            ("Clear Filters", "clear_filters"),
            ("Remove Filtered Points", "remove_filtered")
        ]

        self.callbacks = dict(
            add_filter = lambda: self._floating_panels["add_filter_panel"].open(),
            remove_filtered = lambda: self.remove_filtered_points(),
            clear_filters = lambda: self.remove_all_filters(),
        )

    def update_available_features(self) -> None:
        self._widgets["filterby_feature_selector"].options = self._data_handler.columns

    def _build_add_filter_panel(self, event=None) -> WindowPanel:
        """
        Create the panel for adding a new filter.

        Returns:
            WindowPanel: A floating panel with selectors for
            feature, filter type, filter options, and an apply button.
        """

        self._widgets["filterby_feature_selector"] = pn.widgets.Select(
            name="Filter Points by",
            value="",
            options=self._data_handler.columns,
            sizing_mode="stretch_width",
        )

        self._widgets["filtering_type_selector"] = pn.widgets.Select(
            name="Filtering type",
            sizing_mode="stretch_width"
        )

        self.filterby_options = pn.Row()

        self._widgets["filterby_feature_selector"].param.watch(
            self._sync_filtering_types, 'value'
        )

        self._widgets["filtering_type_selector"].param.watch(
            lambda e: self._update_filterby_opts(e.new), 'value'
        )

        self._widgets["apply_filter_button"] = pn.widgets.Button(
            name="Apply Filter", **self._styling.default_button_style
        )
        self._widgets["apply_filter_button"].on_click(self._add_filter)

        self._sync_filtering_types()

        window = WindowPanel("Add Filter", self._window_destination)
        window.contents = [
            self._widgets["filterby_feature_selector"],
            self._widgets["filtering_type_selector"],
            self.filterby_options,
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["apply_filter_button"],
        ]
        return window

    def _sync_filtering_types(self, event=None):
        """Updates the available types (Range, etc) based on the column dtype."""
        feature = self._widgets["filterby_feature_selector"].value
        dtype = self._data_handler.column_dtypes.get(feature)
        
        if dtype in ["int", "float"]:
            options = ["Range", "Less than", "Greater than", "Equal to Number", "Set of Values"]
        else:
            options = ["Equal to Categorical", "Set of Categoricals"]
            
        self._widgets["filtering_type_selector"].options = options
        self._widgets["filtering_type_selector"].value = options[0]

    def _update_filterby_opts(self, option) -> None:
        """
        Update available filter input widgets based on the selected filter type.

        Args:
            event: Change event from the filter type selector.
        """

        if option == "Range":
            self.filterby_options.objects = [
                pn.widgets.FloatInput(
                    name="Start", value=0.0, step=1, sizing_mode="stretch_width"
                ),
                pn.widgets.FloatInput(
                    name="End", value=1.0, step=1, sizing_mode="stretch_width"
                ),
            ]
        elif option in ["Equal to Number", "Greater than", "Less than"]:
            self.filterby_options.objects = [
                pn.widgets.FloatInput(
                    name="Value", value=0, step=1, sizing_mode="stretch_width"
                ),
            ]
        elif option == "Set of Values":
            self.filterby_options.objects = [
                pn.widgets.TextInput(
                    name="Value",
                    placeholder="Give the numerical values separated by commas",
                    sizing_mode="stretch_width",
                ),
            ]
        elif option == "Equal to Categorical":
            self.filterby_options.objects = [
                pn.widgets.TextInput(
                    name="Value",
                    placeholder="Give categorical value",
                    sizing_mode="stretch_width",
                ),
            ]
        elif option == "Set of Categoricals":
            self.filterby_options.objects = [
                pn.widgets.TextInput(
                    name="Value",
                    placeholder="Give categorical values separated by commas",
                    sizing_mode="stretch_width",
                ),
            ]
        else:
            logger.error("Unrecognized filter.")


    def _add_filter(self, event=None) -> None:
        """
        Apply a new filter to the dataset.

        Args:
            event: Click event from the apply button.
        """

        self._plot = self._parent_view.views["embedding"].plot

        filter_type = self._widgets["filtering_type_selector"].value.lower().replace(" ", "_")
        feature = self._widgets["filterby_feature_selector"].value
        filter_options = [w.value for w in self.filterby_options]

        if not feature:
            pn.state.notifications.warning("No feature selected for the filter...")
            return
        
        formatted = self._data_handler.add_filter(feature, filter_type, filter_options)
        self._widgets["filter_display"].value = formatted

        if self._plot.is_initialized:
            self._plot.apply_filters()

    def _remove_filters(self, event=None) -> None:
        """
        Remove filters based on changes in the filter display.

        Args:
            event: Panel value-change event from `DisplayPanel`.
        """

        old_filters = set(event.old)
        new_filters = set(event.new)
        deleted = old_filters - new_filters
        if deleted:
            formatted = self._data_handler.remove_filters(deleted)
            self._widgets["filter_display"].value = formatted
            if self._plot.is_initialized:
                self._plot.apply_filters()

    def remove_all_filters(self, event=None) -> None:
        """
        Clear all filters from the dataset.

        Args:
            event: Menu selection event.
        """

        self._widgets["filter_display"].value = []

    def remove_filtered_points(self, event=None) -> None:
        """
        Remove all datapoints currently filtered out.

        Args:
            event: Menu selection event.
        """

        self._parent_view.views["embedding"].embedding_data_handler.remove_filtered_points()
        self._parent_view.views["embedding"].embed()