from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import panel as pn
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.views.data_summary_view import DataSummaryView

logger = logging.getLogger(__name__)

class DataSummaryMenu(BaseMenu):
    def __init__(self, parent_view: DataSummaryView):
        super().__init__(
            parent_view=parent_view,
            name="Summarize",
            icon="chart-histogram"
        )

        self._summarizable_fields = self._get_summarizable_fields()
        self._summary_type = None
        self._construct_widgets()
        
        self._floating_panels = dict(
           numerical_summary_panel = self._build_numerical_summary_panel(),
           categorical_summary_panel = self._build_categorical_summary_panel(),
        )

        self.menu_items = [
            ("Numerical Field Summary", "numerical_summary"),
            ("Categorical Field Summary", "categorical_summary"),
        ]

        self.callbacks = dict(
            numerical_summary = lambda: self._open_numerical_summary(),
            categorical_summary = lambda: self._open_categorical_summary()
        )

    def update_fields(self) -> None:
        self._summarizable_fields = self._get_summarizable_fields()
        self._widgets["numerical_field_selector"].options = self._summarizable_fields["Numerical"]
        self._widgets["categorical_field_selector"].options = self._summarizable_fields["Categorical"]
        self._widgets["comparison_field_selector"].options = self._summarizable_fields["Numerical"]

    def _construct_widgets(self) -> None:
        """
        Construct all the Panel widgets used to control selections.
        """

        self._widgets["numerical_field_selector"] = pn.widgets.Select(
            name="Numerical Field",
            sizing_mode="stretch_width"
        )

        self._widgets["categorical_field_selector"] = pn.widgets.Select(
            name="Categorical Field",
            sizing_mode="stretch_width"
        )

        self._widgets["comparison_field_selector"] = pn.widgets.Select(
            name="Comparison Field",
            sizing_mode="stretch_width"
        )

        self._widgets["filter_toggle"] = pn.widgets.Checkbox(
            name="Use filters", 
            value=False
        )

        self._widgets["relative_freq_toggle"] = pn.widgets.Checkbox(
            name="Relative frequency", 
            value=False
        )

        self._widgets["notched_box_toggle"] = pn.widgets.Checkbox(
            name="Notched boxes", 
            value=False
        )

        self._widgets["include_kde_toggle"] = pn.widgets.Checkbox(
            name="Include KDE", 
            value=True
        )

        self._widgets["self.n_buckets_int"] = pn.widgets.IntInput(
            name="Number of Buckets",
            value=30,
            step=1,
            start=1,
            sizing_mode="stretch_width"
        )

        self._widgets["summarize_button"] = pn.widgets.Button(
            name="Summarize",
            **self._styling.default_button_style
        )
        self._widgets["summarize_button"].on_click(lambda event: self._summarize())

        self._widgets["summarize_spinner"] = pn.indicators.LoadingSpinner(
            value=False, **self._styling.default_spinner_style
        )
        
    def _build_numerical_summary_panel(self) -> WindowPanel:
        window = WindowPanel("Numerical Field Summary", self._window_destination)
        window.contents = [
            self._widgets["numerical_field_selector"],
            self._widgets["self.n_buckets_int"],
            self._widgets["filter_toggle"] ,
            self._widgets["relative_freq_toggle"],
            self._widgets["include_kde_toggle"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["summarize_button"],
                self._widgets["summarize_spinner"]
            )
        ]
        return window
    
    def _build_categorical_summary_panel(self) -> WindowPanel:
        window = WindowPanel("Categorical Field Summary", self._window_destination)
        window.contents = [
            self._widgets["categorical_field_selector"],
            self._widgets["comparison_field_selector"],
            self._widgets["filter_toggle"] ,
            self._widgets["relative_freq_toggle"],
            self._widgets["notched_box_toggle"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["summarize_button"],
                self._widgets["summarize_spinner"]
            )
        ]
        return window
    
    def _open_numerical_summary(self) -> None:
        self._summary_type = "numerical"
        self._floating_panels["numerical_summary_panel"].open()

    def _open_categorical_summary(self) -> None:
        self._summary_type = "categorical"
        self._floating_panels["categorical_summary_panel"].open()

    def _summarize(self) -> None:
        with toggle_spinner(self._widgets["summarize_spinner"]):
            if self._summary_type == "numerical":
                self._parent_view.summarize_numerical(
                    field = self._widgets["numerical_field_selector"].value,
                    use_filters = self._widgets["filter_toggle"].value,
                    n_buckets = self._widgets["self.n_buckets_int"].value,
                    use_relative_freq = self._widgets["relative_freq_toggle"].value,
                    include_KDE = self._widgets["include_kde_toggle"].value
                )
            elif self._summary_type == "categorical":
                self._parent_view.summarize_categorical(
                    field = self._widgets["categorical_field_selector"].value,
                    comparison_field = self._widgets["comparison_field_selector"].value,
                    use_filters = self._widgets["filter_toggle"].value,
                    use_relative_freq = self._widgets["relative_freq_toggle"].value,
                    use_notched = self._widgets["notched_box_toggle"].value
                )
            else:
                logger.error("Invalid summary type.")

    def _get_summarizable_fields(self) -> dict[str, list]:
        """
        Get the fields within the current collection that can be meaningfully summarized.

        Returns: 
            (dict[str, list]): Constructed as:
                -"categorical" (list): The list of categorical fields.
                -"numerical" (list): The list of numerical fields.
        """

        column_dtypes = self._data_handler.column_dtypes
        result = {"Categorical": [], "Numerical": []}
        for col in self._data_handler.columns:
            try:
                dtype = column_dtypes[col]
            except:
                continue
            if dtype in ["float", "int", "categorical_int", "categorical_str"]:
                if "categorical" in dtype:
                    result["Categorical"].append(col)
                else:
                    result["Numerical"].append(col)
        
        return result