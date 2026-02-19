from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import holoviews as hv
from holoviews import Store
import panel as pn
from jinja2 import Environment, FileSystemLoader
from .base_view import BaseView
from phiplot.modules.ui.floating import *
from phiplot.modules.plotting.summary_plots import SummaryPlots
from phiplot.modules.ui.menus import *

if TYPE_CHECKING:
    from phiplot.modules.ui.web_ui import WebUI

logger = logging.getLogger(__name__)

# Load HTML templates
env = Environment(loader=FileSystemLoader("phiplot/assets/templates"))


class DataSummaryView(BaseView):
    def __init__(self, ui: WebUI):
        super().__init__(ui)
        self.title = "Data Summary"

        self._menus = dict(
            summary = DataSummaryMenu(self),
            appearance = DataSummaryAppearanceMenu(self)
        )

        self._floating_panels = dict(
            molecule_info = self._build_molecule_info_panel()
        )

        self._init_panes()

        self.center_column = [
            self._boxplot_pane,
            self._distribution_pane,
            self._cdf_pane
        ]

        self.left_column = [
            ("Summary Statistics", self.summary_pane)
        ]

        self.right_column = []

        self._prev_field_type = None
        self._prev_options = None
        self._current_summary = None
        self._current_cat_summaries = None

    def update_options(self) -> None:
        """
        Update the widgets to show options from the currently
        selected database collection.
        """

        self._menus["summary"].update_fields()

    def toggle_legends(self):
        for pane in [self._boxplot_pane, self._distribution_pane, self._cdf_pane]:
            obj = pane.object
            try:
                options = Store.lookup_options(Store.current_backend, obj, "plot")
                current = options.kwargs.get("show_legend", True)
                pane.object = obj.opts(show_legend = not current)
            except:
                pass

    def update_colors(self) -> None:
        colors = self._menus["appearance"].get_color_settings()
        self.change_plot_color_scheme(colors["plot"]["color_scheme"])

        if not self._prev_options:
            return
        
        kwargs = dict(
            **self._prev_options,
            recompute=False
        )
        
        if self._prev_field_type == "numerical":
            self.summarize_numerical(**kwargs)
        elif self._prev_field_type == "categorical":
            self.summarize_categorical(**kwargs)

    def summarize_numerical(
        self,
        field: str,
        use_filters: bool = False,
        n_buckets: int = 30,
        use_relative_freq: bool = False,
        include_KDE: bool = False,
        recompute: bool = True
    ) -> None:
        """
        Compute the summary statiscs for the current selection and update the view.

        Args:
            event: A Panel button click event.
        """

        colors = self._menus["appearance"].get_color_settings()

        current_options = dict(
            field=field,
            n_buckets=n_buckets,
            use_filters=use_filters,
            use_relative_freq=use_relative_freq,
            include_KDE=include_KDE
        )

        if self._prev_options == current_options:
            recompute = False
        else:
            self._prev_field_type = "numerical"
            self._prev_options = current_options

        self.title = f"Summary of {field}"

        if recompute:
            summary = self.data_handler.get_numerical_summary(field, use_filters, n_buckets)
            self._current_summary = summary
        else:
            summary = self._current_summary

        self._distribution_pane.object = SummaryPlots.build_histogram(
            edges=summary["edges"],
            counts=summary["counts"],
            xlabel=field,
            CDF=False,
            KDE=include_KDE,
            relative_freq=use_relative_freq,
            colors=colors["histogram"]
        ).opts(**self.styling.plot_light)

        self._cdf_pane.object = SummaryPlots.build_histogram(
            edges=summary["edges"],
            counts=summary["counts"],
            xlabel=field,
            CDF=True,
            KDE=False,
            relative_freq=use_relative_freq,
            colors=colors["histogram"]
        ).opts(**self.styling.plot_light)

        self._boxplot_pane.object = SummaryPlots.build_individual_box_plot(
            summary=summary["summary_stats"],
            ylabel=field,
            colors=colors["box_plot"]
        ).opts(**self.styling.plot_light)
        
        if summary:
            self._build_statistics_section(summary["summary_stats"])

    def summarize_categorical(
            self,
            field: str,
            comparison_field: str,
            use_filters: bool = False,
            use_relative_freq: bool = False,
            use_notched: bool=False,
            recompute: bool = True
        ) -> None:

        colors = self._menus["appearance"].get_color_settings()

        current_options = dict(
            field=field,
            comparison_field=comparison_field,
            use_filters=use_filters,
            use_relative_freq=use_relative_freq,
            use_notched=use_notched
        )

        if self._prev_options == current_options:
            recompute = False
        else:
            self._prev_field_type = "categorical"
            self._prev_options = current_options
        
        self._prev_options = current_options
            
        self.title = f"Summary of {field}"

        if recompute:
            summary = self.data_handler.get_categorical_summary(field, use_filters)

            cat_summaries = dict(sorted(self.data_handler.get_categorical_comparison(
                field, summary["labels"], comparison_field
            ).items())) # Same sorting as for the summary

            self._current_summary = summary
            self._current_cat_summaries = cat_summaries

        else:
            summary = self._current_summary
            cat_summaries = self._current_cat_summaries

        # Sort the labels in descending order
        labels, counts = zip(*sorted(zip(summary["labels"], summary["counts"]), key=lambda x: x[0]))

        self._boxplot_pane.object = SummaryPlots.build_comparison_box_plot(
            summaries=cat_summaries,
            xlabel=field,
            ylabel=comparison_field,
            notched=use_notched,
            colors=colors["box_plot"]
        ).opts(**self.styling.plot_light)

        self._distribution_pane.object = SummaryPlots.build_bar_plot(
            labels=labels,
            counts=counts,
            xlabel=field,
            relative_freq=use_relative_freq,
            colors=colors["histogram"]
        ).opts(**self.styling.plot_light)

        self._cdf_pane.object = None

        if summary:
            self._build_statistics_section(summary["summary_stats"])
    
    def change_plot_color_scheme(self, color_scheme: str):
        for pane in [self._boxplot_pane, self._distribution_pane, self._cdf_pane]:
            style = self.styling.plot_light
            if color_scheme == "dark":
                style = self.styling.plot_dark
            pane.object = pane.object.opts(**style)

    def _init_panes(self) -> None:
        self._boxplot_pane = pn.pane.HoloViews(
            hv.BoxWhisker([]).opts(
                ylabel="Search Field",
                title="Box Plot"
            ),
            sizing_mode="stretch_both"
        )

        self._distribution_pane = pn.pane.HoloViews(
            hv.Histogram(([], [])).opts(
                xlabel="Search Field",
                ylabel="Frequency",
                title="Distribution"
            ),
            sizing_mode="stretch_both"
        )

        self._cdf_pane = pn.pane.HoloViews(
            hv.Histogram(([], [])).opts(
                xlabel="Search Field",
                ylabel="Frequency",
                title="Cumulative Distribution"
            ),
            sizing_mode="stretch_both"
        )

        self.summary_pane = pn.pane.HTML(
            sizing_mode="stretch_width"
        )

    def _build_statistics_section(self, summary: dict[str, int | float | str]):
        """
        Build the summary statistics section and update the corresponding pane.

        Args:
            summary (dict[str, int | float | str]): The precomputed summary statistics.
        """

        template = env.get_template("simple_table.html")
        html = template.render(info=summary)
        self.summary_pane.object = html

    def _build_molecule_info_panel(self):
        window = WindowPanel("Molecule Info", self.window_destination, width=520)
        self._molecule_info_box_pane = pn.pane.HTML()
        window.contents = [
            self._molecule_info_box_pane
        ]
        return window

    def _search_by_index(self, event=None) -> None:
        search_index = str(self._widgets["search_index_str"].value)
        doc, img_path = self.data_handler.fetch_single_doc(search_index)
        template = env.get_template("mol_info_box.html")
        if doc is not None:
            html = template.render(doc=doc, img_path=img_path)
            self._molecule_info_box_pane.object = html
            self._floating_panels["molecule_info"].open()
        else:
            pn.state.notifications.warning(
                f"Could not find a molecule with index {search_index} in the current collection..."
            )