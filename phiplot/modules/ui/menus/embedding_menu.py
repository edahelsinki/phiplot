from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import panel as pn
from jinja2 import Environment, FileSystemLoader
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.utils import *
from phiplot.modules.ui.utils import *
from phiplot.modules.data.handlers import *
from phiplot.modules.ui.widgets import *

if TYPE_CHECKING:
    from phiplot.modules.ui.views.embedding_view import EmbeddingView

logger = logging.getLogger(__name__)

# Load HTML templates
env = Environment(loader=FileSystemLoader("phiplot/assets/templates"))


class EmbeddingMenu(BaseMenu):
    def __init__(self, parent_view: EmbeddingView):
        super().__init__(
            parent_view=parent_view,
            name="Embed",
            icon="chart-scatter"
        )

        self._embedding_data_handler = parent_view.embedding_data_handler

        self._available_fps = MoleculeHandler().supported_generators

        self._interactive_emb_param_parser = DefaultParamParser("interactive_embedding_hyperparams.json")
        self._interactive_emb_widget_construtor = WidgetConstructor(self._interactive_emb_param_parser)

        self._static_emb_param_parser = DefaultParamParser("static_embedding_hyperparams.json")
        self._static_emb_widget_construtor = WidgetConstructor(self._static_emb_param_parser)
        
        self._kernel_param_parser = DefaultParamParser("kernel_hyperparams.json")
        self._kernel_widget_construtor = WidgetConstructor(self._kernel_param_parser)

        self._construct_widgets()

        self._widgets.update(self._interactive_emb_widget_construtor.widgets)
        self._widgets.update(self._static_emb_widget_construtor.widgets)
        self._widgets.update(self._kernel_widget_construtor.widgets)

        self._interactive_embeddings = self._interactive_emb_param_parser.supported
        self._static_embeddings = self._static_emb_param_parser.supported
        self._supports_projection = ["cKPCA"]

        self.menu_items = (
            [(emb, emb.lower().replace(" ", "_")) for emb in self._static_embeddings]
            + [None]
            + [(emb, emb.lower().replace(" ", "_")) for emb in self._interactive_embeddings]
            #+ [None]
            #+ [("Project New Points", "project_new_points")]
        )

        self._floating_panels = dict()
        callbacks = dict()

        for emb in self._static_embeddings + self._interactive_embeddings:
            identifyer = emb.lower().replace(" ", "_")
            self._floating_panels[identifyer] = self._build_embedding_panel(emb)
            callbacks[identifyer] = lambda alg=emb: self._open_embed_panel(alg)

        #self._floating_panels["project"] = self._build_projection_panel()
        #callbacks["project_new_points"] = lambda: self._floating_panels["project"].open()
        
        self.callbacks = callbacks

        self._algo = "PCA"
        self._prev_fingerprint = None
        self._mpds = dict()

    @property
    def algo(self):
        return self._algo

    @property
    def params(self):
        params = dict()
        if self._algo in self._static_embeddings:
            values = self._static_emb_widget_construtor.values
        elif self._algo in self._interactive_embeddings:
            values = self._interactive_emb_widget_construtor.values
            if self._algo == "cKPCA":
                values = values | self._kernel_widget_construtor.values
        else:
            logger.error(f"Could not get params for {self._algo} embedding...")
            return dict()
        for name, value in values.items():
            if self._algo in name:
                params["_".join(name.split("_")[1:-1])] = value
        return params

    def recompute_kernel_heuristics(self, event=None) -> None:
        """
        Create defaults for the kernel parameters based on some heuristics.
        """

        fp = self._widgets["fingerprint_selector"].value

        with toggle_spinner(self._widgets["computing_kernel_heusristic_spinner"]):
            kernel = self._widgets["cKPCA_kernel_str"].value
            d = self._embedding_data_handler.fp_dimensions
            if kernel == "Polynomial":
                self._widgets["Polynomial_gamma_float"].value = 1/d
            elif kernel == "RBF":
                mpd = self._mpds.get(fp + "_euclidean", None)
                if mpd is None:
                    mpd = self._embedding_handler.median_pairwise_dist(metric="euclidean")
                    self._mpds["euclidean"] = mpd
                if mpd:
                    gamma = 1 / (2 * mpd ** 2)
                else:
                    gamma = 1.0
                self._widgets["RBF_gamma_float"].value = gamma
            elif kernel == "manhattan":
                mpd = self._mpds.get(fp + "_cityblock", None)
                if mpd is None:
                    mpd = self._embedding_handler.median_pairwise_dist(metric="cityblock")
                    self._mpds["cityblock"] = mpd
                if mpd:
                    gamma = 1 / mpd
                else:
                    gamma = 1.0
                self._widgets["Manhattan_gamma_float"].value = gamma
            elif kernel == "Hamming":
                mpd = self._mpds.get(fp + "_cityblock", None)
                if mpd is None:
                    mpd = self._embedding_handler.median_pairwise_dist(metric="cityblock")
                    self._mpds["cityblock"] = mpd
                if mpd:
                    gamma = 1 / mpd if mpd > 0 else 1.0
                else:
                    gamma = 1.0
                self._widgets["Hamming_gamma_float"].value = gamma

    def flush_mpds(self):
        self._mpds = dict()

    def _construct_widgets(self) -> None:
        self._widgets["fingerprint_selector"] = pn.widgets.Select(
            name="Fingerprint",
            options=self._available_fps,
            sizing_mode="stretch_width"
        )
        self._widgets["fingerprint_selector"].param.watch(self._on_fingerprint_selection, "value")

        self._widgets["fingerprint_selector"].value = "Morgan"

        self._widgets["computing_kernel_heusristic_spinner"] = pn.widgets.LoadingSpinner(
            value=False, **self._styling.default_spinner_style
        )

        self._widgets["embed_button"] = pn.widgets.Button(
            name="Embed", **self._styling.default_button_style
        )
        self._widgets["embed_button"].on_click(self._on_embed)

    def _on_fingerprint_selection(self, event=None) -> None:
        """
        Apply the selected fingerprint to the data handler.
        """

        fp = self._widgets["fingerprint_selector"].value

        if fp != self._prev_fingerprint:
            self.recompute_kernel_heuristics()

        if fp is not None:
            self._embedding_data_handler.fingerprint = fp
        else:
            pn.state.notifications.warning("No fingerprint selected...")
            logger.warning("Tried to set feature matrix without features.")

    def _build_embedding_panel(self, algo):
        emb_params_section = self._build_embedding_params_section(algo)

        window = WindowPanel(f"{algo} Embedding", self._window_destination)
        window.contents = [
            emb_params_section,
            self._widgets["fingerprint_selector"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["embed_button"],
                self._parent_view._widgets["main_refresh_indicator"]
            )
        ]
        return window

    def _build_projection_panel(self) -> None:
        """
        Create a panel for projecting new points into an existing embedding.
        """

        if self._data_handler.use_external_data:
            msg = "Give the index of a data point in the provided dataset that has not yet been embedded."
            self.project_new_by = pn.pane.Markdown(msg)
        else:
            self.project_new_by = pn.widgets.Select(
                name="Project by",
                options=["New Molecular Index"],
                value="New Molecular Index",
                sizing_mode="stretch_width",
            )

        self._widgets["new_index_str"] = pn.widgets.TextInput(
            name="Index", sizing_mode="stretch_width"
        )

        self._widgets["project_new_button"] = pn.widgets.Button(
            name="Project", **self._styling.default_button_style
        )
        self._widgets["project_new_button"].on_click(self._project_new_point)

        window = WindowPanel("Project New Points", self._window_destination)
        window.contents =[
            self.project_new_by,
            self._widgets["new_index_str"],
            pn.Spacer(height=self._styling.default_spacer_height),
            self._widgets["project_new_button"],
        ]
        return window
    
    def _project_new_point(self):
        return None

    def _build_embedding_params_section(self, algo) -> None:
        if algo is not None:
            if algo in self._static_embeddings:
                widget_layout = self._static_emb_widget_construtor.layouts(algo)["two_cols"]
            elif algo in self._interactive_embeddings:
                widget_layout = self._interactive_emb_widget_construtor.layouts(algo)["two_cols"]
                if algo == "cKPCA":
                    kernel_selector = self._widgets["cKPCA_kernel_str"]
                    kernel_selector.param.watch(self.recompute_kernel_heuristics, "value")
                    kernel_params_section = pn.bind(
                        lambda kernel: self._kernel_widget_construtor.layouts(kernel)["two_cols"],
                        kernel = kernel_selector
                    )
                    widget_layout.append(
                        pn.Row(
                            pn.Accordion(
                                ("Kernel Hyperparams", kernel_params_section),
                                sizing_mode="stretch_width"
                            ),
                            self._widgets["computing_kernel_heusristic_spinner"]
                        )
                    )

        return pn.Accordion(
            ("Embedding Hyperparams", widget_layout),
            sizing_mode="stretch_width"
        )
    
    def _open_embed_panel(self, algo):
        self._clicked_algo = algo
        identifyer = algo.lower().replace(" ", "_")
        self._floating_panels[identifyer].open()

    def _on_embed(self, event=None):
        self._embedding_data_handler.fingerprint = self._widgets["fingerprint_selector"].value
        self._algo = self._clicked_algo
        self._parent_view.embed()