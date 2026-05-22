from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import panel as pn
from scipy.spatial import distance
from .base_menu import BaseMenu
from phiplot.modules.ui.floating import *
from phiplot.modules.clustering import *
from phiplot.modules.utils import *
from phiplot.modules.data.handlers import *
from phiplot.modules.ui.utils import *
from phiplot.modules.ui.widgets.success_spinner import SuccessSpinner

if TYPE_CHECKING:
    from phiplot.modules.ui.views.clustering_view import ClusteringView


logger = logging.getLogger(__name__)

class ClusteringMenu(BaseMenu):
    def __init__(self, parent_view: ClusteringView):
        super().__init__(
            parent_view=parent_view,
            name="Cluster",
            icon="topology-ring-2"
        )

        self._available_fps = MoleculeHandler().supported_generators
        self._clusterer = Clusterer()
        self._embedder = ClusterEmbedder()

        self._clust_param_parser = DefaultParamParser("clustering_hyperparams.json")
        self._clust_widget_constructor = WidgetConstructor(self._clust_param_parser)
        
        self._emb_param_parser = DefaultParamParser("static_embedding_hyperparams.json")
        self._emb_widget_constructor = WidgetConstructor(self._emb_param_parser)

        self._widgets.update(self._clust_widget_constructor.widgets)
        self._widgets.update(self._emb_widget_constructor.widgets)

        self._clustering_algo = "KMeans"
        self._construct_widgets()

        self._clustering_algos = self._clust_param_parser.supported
        
        self._floating_panels = dict()
        callbacks = dict()
        menu_items = []

        for algo in self._clustering_algos:
            identifier = algo.lower().replace(" ", "_")
            menu_items.append((algo, identifier))
            self._floating_panels[identifier] = self._build_cluster_panel(algo)
            callbacks[identifier] = lambda alg=algo: self._open_cluster_panel(alg)

        self._floating_panels["stratified_subsampling"] = self._build_stratified_sample_panel()
        callbacks["stratified_subsampling"] = lambda: self._floating_panels["stratified_subsampling"].open()
        
        menu_items.extend([
            None,
            ("Generate Features from Labels", "labels_to_features"),
            ("Stratified Subsampling", "stratified_subsampling")
        ])
        callbacks["labels_to_features"] = self._on_generate_features

        self.menu_items = menu_items
        self.callbacks = callbacks

    @property
    def clustering_algo(self):
        return self._clustering_algo
    
    @property
    def clustering_params(self):
        params = dict()
        for name, value in self._clust_widget_constructor.values.items():
            if self.clustering_algo in name:
                params["_".join(name.split("_")[1:-1])] = value
        return params
    
    @property
    def embedding_algo(self):
        return self._widgets["embedding_algorithm_selector"].value
    
    @property
    def fingerprint(self):
        return self._widgets["fingerprint_selector"].value

    @property
    def embedding_params(self):
        params = dict()
        for name, value in self._emb_widget_constructor.values.items():
            if self.embedding_algo in name:
                params["_".join(name.split("_")[1:-1])] = value
        return params
    
    @property
    def medoid_metric(self):
        return self._widgets["medoid_metric_selector"].value

    def _build_cluster_panel(self, algo):
        window = WindowPanel(f"{algo} Clustering", self._window_destination)

        embedding_params_section = pn.bind(
            lambda algo: self._emb_widget_constructor.layouts(algo)["two_cols"],
            algo = self._widgets["embedding_algorithm_selector"]
        )
        window.contents = [
            self._widgets["fingerprint_selector"],
            self._widgets["embedding_algorithm_selector"],
            self._widgets["medoid_metric_selector"],
            pn.Accordion(
                ("Clustering Hyperparams", self._clust_widget_constructor.layouts(algo)["two_cols"]),
                sizing_mode="stretch_width"
            ),
            pn.Accordion(
                ("Embedding Hyperparams", embedding_params_section),
                sizing_mode="stretch_width"
            ),
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["cluster_button"],
                self._widgets["refresh_plot_spinner"]
            ) 
        ]
        return window
    
    def _build_stratified_sample_panel(self):
        window = WindowPanel(f"Stratified Subsampling", self._window_destination)

        self._widgets["subsample_frac_float"] = pn.widgets.FloatInput(
            name="Subsampling fraction",
            start=0,
            end=1,
            value=0.1
        )

        self._widgets["subsample_button"] = pn.widgets.Button(
            name="Subsample", **self._styling.default_button_style
        )
        self._widgets["subsample_button"].on_click(
            lambda event: self._on_subsample()
        )

        self._widgets["subsample_spinner"] = SuccessSpinner(
            value=False, **self._styling.default_spinner_style
        )

        window.contents = [
            self._widgets["subsample_frac_float"],
            pn.Spacer(height=self._styling.default_spacer_height),
            pn.Row(
                self._widgets["subsample_button"],
                self._widgets["subsample_spinner"]
            )
        ]
        return window

    def _construct_widgets(self) -> None:
        self._widgets["refresh_plot_spinner"] = SuccessSpinner(
            value=False, **self._styling.default_spinner_style
        )

        self._widgets["fingerprint_selector"] = pn.widgets.Select(
            name="Fingerprint",
            options=self._available_fps,
            sizing_mode="stretch_width"
        )

        self._widgets["embedding_algorithm_selector"] = pn.widgets.Select(
            name="Embedding Algorithm",
            options=self._emb_param_parser.supported,
            sizing_mode="stretch_width"
        )

        self._widgets["cluster_button"] = pn.widgets.Button(
            name="Cluster", **self._styling.default_button_style
        )
        self._widgets["cluster_button"].on_click(
            lambda event: self._on_cluster()
        )

        self._widgets["labels_to_features_button"] = pn.widgets.Button(
            name="Featurize", **self._styling.default_button_style
        )

        self._widgets["medoid_metric_selector"] = pn.widgets.Select(
            name="Medoid distance metric",
            options=list(distance._METRICS.keys()),
            value="euclidean",
            sizing_mode="stretch_width"
        )

    def _open_cluster_panel(self, algo):
        self._clicked_algo = algo
        identifyer = algo.lower().replace(" ", "_")
        self._floating_panels[identifyer].open()

    def _on_cluster(self, event=None):
        with toggle_spinner(self._widgets["refresh_plot_spinner"]):
            self._clustering_algo = self._clicked_algo
            self._parent_view.cluster()

    def _on_generate_features(self, event=None):
        if self._parent_view.has_clusters:
            df = self._parent_view.cluster_labels_df
            self._data_handler.add_features(df, "mol_id")
            self._parent_view._ui.update_available_features()
            pn.state.notifications.success(
                "Cluster labels added as new features!"
            )
        else:
            pn.state.notifications.warning(
                "No clusters to generate features from..."
            )

    def _on_subsample(self, event=None):
        with toggle_spinner(self._widgets["subsample_spinner"]):
            if self._parent_view.has_clusters:
                df = self._parent_view.cluster_labels_df
                frac = self._widgets["subsample_frac_float"].value
                self._data_handler.stratified_subsampling(df, "cluster_label", "mol_id", frac)
                self._parent_view._ui.update_available_features()
                pn.state.notifications.success(
                    "Subsampling ready!"
                )
                self._parent_view._update_info()
            else:
                pn.state.notifications.warning(
                    "No clusters to subsample from..."
                )