from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from bokeh.models import HoverTool
import numpy as np
import holoviews as hv
from holoviews import Store
import holoviews.operation.datashader as hd
import pandas as pd
import panel as pn
from jinja2 import Environment, FileSystemLoader
from .base_view import BaseView
from phiplot.modules.ui.menus import *
from phiplot.modules.clustering import *
from phiplot.modules.data.handlers import *
from phiplot.modules.plotting.highlighter import Highlighter
from phiplot.modules.ui.utils import *

if TYPE_CHECKING:
    from phiplot.modules.ui.web_ui import WebUI

# Load HTML templates
env = Environment(loader=FileSystemLoader("phiplot/assets/templates"))


class ClusteringView(BaseView):
    def __init__(self, ui: WebUI):
        super().__init__(ui)
        self.title = "Clustering"

        self._clusterer = Clusterer()
        self._embedder = ClusterEmbedder()
        self._available_fps = MoleculeHandler().supported_generators

        self._legend_on = True

        self._mol_ids = None
        self._X = None
        self._labels = None
        self._n_clusters = None

        self._menus = dict(
            clustering = ClusteringMenu(self),
            appearance = ClusteringAppearanceMenu(self)
        )

        self._display_panes = dict(
            info_section = pn.pane.HTML(sizing_mode="stretch_width"),
            metrics_section = pn.pane.HTML(sizing_mode="stretch_width")
        )

        self._cluster_plot_pane = pn.pane.HoloViews(
            hv.Points(pd.DataFrame(columns=["x", "y", "cluster_label"]), kdims=["x", "y"]).opts(
                show_grid=True,
                show_legend=True
            ),
            sizing_mode="stretch_both"
        )

        self._highlighter = Highlighter()

        self.center_column = [
            self._cluster_plot_pane
        ]

        self.left_column = [
            ("Clustering Info", self._display_panes["info_section"]),
            ("Clustering Metrics", self._display_panes["metrics_section"]),
        ]

        self.right_column = []

    @property
    def has_clusters(self):
        return not self._cluster_df.empty

    @property
    def cluster_labels_df(self):
        return self._cluster_df[["mol_id", "cluster_label"]]

    def toggle_legend(self):
        self._legend_on = not self._legend_on
        self._set_shaded()

    def update_color_palette(self):
        self.cluster(recompute=False)

    def cluster(self, recompute: bool = True) -> None:
        with toggle_spinner(self._menus["clustering"].widgets["refresh_plot_spinner"]):
            if recompute:
                clustering_algo = self._menus["clustering"].clustering_algo
                embedding_algo = self._menus["clustering"].embedding_algo

                self.title = f"{clustering_algo} Clustering with {embedding_algo} Embedding"
                
                clustering_kwargs = self._menus["clustering"].clustering_params
                embedding_kwargs = self._menus["clustering"].embedding_params
                
                self._construct_X()
                self._clusterer.algorithm = clustering_algo
                self._labels = self._clusterer.cluster(self._X, **clustering_kwargs)
                self._n_clusters = len(set(self._labels))
            
                self._embedder.algorithm = embedding_algo
            
                coords = self._embedder.embed(self._X, **embedding_kwargs)

                self._cluster_df = pd.DataFrame({
                    "x": coords[:,0],
                    "y": coords[:,1],
                    "cluster_label": list(map(str, self._labels)),
                    "mol_id": self._mol_ids
                })

            if self._cluster_df is not None:
                points = hv.Points(self._cluster_df, kdims=["x", "y"], vdims=["cluster_label", "mol_id"])
                color_key = {str(i): c for i, c in enumerate(self._menus["appearance"].color_palette)}

                self._shaded_clustering = hd.spread(hd.datashade(
                    points,
                    aggregator=hd.ds.count_cat("cluster_label"),
                    color_key=color_key,
                    cnorm="log"
                ), px=10)

                self._legend = self._construct_legend(self._labels, color_key)
                self._set_shaded()
                
                self._update_info()
                self._update_metrics()

    def _set_shaded(self):
        if self._legend_on:
            obj = self._shaded_clustering * self._legend
        else:
            obj = self._shaded_clustering

        self._obj = obj.opts(
            responsive=True,
            show_grid=True,
            show_legend=True
        )
        self._cluster_plot_pane.object = self._obj

    def _construct_X(self) -> None:
        self._mol_ids = self.data_handler.indices
        try:
            X = self.data_handler.fingerprints[
                self._menus["clustering"].widgets["fingerprint_selector"].value
            ]
            self._X = np.vstack(X.values)
        except Exception:
            pn.state._notification.warning(
                "Please generate fingerprints first."
            )

    def _update_metrics(self) -> None:
        template = env.get_template("simple_table.html")
        info = {name: f"{val:.3e}" for name, val in self._clusterer.eval_metrics(self._X, self._labels).items()}
        html = template.render(info=info)
        self._display_panes["metrics_section"].object = html

    def _update_info(self) -> None:
        template = env.get_template("simple_table.html")
        info = {
            "Fingerprint": self._menus["clustering"].fingerprint,
            "Clustering Algorithm": self._menus["clustering"].clustering_algo,
            "Emebdding Algorithm": self._menus["clustering"].embedding_algo,
            "Number of Clusters": self._n_clusters,
            "Number of Datapoints": len(self._cluster_df)
        }
        html = template.render(info=info)
        self._display_panes["info_section"].object = html

    def _construct_legend(self, labels, color_key):
        unique_labels = sorted(set(labels))
        legend_points = hv.NdOverlay({
            label: hv.Points([0,0]).opts(
                color=color_key[str(label)], size=0
            ) 
            for label in unique_labels
        })
        return legend_points
    
    def _highlight(self, x, y):
        self._highlighter.highlight(x, y)
        self._cluster_plot_pane.object = (self._obj * self._highlighter.get_object()).opts(
            responsive=True,
            show_grid=True,
            show_legend=True
        )

    def _search_by_index(self, event=None):
        if self._cluster_df is None:
            pn.state.notifications.warning("No clusters to search from...")
            return
        
        search_index = str(self._widgets["search_index_str"].value)
        mol = self._cluster_df[self._cluster_df["mol_id"] == search_index]
        
        if mol.empty:
            pn.state.notifications.warning(
                f"Could not find a molecule with index {search_index} in the current clustering..."
            )
            return
        
        x, y = float(mol["x"].iloc[0]), float(mol["y"].iloc[0])
        label = str(mol["cluster_label"].iloc[0])
        pn.state.notifications.info(
            f"The molecule with index {search_index} belongs to the cluster labelled {label}."
        )
        self._highlight(x, y)