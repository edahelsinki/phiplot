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
from scipy.spatial import distance
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
            metrics_section = self._create_metrics_section(),
        )

        self._cluster_plot_pane = pn.pane.HoloViews(
            hv.Points(pd.DataFrame(columns=["x", "y", "cluster_label"]), kdims=["x", "y"]).opts(
                show_grid=True,
                show_legend=True,
                xlabel="Latent dimension 1",
                ylabel="Latent dimension 2"
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
        clustering_algo = self._menus["clustering"].clustering_algo
        embedding_algo = self._menus["clustering"].embedding_algo

        self.title = f"{clustering_algo} Clustering with {embedding_algo} Embedding"

        clustering_kwargs = self._menus["clustering"].clustering_params
        embedding_kwargs = self._menus["clustering"].embedding_params
        
        if recompute:
            self._construct_X()
            self._clusterer.algorithm = clustering_algo
            self._clusterer.fit(self._X, medoid_metric=self._menus["clustering"].medoid_metric, **clustering_kwargs)
            self._embedder.algorithm = embedding_algo
            self._embedder.embed(self._clusterer.X, self._clusterer.centroids, **embedding_kwargs)
            cluster_metrics = self._clusterer.eval_clusterwise_metrics()

        self._labels = self._clusterer.labels
        self._n_clusters = len(set(self._labels))
        coords = self._embedder.X

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
        else:
            return
        
        metrics_df = pd.DataFrame.from_dict(cluster_metrics, orient='index')
        metric_vdims = list(metrics_df.columns)

        centroid_df = pd.DataFrame({
            "x": self._embedder.centroids[:, 0],
            "y": self._embedder.centroids[:, 1],
            "Label": [str(i) for i in range(len(self._embedder.centroids))]
        })

        centroid_df["Label"] = centroid_df["Label"].astype(str)
        metrics_df["Label"] = metrics_df["Label"].astype(str)
        full_centroid_df = pd.merge(centroid_df, metrics_df, on="Label")

        centroid_hover = HoverTool(tooltips=[
            ("Cluster", "@Label"),
            ("Number of Molecules", "@count"),
            ("Silhouette Coefficient", "@silhouette_coeff{0.0000a}"),
            ("Diameter", "@diameter{0.0000}"),
            ("Radius (Avg dist to medoid)", "@radius{0.0000}"),
            ("Max Radius (Max dist to medoid)", "@max_radius{0.0000}"),
            ("Intracluster Variance", "@intraclust_variance{0.0000}"),
            ("Medoid Separation Ratio", "@medoid_sep_ratio{0.0000}")
        ])

        centroids_plot = hv.Points(full_centroid_df, kdims=["x", "y"], vdims=["Label"] + metric_vdims).opts(
            color="Label",
            cmap=color_key,
            marker="circle_cross",
            size=30,
            line_color="black",
            line_width=4,
            show_legend=False,
            legend_cols=0,
            tools=[centroid_hover]
        )

        primary_medoid_indices = []
        seen_clusters = set()

        for idx in self._clusterer.medoid_idx:
            cluster_id = self._labels[idx]
            if cluster_id not in seen_clusters:
                primary_medoid_indices.append(idx)
                seen_clusters.add(cluster_id)

        medoid_df = self._cluster_df.loc[primary_medoid_indices].copy()

        html_cache = {}
        for label, idx in zip(medoid_df["cluster_label"], primary_medoid_indices):
            if idx not in html_cache:
                html_content = self._gen_medoid_info(label, idx)
                html_cache[idx] = html_content

        medoid_df = medoid_df[~medoid_df.index.duplicated(keep='first')]
        medoid_df["hover_html"] = medoid_df.index.map(html_cache)

        medoid_hover = HoverTool(tooltips="@hover_html{safe}")
        
        medoids_plot = hv.Points(medoid_df, kdims=["x", "y"], vdims=["cluster_label", "mol_id", "hover_html"]).opts(
            color="cluster_label",
            cmap=color_key,
            marker="star_dot",
            size=30,
            line_color="black",
            line_width=4,
            show_legend=False,
            legend_cols=0,
            tools=[medoid_hover]
        )

        self._final_plot = self._shaded_clustering * centroids_plot * medoids_plot
        self._legend = self._construct_legend(self._labels, color_key)
        self._set_shaded()
        
        if recompute:
            self._update_info()
            self.update_metrics_display()

    def _set_shaded(self):
        data_plot = self._final_plot.opts(
            hv.opts.RGB(show_legend=False), 
            hv.opts.Points(show_legend=False)
        )

        x_min, x_max = self._cluster_df['x'].min(), self._cluster_df['x'].max()
        y_min, y_max = self._cluster_df['y'].min(), self._cluster_df['y'].max()

        if self._legend_on:
            dummy_x, dummy_y = -1000, -1000

            medoid_legend = hv.Points([(dummy_x, dummy_y)], label="Medoid").opts(
                marker="star_dot",
                size=15,
                color="white",
                line_color="black"
            )

            centroid_legend = hv.Points([(dummy_x, dummy_y)], label="Centroid").opts(
                marker="circle_cross",
                size=15,
                color="white",
                line_color="black"
            )

            obj = data_plot * self._legend * medoid_legend * centroid_legend

        self._obj = obj.opts(
            xlim=(x_min, x_max),
            ylim=(y_min, y_max),
            xlabel="Latent dimension 1",
            ylabel="Latent dimension 2",
            responsive=True,
            show_grid=True,
            show_legend=True,
            toolbar='above',
            active_tools=['wheel_zoom', 'pan']
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

    def _update_info(self) -> None:
        template = env.get_template("simple_table.html")
        info = {
            "Fingerprint": self._menus["clustering"].fingerprint,
            "Clustering Algorithm": self._menus["clustering"].clustering_algo,
            "Emebdding Algorithm": self._menus["clustering"].embedding_algo,
            "Number of Clusters": self._n_clusters,
            "Number of Molecules": len(self._cluster_df),
            "Subsample Size": self.data_handler.n_data_points
        }
        html = template.render(info=info)
        self._display_panes["info_section"].object = html

    def _construct_legend(self, labels, color_key):
        unique_labels = sorted(set(list(map(str, labels))))
        legend_points = hv.NdOverlay({
            label: hv.Points([0, 0]).opts(
                color=color_key[label], size=0
            ) 
            for label in unique_labels
        }, kdims='Cluster')
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

    def _gen_medoid_info(self, label, medoid_idx, event=None) -> str | None:
        mol_idx = self._mol_ids[medoid_idx]
        doc, img_path = self.data_handler.fetch_single_doc(mol_idx)
        doc = {"Cluster": label} | doc
        template = env.get_template("mol_info_box.html")
        if doc is not None:
            html = template.render(doc=doc, img_path=img_path)
            return html

    def _create_metrics_section(self) -> pn.Column:
        """
        Create and configure the metrics evaluation section.
        """

        self._widgets["recompute_metrics_button"] = pn.widgets.Button(
            name="Recompute", **self.styling.default_button_style
        )

        self._widgets["computing_metrics_spinner"] = pn.indicators.LoadingSpinner(
            **self.styling.default_spinner_style
        )

        self._widgets["metrics_distance_selector"] = pn.widgets.Select(
            name="Distance Measure",
            options=sorted(list(distance._METRICS.keys())),
            value="cosine",
            sizing_mode="stretch_width",
        )

        self._widgets["recompute_metrics_button"].on_click(self.update_metrics_display)

        self._widgets["metrics_display"] = pn.pane.HTML(sizing_mode="stretch_width")

        return pn.Column(
            self._widgets["metrics_display"],
            self._widgets["metrics_distance_selector"],
            pn.Row(
                self._widgets["recompute_metrics_button"],
                self._widgets["computing_metrics_spinner"]
            )
        )
    
    def update_metrics_display(self, event=None):
        """
        Update the embedding metrics display.

        Renders metrics in HTML with with threshold-based color coding.
        """

        with toggle_spinner(self._widgets["computing_metrics_spinner"]):
            template = env.get_template("metrics_display.html")

            clustering_metrics = self._clusterer.eval_global_metrics(
                self._widgets["metrics_distance_selector"].value
            )

            N = self._clusterer.N

            relative_metrics = ["Calinski-Harabasz Index"]
            

            direction_map = {
                "Silhouette Index": "Higher is better",
                "Calinski-Harabasz Index": "Higher is better",
                "Davies-Bouldin Index": "Lower is better",
            }

            threshold_color_map = {
                "Silhouette Index": [0.25, 0.5, 0.7, 1.0],
                "Davies-Bouldin Index": [2.0, 1.5, 1.0, 0.5],
            }

            palette = ["#440154", "#3b528b", "#21908d", "#5dc963"]
            palette_labels = ["Poor", "Fair", "Good", "Excellent"]
            relative_color = "#7f8c8d"

            metrics = []
            for metric, value in clustering_metrics.items():
                remark = direction_map.get(metric, "")
                
                # Handle Relative Metrics
                if metric in relative_metrics:
                    metrics.append((metric, value, relative_color, "white", remark))
                    continue

                # Handle Absolute Metrics
                thresholds = threshold_color_map.get(metric)
                if not thresholds:
                    metrics.append((metric, value, "white", "black", remark))
                    continue

                reverse = thresholds[0] > thresholds[-1]
                try:
                    if not reverse:
                        idx = next(i for i, x in enumerate(thresholds) if value < x)
                    else:
                        idx = next(i for i, x in enumerate(thresholds) if x < value)
                    color = palette[idx]
                except StopIteration:
                    color = palette[-1]
                    idx = 3

                font_color = "white" if idx < 3 else "black"
                metrics.append((metric, value, color, font_color, remark))

            legend = list(zip(palette, palette_labels))
            legend.append((relative_color, "Relative (Comparative Only)"))

            distance_measure = self._widgets["metrics_distance_selector"].value
            
            html = template.render(
                metrics=metrics, 
                legend=legend, 
                distance_measure=distance_measure
            )
            self._widgets["metrics_display"].object = html