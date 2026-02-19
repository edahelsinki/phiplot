from bokeh.models import HoverTool
import numpy as np
import holoviews as hv
import pandas as pd
import scipy.stats as sts
from statsmodels.nonparametric.kde import KDEUnivariate
from phiplot.modules.ui.styling.styling import Styling

class SummaryPlots:
    styling = Styling()

    @staticmethod
    def build_histogram(
            edges: list[float | int],
            counts: list[int],
            xlabel: str,
            CDF: bool = False,
            KDE: bool = False,
            relative_freq: bool = False,
            colors: dict | None = None
        ) -> hv.Rectangles | hv.Overlay:
        """
        Build a histogram based on provided counts and edges. Optinally supports
        the creation of a CDF plot and the inclusion of a KDE curve.

        Args:
            edges (list[float | int]): The edges of the bins.
            counts (list[ints]): The count of each bin.
            xlabel (str): Label for the x-axis.
            CDF (bool): If True, use cumulative counts. Defaults to False.
            KDE (bool): If True, include KDE curve. Defaults to False.
            relative_freq (bool): If True, use relative frequencies. Defaults to False.
            colors (dict | None): Color settings to use. If None, defaults will be used. Defaults to None.

        Returns:
            (hv.Rectangles | hv.Overlay): The fully constructed histogram plot.
        """

        colors = colors or {}
        fill_color = colors.get("fill", SummaryPlots.styling.plot_blue)
        line_color = colors.get("line", SummaryPlots.styling.neutral_gray)
        kde_color = colors.get("kde", "#ff0000")

        if KDE:
            kde_curve = SummaryPlots.empirical_kde(edges, counts, kde_color)

        ylabel = "Frequency"
        hover_count = "@top"
        if relative_freq:
            counts = np.array(counts)/np.sum(counts)*100
            ylabel = "Relative Frequency (%)"
            hover_count = "@top %"

        title = "Distribution"
        if CDF:
            cumulative_counts = []
            running_sum = 0
            for count in counts:
                running_sum += count
                cumulative_counts.append(running_sum)
            counts = cumulative_counts
            title = "Cumulative Distribution"

        bin_starts, bin_ends = zip(*edges)
        bin_starts = np.array(bin_starts)
        bin_ends = np.array(bin_ends)
        max_count = max(counts)
        range = [f"[{e[0]:.3e}, {e[1]:.3e}]" for e in edges]
        
        max_count = max(counts)
        
        df = pd.DataFrame({
            "left": bin_starts,
            "right": bin_ends,
            "bottom": 0,
            "top": counts,
            "range": range
        })

        hover = HoverTool(tooltips=[
            ("Range", "@range"),
            ("Frequency", hover_count)
        ])

        hist = hv.Rectangles(df, kdims=["left", "bottom", "right", "top"], vdims=["range"], label="Histogram").opts(
            fill_color=fill_color,
            line_color=line_color,
            shared_axes=False,
            xlabel=xlabel,
            ylabel=ylabel,
            ylim=(0, max_count * 1.1),
            title=title,
            tools=[hover]
        )

        if KDE:
            return (hist * kde_curve).opts(
                shared_axes=False,
                multi_y=True
            )
        return hist

    @staticmethod
    def build_individual_box_plot(
            summary: dict[str, int | float | str],
            ylabel: str,
            colors: dict | None = None
        ) -> hv.Overlay:
        """
        Build an individual box plot based on precomputed statistics.

        Args:
            summary (dict[str, int | float | str]): The precomputed summary statistics.
            ylabel (str): Label for the y-axis.
            colors (dict | None): Color settings to use. If None, defaults will be used. Defaults to None.

        Returns:
            hv.Overlay: The fully constructed box plot.
        """

        colors = colors or {}

        boxplot = SummaryPlots.build_box_plot(summary, colors=colors)
        return boxplot.opts(
            invert_axes=True,
            yaxis=None,
            ylabel=ylabel,
            xlim=(-0.5, 1.5),
            show_legend=True,
            title="Box Plot"
        )

    @staticmethod
    def build_comparison_box_plot(
            summaries: dict[str, dict],
            xlabel: str,
            ylabel: str,
            notched: bool = False,
            colors: dict | None = None
        ) -> hv.Overlay:
        """
        Build a set of box plots for a categorical field based on 
        precomputed statistics on some comparison field.

        Args:
            summaries (dict[str, dict]): The precomputed summary statistics.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            notched (bool): If True, make notched boxes. Defaults to False.
            colors (dict | None): Color settings to use. If None, defaults will be used. Defaults to None.

        Returns:
            hv.Overlay: The fully constructed box plot.
        """

        colors = colors or {}

        shift = 1.5
        box_plots = []
        i = 0
        labels = list(summaries.keys())
        for label, summary in summaries.items():
            box_plots.append(SummaryPlots.build_box_plot(summary, i*shift, notched=notched, colors=colors))
            i += 1
        xticks = [(i*1.5 + 0.5, labels[i]) for i in range(len(labels))]
        return hv.Overlay(box_plots).opts(
            hv.opts.Overlay(
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=xticks,
                shared_axes=False,
                title="Box Plot"
            )
        )

    @staticmethod
    def build_bar_plot(
            labels: list[str],
            counts: list[int],
            xlabel: str,
            relative_freq: bool = False,
            colors: dict | None = None
        ) -> hv.Bars:
        """
        Build a bar plot based on provided counts and labels of a categorical field.

        Args:
            labels (list[float | int | str]): The lables of the categories.
            counts (list[ints]): The count of each category.
            xlabel (str): Label for the x-axis.
            relative_freq (bool): If True, use relative frequencies. Defaults to False.
            colors (dict | None): Color settings to use. If None, defaults will be used. Defaults to None.

        Returns:
            hv.Bars: The fully constructed bar plot.
        """

        colors = colors or {}
        fill_color = colors.get("fill", SummaryPlots.styling.plot_blue)
        line_color = colors.get("line", SummaryPlots.styling.neutral_gray)

        ylabel = "Frequency"
        hover_count = "@count"
        if relative_freq:
            counts = np.array(counts)/np.sum(counts)*100
            ylabel = "Relative Frequency (%)"
            hover_count = "@count %"

        hover = HoverTool(tooltips=[
            (xlabel, "@category"),
            ("Frequency", hover_count)
        ])

        df = pd.DataFrame({
            "count": counts,
            "category": labels
        })

        bars = hv.Bars(df, kdims=["category"], vdims=["count"])
        return bars.opts(
            fill_color=fill_color,
            line_color=line_color,
            xlabel=xlabel,
            ylabel=ylabel,
            title="Distribution",
            tools=[hover]
        )

    @staticmethod
    def build_box_plot(
            summary: dict[str, int | float | str], 
            shift: float = 0, 
            notched: bool = False,
            colors: dict | None = None
        ) -> hv.Overlay:
        """
        Build an individual box plot based on precomputed statistics.

        Args:
            summary (dict[str, int | float | str]): The precomputed summary statistics.
            shift (float): The amount to shift along the x-axis. Defaults to 0.
            notched (bool): If True, a notched box plot is created. Defaults to False.
            colors (dict | None): Color settings to use. If None, defaults will be used. Defaults to None.

        Returns:
            hv.Overlay: The fully constructed box plot.
        """

        colors = colors | {}
        iqr_fill_color = colors.get("iqr_fill", SummaryPlots.styling.plot_blue)
        iqr_line_color = colors.get("iqr_line", SummaryPlots.styling.neutral_gray)
        whiskers_color = colors.get("whiskers", SummaryPlots.styling.neutral_gray)
        median_color = colors.get("median", "#00ff00")
        std_color = colors.get("std", "#ff0000")

        std_lower = summary["mean"] - summary["std"]
        std_upper = summary["mean"] + summary["std"]

        count = summary.get("count", 30)
        iqr = summary["75%"] - summary["25%"]
        median = summary["50%"]

        hover = HoverTool(tooltips=[
            ("IQR", "@iqr"),
            ("min-max", "@min_max"),
            ("std", "@std"),
            ("median", "@median")
        ])

        # Styling options
        box_opts = hv.opts.Rectangles(fill_color=iqr_fill_color, line_color=iqr_line_color, tools=[hover])
        notch_opts = hv.opts.Polygons(line_color=iqr_line_color, show_legend=True, tools=[hover])
        min_max_opts = hv.opts.Segments(line_width=2, line_color=whiskers_color)
        std_opts = hv.opts.Segments(line_width=2, line_color=std_color, line_dash="dashed")
        median_opts = hv.opts.Segments(line_width=3, line_color=median_color)

        # Interquartile box and median line
        meta = {
            "iqr": f"[{summary["25%"]:.3e}, {summary["75%"]:.3e}]",
            "min_max": f"[{summary["min"]:.3e}, {summary["max"]:.3e}]",
            "std": f"{summary["std"]:.3e}",
            "median": f"{summary["50%"]:.3e}"
        }

        if notched:
            notch_span = 1.57 * iqr / count**0.5
            notch_lower = median - notch_span
            notch_upper = median + notch_span

            x, y = zip(*[
                (shift, summary['25%']),
                (shift, notch_lower),
                (shift + 0.25, median), 
                (shift, notch_upper),          
                (shift, summary['75%']),
                (shift + 1, summary['75%']),
                (shift + 1, notch_upper),
                (shift + 0.75, median),
                (shift + 1, notch_lower),
                (shift + 1, summary['25%'])   
            ])

            poly_df = pd.DataFrame({
                "x": x,
                "y": y,
                "fill": [iqr_fill_color]*len(x) # force same color for all boxes
            } | meta)

            box = hv.Polygons(poly_df, kdims=["x", "y"], vdims=list(meta.keys()) + ["fill"], label="IQR").opts(color="fill").opts(notch_opts)
            median_line = hv.Segments([((shift + 0.25, median), (shift + 0.75, median))], label="median").opts(median_opts)
        else:
            box_df = pd.DataFrame({
                "left": [shift],
                "bottom": [summary['25%']],
                "right": [shift + 1],
                "top": [summary['75%']]
            } | meta)

            box = hv.Rectangles(box_df, kdims=["left", "bottom", "right", "top"], vdims=list(meta.keys()), label="IQR").opts(box_opts)
            median_line = hv.Segments([((shift, median), (shift + 1, median))], label="median").opts(median_opts)

        # Whiskers
        cap_width = 0.5
        min_max_whiskers = hv.Segments([
            ((shift + 0.5, summary['min']), (shift + 0.5, summary['25%'])),
            ((shift + 0.5 - cap_width/2, summary['min']), (shift + 0.5 + cap_width/2, summary['min'])),
            ((shift + 0.5, summary['75%']), (shift + 0.5, summary['max'])),
            ((shift + 0.5 - cap_width/2, summary['max']), (shift + 0.5 + cap_width/2, summary['max']))
        ], label="min-max").opts(min_max_opts)
        
        cap_width = 0.3
        std_whisker = hv.Segments([
            ((shift + 0.5, std_lower), (shift + 0.5, std_upper)),   
            ((shift + 0.5 - cap_width/2, std_lower), (shift + 0.5 + cap_width/2, std_lower)),
            ((shift + 0.5 - cap_width/2, std_upper), (shift + 0.5 + cap_width/2, std_upper))
        ], label="std").opts(std_opts)

        # Combine elements
        return box * min_max_whiskers * std_whisker * median_line

    @staticmethod
    def old_empirical_kde(
            edges: list[float | int ],
            counts: list[int],
            color: str | None = None
        ) -> hv.Curve:
        """
        Estimate a smooth kernel density from binned sample data using a Gaussian kernel.

        The underlying continuous probability density function is approximated
        from discrete binned data (histogram). The procedure expands bin counts into
        a sample set, adds small Gaussian jitter to bin centers, and computes a kernel
        density estimate (KDE).

        Args:
            edges (list[float | int]): The edges of the bins.
            counts (list[ints]): The count of each bin.
            color (str): Line color setting to use.

        Returns:
            hv.Curve: The fully constructed curve plot.
        """
        
        e = np.array([start for start, end in edges] + [edges[-1][1]])
        h = np.array(counts)
        n = h.sum()
        bin_centers = (e[:-1] + e[1:]) / 2

        repeat_factor = max(1, int(n * 5 / h.sum()))
        samples = np.repeat(bin_centers, (h * repeat_factor).astype(int))
        bandwidth = (e[-1] - e[0]) / (len(edges) * 2)  
        jitter = np.random.normal(scale=bandwidth, size=len(samples))
        samples = samples + jitter
        rkde = sts.gaussian_kde(samples)

        x = np.linspace(e.min(), e.max(), 100)
        y = rkde.pdf(x)
        y = y/np.sum(y)
        df = pd.DataFrame({"x": x, "y": y})

        return hv.Curve(df, label="Empirical KDE\n(Gaussian)").opts(
            line_color=color,
            shared_axes=False,
            ylabel="Probability",
            ylim=(0, 1.1*max(y))
        )
    
    @staticmethod
    def empirical_kde(edges, counts, color=None):
        e = np.array([start for start, _ in edges] + [edges[-1][1]], dtype=float)
        bin_centers = (e[:-1] + e[1:]) / 2
        weights = np.array(counts, dtype=float)
        
        kde = KDEUnivariate(bin_centers.astype(float))
        kde.fit(weights=weights, fft=False, bw='scott')

        x = np.linspace(e.min(), e.max(), 200)
        y = kde.evaluate(x)
        
        y = y / y.sum() 
        
        df = pd.DataFrame({"x": x, "y": y})
        
        return hv.Curve(df, label="Weighted KDE").opts(
            line_color=color,
            ylabel="Probability",
            ylim=(0, 1.1 * max(y))
        )