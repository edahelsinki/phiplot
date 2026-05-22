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
        edges: list[tuple[float, float]], # Note: zip(*edges) implies a list of tuples
        counts: list[int],
        xlabel: str,
        CDF: bool = False,
        KDE: bool = False,
        relative_freq: bool = False,
        colors: dict | None = None,
        xlog: bool = False,
        ylog: bool = False
    ) -> hv.Rectangles | hv.Overlay:
        """
        Build a histogram based on provided counts and edges with log scale support.
        """

        colors = colors or {}
        fill_color = colors.get("fill", SummaryPlots.styling.plot_blue)
        line_color = colors.get("line", SummaryPlots.styling.neutral_gray)
        kde_color = colors.get("kde", "#ff0000")

        ylabel = "Frequency"
        hover_val_name = "Frequency"
        if relative_freq:
            counts = np.array(counts) / np.sum(counts) * 100
            ylabel = "Relative Frequency (%)"
            hover_val_name = "Rel. Freq %"

        title = f"Histogram of '{xlabel}' (Bins = {len(counts)} | N = {int(sum(counts))})"
        if CDF:
            counts = np.cumsum(counts)
            title = f"CDF of '{xlabel}' (Bins = {len(counts)} | N = {int(counts[-1])})"

        bin_starts, bin_ends = zip(*edges)
        bin_starts = np.array(bin_starts)
        bin_ends = np.array(bin_ends)
        
        range_tooltips = [f"[{e[0]:.3e}, {e[1]:.3e}]" for e in edges]

        if xlog:
            # Avoid log(0)
            bin_starts = np.log10(np.where(bin_starts > 0, bin_starts, 1e-12))
            bin_ends = np.log10(np.where(bin_ends > 0, bin_ends, 1e-12))
            xlabel = f"log10({xlabel})"

        bottom_val = 0
        if ylog:
            bottom_val = 1e-3 if relative_freq else 0.1
            valid_idx = np.array(counts) > 0
            bin_starts = bin_starts[valid_idx]
            bin_ends = bin_ends[valid_idx]
            counts = np.array(counts)[valid_idx]
            range_tooltips = np.array(range_tooltips)[valid_idx]

        df = pd.DataFrame({
            "left": bin_starts,
            "right": bin_ends,
            "bottom": bottom_val,
            "top": counts,
            "range": range_tooltips
        })

        hover = HoverTool(tooltips=[
            ("Range", "@range"),
            (hover_val_name, "@top")
        ])

        hist = hv.Rectangles(df, kdims=["left", "bottom", "right", "top"], vdims=["range"], label="Histogram")
        
        y_min = bottom_val if ylog else 0
        y_max = max(counts) * (10 if ylog else 1.1) # Log space needs more padding

        hist_opts = hv.opts.Rectangles(
            fill_color=fill_color,
            line_color=line_color,
            shared_axes=False,
            xlabel=xlabel,
            ylabel=ylabel,
            ylim=(y_min, y_max),
            logy=ylog,
            title=title,
            tools=[hover],
            fontscale=1.25
        )

        hist = hist.opts(hist_opts)

        if KDE:
            kde_curve = SummaryPlots.empirical_kde(edges, counts, kde_color, is_log_scale=xlog)
            return (hist * kde_curve).opts(shared_axes=False, multi_y=True)
            
        return hist

    @staticmethod
    def build_individual_box_plot(
            summary: dict[str, int | float | str],
            ylabel: str,
            log_scale: bool = False,
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

        boxplot = SummaryPlots.build_box_plot(summary, colors=colors, log_scale=log_scale)
        return boxplot.opts(
            invert_axes=True,
            yaxis=None,
            ylabel=ylabel,
            xlim=(-0.5, 1.5),
            show_legend=True,
            title=f"Property Distribution of '{ylabel}' (N = {int(summary["count"])})",
            fontscale=1.25
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
        n = 0
        for label, summary in summaries.items():
            n += int(summary["count"])
            box_plots.append(SummaryPlots.build_box_plot(summary, i*shift, notched=notched, colors=colors))
            i += 1
        xticks = [(i*1.5 + 0.5, labels[i]) for i in range(len(labels))]
        return hv.Overlay(box_plots).opts(
            hv.opts.Overlay(
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=xticks,
                shared_axes=False,
                title=f"Property Distribution of '{ylabel}' across '{xlabel}' | N = {n}",
                fontscale=1.25
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
            title=f"Bar Plot of '{xlabel}' (N = {int(sum(counts))})",
            tools=[hover],
            fontscale=1.25
        )

    @staticmethod
    def build_box_plot(
            summary: dict[str, int | float | str], 
            shift: float = 0, 
            notched: bool = False,
            colors: dict | None = None,
            log_scale: bool = False
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

        def transform(val):
            if not log_scale:
                return val
            return np.log10(max(val, 1e-30)) if isinstance(val, (int, float)) else val

        raw_std_lower = summary["mean"] - summary["std"]
        raw_std_upper = summary["mean"] + summary["std"]

        # Transform all necessary coordinates
        t_min = transform(summary["min"])
        t_25 = transform(summary["25%"])
        t_50 = transform(summary["50%"])
        t_75 = transform(summary["75%"])
        t_max = transform(summary["max"])
        t_std_low = transform(raw_std_lower)
        t_std_high = transform(raw_std_upper)

        t_25_lin = summary["25%"]
        t_75_lin = summary["75%"]
        t_25_log = np.log10(max(summary["25%"], 1e-30))
        t_75_log = np.log10(max(summary["75%"], 1e-30))

        iqr_dist_lin = t_75_lin - t_25_lin
        iqr_dist_log =  t_75_log - t_25_log
        current_iqr_dist = iqr_dist_log if log_scale else iqr_dist_lin

        count = summary.get("count", 30)

        meta = {
            "iqr_linear": f"[{t_25_lin:.3e}, {t_75_lin:.3e}]",
            "linear_outlier_bounds": f"[{t_25_lin - 1.5*iqr_dist_lin:.3e}, {t_75_lin + 1.5*iqr_dist_lin:.3e}]",
            "iqr_log": f"[{t_25_log:.3f}, {t_75_log:.3f}]",
            "log_outlier_bounds": f"[{10**(t_25_log - 1.5*iqr_dist_log):.3e}, {10**(t_75_log + 1.5*iqr_dist_log):.3e}] linear units",
            "min_max": f"[{summary['min']:.3e}, {summary['max']:.3e}]",
            "std": f"{summary['std']:.3e}",
            "median": f"{summary['50%']:.3e}"
        }

        hover = HoverTool(tooltips=[
            ("Linear IQR", "@iqr_linear"),
            ("Linear Outlier Bounds", "@linear_outlier_bounds"), # Fixed missing comma
            ("Log IQR", "@iqr_log"),
            ("Log Outlier Bounds", "@log_outlier_bounds"),
            ("min-max", "@min_max"),
            ("std", "@std"),
            ("median", "@median")
        ])

        box_opts = hv.opts.Rectangles(fill_color=iqr_fill_color, line_color=iqr_line_color, tools=[hover], fontscale=1.25)
        notch_opts = hv.opts.Polygons(line_color=iqr_line_color, show_legend=True, tools=[hover])
        min_max_opts = hv.opts.Segments(line_width=2, line_color=whiskers_color)
        std_opts = hv.opts.Segments(line_width=2, line_color=std_color, line_dash="dashed")
        median_opts = hv.opts.Segments(line_width=3, line_color=median_color)

        if notched:
            # Notch calculation happens in the transformed space
            notch_span = 1.57 * current_iqr_dist / count**0.5
            notch_lower = t_50 - notch_span
            notch_upper = t_50 + notch_span

            x, y = zip(*[
                (shift, t_25),
                (shift, notch_lower),
                (shift + 0.25, t_50), 
                (shift, notch_upper),          
                (shift, t_75),
                (shift + 1, t_75),
                (shift + 1, notch_upper),
                (shift + 0.75, t_50),
                (shift + 1, notch_lower),
                (shift + 1, t_25)   
            ])

            poly_df = pd.DataFrame({
                "x": x,
                "y": y,
                "fill": [iqr_fill_color]*len(x)
            } | meta)

            box = hv.Polygons(poly_df, kdims=["x", "y"], vdims=list(meta.keys()) + ["fill"], label="IQR").opts(color="fill").opts(notch_opts)
            median_line = hv.Segments([((shift + 0.25, t_50), (shift + 0.75, t_50))], label="median").opts(median_opts)
        else:
            box_df = pd.DataFrame({
                "left": [shift],
                "bottom": [t_25],
                "right": [shift + 1],
                "top": [t_75]
            } | meta)

            box = hv.Rectangles(box_df, kdims=["left", "bottom", "right", "top"], vdims=list(meta.keys()), label="IQR").opts(box_opts)
            median_line = hv.Segments([((shift, t_50), (shift + 1, t_50))], label="median").opts(median_opts)

        # Whiskers (Min-Max)
        cap_width = 0.5
        min_max_whiskers = hv.Segments([
            ((shift + 0.5, t_min), (shift + 0.5, t_25)),
            ((shift + 0.5 - cap_width/2, t_min), (shift + 0.5 + cap_width/2, t_min)),
            ((shift + 0.5, t_75), (shift + 0.5, t_max)),
            ((shift + 0.5 - cap_width/2, t_max), (shift + 0.5 + cap_width/2, t_max))
        ], label="min-max").opts(min_max_opts)
        
        # Whiskers (Std Dev)
        cap_width = 0.3
        std_whisker = hv.Segments([
            ((shift + 0.5, t_std_low), (shift + 0.5, t_std_high)),   
            ((shift + 0.5 - cap_width/2, t_std_low), (shift + 0.5 + cap_width/2, t_std_low)),
            ((shift + 0.5 - cap_width/2, t_std_high), (shift + 0.5 + cap_width/2, t_std_high))
        ], label="std").opts(std_opts)

        return box * min_max_whiskers * std_whisker * median_line

    @staticmethod
    def empirical_kde(edges, counts, color=None, is_log_scale=False):
        raw_edges = np.array(edges)
        starts, ends = raw_edges[:, 0], raw_edges[:, 1]
        weights = np.array(counts, dtype=float)

        if is_log_scale:
            log_starts = np.log10(np.where(starts > 0, starts, 1e-12))
            log_ends = np.log10(np.where(ends > 0, ends, 1e-12))
            log_centers = (log_starts + log_ends) / 2
            
            kde = KDEUnivariate(log_centers)
            log_range = log_ends[-1] - log_starts[0]
            kde.fit(weights=weights, fft=False, bw=max(log_range * 0.05, 1e-5))
            
            x_final = np.linspace(log_starts[0], log_ends[-1], 500)
            y_final = kde.evaluate(x_final)
            
            y_final = y_final * (np.sum(counts) / np.sum(y_final))
        else:
            bin_centers = (starts + ends) / 2
            kde = KDEUnivariate(bin_centers)
            data_range = ends[-1] - starts[0]
            kde.fit(weights=weights, fft=False, bw=max(data_range * 0.05, 1e-5))
            
            x_final = np.linspace(starts[0], ends[-1], 500)
            y_final = kde.evaluate(x_final)
            y_final = y_final * (np.sum(counts) / np.sum(y_final))

        dx = x_final[1] - x_final[0]
        area = np.sum(y_final) * dx
        if area > 0:
            y_final /= area

        df = pd.DataFrame({"x": x_final, "y": y_final})
        return hv.Curve(df, label="Weighted KDE").opts(
            line_color=color, 
            ylim=(0, None), 
            ylabel="Density",
            fontscale=1.25
        )