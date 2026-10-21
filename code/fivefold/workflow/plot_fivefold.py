"""Publication-quality diagnostic panels from aggregate CV metric tables.

No observation-level inputs are read.  Every mark is a saved national score for
one fold or repeat.  Intervals show the actual Q25--Q75, never confidence limits.
No points, including very negative NSE, are clipped from the axes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from _paths import work_root
ROOT = work_root()
if "--use-workspace-deps" in sys.argv:
    sys.path.insert(0, str(ROOT / "deps"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

DOMAINS = ["within", "above", "below", "all"]
DOMAIN_LABELS = {"within": "Within range", "above": "Above range", "below": "Below range", "all": "All observations"}
METHODS = ["full", "fixed_reference_n", "beta_fixed_1", "n_only_fixed_geometry",
           "no_monotonic", "corrected_fixed", "observed_fixed"]
METHOD_LABELS = {
    "full": "All four parameters calibrated",
    "fixed_reference_n": "Fixed roughness n",
    "beta_fixed_1": r"Fixed $\beta$ = 1",
    "n_only_fixed_geometry": "Only n calibrated; fixed geometry",
    "no_monotonic": "Observed width; calibrated parameters",
    "corrected_fixed": "Corrected width; fixed parameters",
    "observed_fixed": "Observed width; fixed parameters",
}
BLUE, ORANGE, CHARCOAL = "#2166AC", "#C66A19", "#28323C"


def read(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, _position=None):
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        power=int(np.floor(np.log10(abs(value))))
        coefficient=abs(value)/(10.0**power)
        sign='-' if value<0 else ''
        if np.isclose(coefficient,1):
            return rf"${sign}10^{{{power}}}$"
        return rf"${sign}{coefficient:g}\times10^{{{power}}}$"
    if abs(value) >= 10:
        return f"{value:,.0f}"
    return f"{value:g}"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#A5ADB4", "axes.labelcolor": CHARCOAL,
        "text.color": CHARCOAL, "xtick.color": CHARCOAL, "ytick.color": CHARCOAL,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def interval(ax, values, position, color=BLUE, horizontal=False, jitter=.065):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    offsets = np.linspace(-jitter, jitter, len(a)) if len(a) > 1 else np.zeros(1)
    q25, median, q75 = np.quantile(a, [.25, .5, .75], method="linear")
    if horizontal:
        ax.scatter(a, position + offsets, s=22, color=color, alpha=.55,
                   edgecolors="white", linewidths=.45, zorder=3)
        ax.plot([q25, q75], [position, position], color=color, linewidth=3, zorder=4)
        ax.plot([median, median], [position - .12, position + .12], color=CHARCOAL, linewidth=2.2, zorder=5)
    else:
        ax.scatter(position + offsets, a, s=25, color=color, alpha=.6,
                   edgecolors="white", linewidths=.45, zorder=3)
        ax.plot([position, position], [q25, q75], color=color, linewidth=3, zorder=4)
        ax.plot([position - .13, position + .13], [median, median], color=CHARCOAL, linewidth=2.2, zorder=5)
    return dict(n=len(a), median=float(median), q25=float(q25), q75=float(q75))


def choose_scale(ax, values, metric, horizontal=False):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    axis = ax.xaxis if horizontal else ax.yaxis
    setter = ax.set_xscale if horizontal else ax.set_yscale
    if metric == "NSE" and len(a) and (a.min() < -2 or a.max() > 2):
        setter("symlog", linthresh=1, linscale=1.3, base=10)
        scale = "symlog (linear for |NSE| <= 1)"
    elif metric == "RMSE" and len(a) and a.min() > 0 and a.max() / a.min() > 12:
        setter("log", base=10)
        scale = "log"
    else:
        scale = "linear"
    axis.set_major_formatter(FuncFormatter(number))
    if metric == "NSE":
        if horizontal:
            ax.axvline(0, color="#7A8288", linewidth=.7, linestyle="--", zorder=1)
            ax.axvline(1, color="#BEC4C9", linewidth=.6, linestyle=":", zorder=1)
        else:
            ax.axhline(0, color="#7A8288", linewidth=.7, linestyle="--", zorder=1)
            ax.axhline(1, color="#BEC4C9", linewidth=.6, linestyle=":", zorder=1)
    ax.margins(x=.07 if horizontal else .12, y=.12 if horizontal else .1)
    ax.grid(axis="x" if horizontal else "y", color="#E2E6E9", linewidth=.6, zorder=0)
    return scale


def legend_items():
    return [Line2D([], [], linestyle="none", marker="o", markersize=5, color=BLUE, alpha=.6, label="One fold / repeat"),
            Line2D([], [], linewidth=3, color=BLUE, label="Q25–Q75"),
            Line2D([], [], linestyle="none", marker="_", markersize=11, markeredgewidth=2, color=CHARCOAL, label="Median")]


def save_figure(fig, output, name, dpi, manifest, info):
    if manifest.get("analysis_status", "UNKNOWN") != "PASS":
        fig.text(.99, 1.055, f"Analysis status: {manifest.get('analysis_status', 'UNKNOWN')}",
                 ha="right", va="top", fontsize=7.5, color="#A52E2E")
    for extension in ["png", "pdf"]:
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight", pad_inches=.13)
        info.setdefault("files", []).append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
    manifest["figures"][name] = info
    plt.close(fig)


def overview(scores, output, dpi, manifest):
    z = scores[scores.scheme.eq("wse_stratified_5fold") & scores.support.eq("native") &
               scores.method.eq("full") & scores.domain.isin(DOMAINS)].copy()
    if z.empty:
        manifest["skipped"]["fivefold_validation_overview"] = "No new full-model fold scores."
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"wspace": .27})
    scale_names = {}
    for j, (ax, metric) in enumerate(zip(axes, ["NSE", "RMSE"])):
        for i, domain in enumerate(DOMAINS):
            a = z[z.domain.eq(domain)]
            interval(ax, a[metric], i)
        scale_names[metric] = choose_scale(ax, z[metric], metric)
        ax.set_xticks(range(len(DOMAINS)), [DOMAIN_LABELS[x] for x in DOMAINS], rotation=12, ha="right")
        ax.set_ylabel("Nash–Sutcliffe efficiency" if metric == "NSE" else r"RMSE (m$^3$ s$^{-1}$)")
        ax.set_title(f"({'ab'[j]})  Full model: {metric}", loc="left", fontweight="bold")
        if scale_names[metric] != "linear":
            ax.text(.02, .03, scale_names[metric], transform=ax.transAxes, fontsize=8, va="bottom")
    fig.legend(handles=legend_items(), loc="lower center", bbox_to_anchor=(.5, -.015), ncol=3, frameon=False)
    fig.suptitle("Standard five-fold WSE-stratified cross-validation", x=.08, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.text(.08, -.065, "Native full-model observations. Ranges refer to each fold's calibration WSE. Intervals show quartiles, not confidence limits.", fontsize=8.3)
    fig.subplots_adjust(bottom=.22)
    save_figure(fig, output, "fivefold_validation_overview", dpi, manifest,
                dict(source="cross_gauge_metrics_by_split.csv", support="native", method="full", scales=scale_names,
                     marks="One national metric per fold; actual Q25–Q75; median", rows=len(z)))


def method_comparison(scores, output, dpi, manifest):
    z = scores[scores.scheme.eq("wse_stratified_5fold") & scores.support.eq("all_methods_common") &
               scores.method.isin(METHODS) & scores.domain.isin(["all", "within"])].copy()
    if z.empty:
        manifest["skipped"]["fivefold_method_comparison"] = "No all-13-method common-sample fold scores."
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), sharey=True,
                             gridspec_kw={"wspace": .23, "hspace": .33})
    scales = {}
    for row, domain in enumerate(["all", "within"]):
        for col, metric in enumerate(["NSE", "RMSE"]):
            ax = axes[row, col]
            a = z[z.domain.eq(domain)]
            for i, method in enumerate(METHODS):
                interval(ax, a[a.method.eq(method)][metric], i, color=BLUE if method == "full" else "#738C9E", horizontal=True)
            scales[f"{domain}_{metric}"] = choose_scale(ax, z[metric], metric, horizontal=True)
            ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[x] for x in METHODS])
            ax.set_ylim(len(METHODS) - .6, -.6)
            ax.set_xlabel("NSE" if metric == "NSE" else r"RMSE (m$^3$ s$^{-1}$)")
            ax.set_title(f"({'abcd'[row*2+col]})  {DOMAIN_LABELS[domain]}: {metric}", loc="left", fontweight="bold")
            if scales[f"{domain}_{metric}"] != "linear":
                ax.text(.98, .02, scales[f"{domain}_{metric}"], transform=ax.transAxes, fontsize=8, ha="right", va="bottom")
        # Common limits preserve magnitude comparisons between domain panels.
    for col in range(2):
        bounds = [ax.get_xlim() for ax in axes[:, col]]
        shared = (min(x[0] for x in bounds), max(x[1] for x in bounds))
        for ax in axes[:, col]:
            ax.set_xlim(shared)
    fig.suptitle("Width treatment and hydraulic parameter ablations", x=.065, ha="left", fontsize=13, fontweight="bold", y=1.015)
    fig.legend(handles=legend_items(), loc="lower center", bbox_to_anchor=(.58, -.01), ncol=3, frameon=False)
    fig.text(.065, -.05, "Identical observations shared by all 13 methods; domains follow the full model. Dots: five fold scores. Intervals: Q25–Q75, not confidence limits.", fontsize=8.4)
    fig.subplots_adjust(left=.29, right=.99, bottom=.1)
    save_figure(fig, output, "fivefold_method_comparison", dpi, manifest,
                dict(source="cross_gauge_metrics_by_split.csv", support="all_methods_common", methods=METHODS,
                     scales=scales, rows=len(z), common_axis_limits_across_domains=True))


def old_new_comparison(scores, tables, output, dpi, manifest):
    intersection = tables / "old_new_intersection_metrics_by_split.csv"
    if intersection.exists():
        z = read(intersection)
        z = z[z.support.eq("old_new_common_stations") & z.method.eq("full") & z.domain.isin(DOMAINS)]
        source = intersection.name
        support_note = "Common station set. Training fractions and partitions differ (80/20 versus 70/30)."
    else:
        z = scores[scores.support.eq("native") & scores.method.eq("full") & scores.domain.isin(DOMAINS)]
        source = "cross_gauge_metrics_by_split.csv"
        support_note = "Native samples differ between designs; training fractions and partitions also differ."
    required = ["wse_stratified_5fold", "wse_stratified_10_repeats"]
    if not set(required).issubset(set(z.scheme)):
        manifest["skipped"]["fivefold_vs_ten_repeats"] = "Both new five-fold and old ten-repeat metrics are required."
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.65), gridspec_kw={"wspace": .28})
    scales = {}
    groups = [("wse_stratified_5fold", "Standard five-fold (80/20)", BLUE, -.17),
              ("wse_stratified_10_repeats", "Ten repeated splits (70/30)", ORANGE, .17)]
    for j, (ax, metric) in enumerate(zip(axes, ["NSE", "RMSE"])):
        for scheme, label, color, offset in groups:
            for i, domain in enumerate(DOMAINS):
                values = z[z.scheme.eq(scheme) & z.domain.eq(domain)][metric]
                interval(ax, values, i + offset, color=color, jitter=.075)
        scales[metric] = choose_scale(ax, z[metric], metric)
        ax.set_xticks(range(len(DOMAINS)), [DOMAIN_LABELS[x] for x in DOMAINS], rotation=12, ha="right")
        ax.set_ylabel("Nash–Sutcliffe efficiency" if metric == "NSE" else r"RMSE (m$^3$ s$^{-1}$)")
        ax.set_title(f"({'ab'[j]})  {metric}", loc="left", fontweight="bold")
        if scales[metric] != "linear":
            ax.text(.02, .03, scales[metric], transform=ax.transAxes, fontsize=8, va="bottom")
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=6, color=x[2], label=x[1]) for x in groups]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.015), ncol=2, frameon=False)
    fig.suptitle("Full-model sensitivity to cross-validation design", x=.075, ha="left", fontsize=13, fontweight="bold", y=1.01)
    fig.text(.075, -.062, support_note + " Each design uses its own calibration WSE bounds.", fontsize=8.1)
    fig.text(.075, -.105, "Dots: individual fold/repeat scores. Lines: Q25–Q75; dark bars: median. These intervals are not confidence limits.", fontsize=8.1)
    fig.subplots_adjust(bottom=.25)
    save_figure(fig, output, "fivefold_vs_ten_repeats", dpi, manifest,
                dict(source=source, support=support_note, scales=scales, rows=len(z), schemes=required))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=ROOT / "tables")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--use-workspace-deps", action="store_true", help="Prefer the runner's deps/ directory over this Python environment")
    args = parser.parse_args(argv)
    source = args.tables / "cross_gauge_metrics_by_split.csv"
    if not source.exists():
        parser.error(f"Aggregate fold table is not available yet: {source}; run summarize_fivefold.py after model fitting.")
    scores = read(source)
    required = {"scheme", "support", "method", "domain", "repeat_id", "fold_id", "NSE", "RMSE"}
    if not required.issubset(scores):
        raise AssertionError(f"Missing aggregate columns: {required - set(scores)}")
    if {"Q_observed", "Q_calculated", "obs_id", "time"} & set(scores):
        raise AssertionError("Plotting only accepts aggregate metric tables")
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    manifest = dict(figures={}, skipped={}, source=dict(path=str(source), sha256=digest(source)),
                    intervals="Actual 25th–75th percentiles across fold/repeat scores; not confidence intervals.",
                    quartiles="numpy.quantile(method='linear')", points="Every finite individual fold/repeat score, without clipping.",
                    no_observation_level_inputs=True)
    summary_path = args.tables / "summary.json"
    manifest["analysis_status"] = json.loads(summary_path.read_text(encoding="utf-8")).get("status", "UNKNOWN") if summary_path.exists() else "UNKNOWN"
    overview(scores, args.output, args.dpi, manifest)
    method_comparison(scores, args.output, args.dpi, manifest)
    old_new_comparison(scores, args.tables, args.output, args.dpi, manifest)
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": list(manifest["figures"]), "skipped": manifest["skipped"],
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
