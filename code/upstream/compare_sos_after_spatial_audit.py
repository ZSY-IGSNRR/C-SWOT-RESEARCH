#!/usr/bin/env python3
"""Rebuild and plot local-model versus SoS comparisons after spatial audit.

The model prediction table remains unchanged (nearest node plus up to two
adjacent nodes).  Official SoS observations are rebuilt from Hydrocron
Version-2.0 files addressed by the audited SWORD v16 reach ID, then matched to
the v17b-node overpass within 30 minutes.  Distance thresholds are sensitivity
cohorts, not hard exclusions from the main model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


BLUE = "#2E64A9"
ORANGE = "#D0662D"
GOLD = "#D9A332"
INK = "#25313C"
MID_GREY = "#87929D"
LIGHT_GREY = "#E6E9EC"

MAIN_PRODUCTS = [
    "SoS V3 consensus",
    "SoS V3 MetroMan",
    "SoS V3 MOMMA",
    "SoS V3 SIC4DVar",
]
SCHEMES = ["wse_stratified", "temporal_forward"]
COHORT_ORDER = [
    "audited_mapped",
    "audited_high_confidence",
    "audited_same_numeric_id",
    "audited_le5km",
    "audited_le2km",
    "audited_gt5km",
]
COHORT_LABELS = {
    "audited_mapped": "Audited mapping",
    "audited_high_confidence": "High confidence",
    "audited_same_numeric_id": "Stable same ID",
    "audited_le5km": "Audited, <=5 km",
    "audited_le2km": "Audited, <=2 km",
    "audited_gt5km": "Audited, >5 km",
}
SCHEME_LABELS = {
    "wse_stratified": "WSE",
    "temporal_forward": "Time",
}


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def truthy(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().isin({"true", "1", "1.0"})


SOS_PRODUCTS = {
    "sos_consensus_q": "SoS V3 consensus",
    "sos_hivdi_q": "SoS V3 HiVDI",
    "sos_metroman_q": "SoS V3 MetroMan",
    "sos_momma_q": "SoS V3 MOMMA",
    "sos_sad_q": "SoS V3 SAD",
    "sos_sic4dvar_q": "SoS V3 SIC4DVar",
    "sos_lakeflow_q": "SoS V3 LakeFlow",
}


def collapse_model_predictions(path: Path) -> pd.DataFrame:
    requested = [
        "station_id", "time", "method", "validation_scheme", "repeat_id",
        "Q_observed", "Q_calculated",
    ]
    frame = pd.read_csv(path, usecols=requested, dtype={"station_id": "string"}, low_memory=False)
    frame = frame[
        frame["method"].eq("full")
        & frame["validation_scheme"].isin(SCHEMES)
    ].copy()
    frame["station_id"] = normalize_id(frame["station_id"])
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame["Q_observed"] = pd.to_numeric(frame["Q_observed"], errors="coerce")
    frame["Q_calculated"] = pd.to_numeric(frame["Q_calculated"], errors="coerce")
    return frame.groupby(
        ["station_id", "time", "validation_scheme"], as_index=False, dropna=False
    ).agg(
        Q_observed=("Q_observed", "median"),
        Q_model=("Q_calculated", "median"),
        holdout_prediction_count=("Q_calculated", "size"),
    )


def load_mapped_sos(raw_dir: Path, requested_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    value_columns = list(SOS_PRODUCTS)
    requested = {"reach_id", "time_str", "cycle_id", "pass_id", "sword_version", "collection_version", *value_columns}
    for reach_id in sorted(requested_ids, key=int):
        path = raw_dir / f"{reach_id}.csv"
        if not path.exists():
            audit_rows.append({"v16_reach_id": reach_id, "status": "missing_file", "rows": 0})
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in requested, dtype={"reach_id": "string"}, low_memory=False)
            if frame.empty:
                audit_rows.append({"v16_reach_id": reach_id, "status": "empty_file", "rows": 0})
                continue
            frame["reach_id"] = normalize_id(frame["reach_id"])
            frame["time_sos"] = pd.to_datetime(frame.get("time_str"), errors="coerce", utc=True)
            version = normalize_id(frame.get("sword_version", pd.Series(index=frame.index, dtype="string")))
            id_ok = frame["reach_id"].eq(reach_id)
            version_ok = version.isna() | version.eq("16")
            frame = frame[id_ok & version_ok & frame["time_sos"].notna()].copy()
            for column in value_columns:
                if column not in frame:
                    frame[column] = np.nan
                values = pd.to_numeric(frame[column], errors="coerce")
                frame[column] = values.mask(values.le(-1.0e10))
            frame = frame[["reach_id", "time_sos", "cycle_id", "pass_id", *value_columns]].drop_duplicates()
            frames.append(frame)
            audit_rows.append({
                "v16_reach_id": reach_id,
                "status": "usable",
                "rows": len(frame),
                "rows_any_positive_product": int(frame[value_columns].gt(0).any(axis=1).sum()),
            })
        except Exception as exc:
            audit_rows.append({
                "v16_reach_id": reach_id, "status": "read_error", "rows": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    sos = pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame()
    if not sos.empty:
        sos = sos.sort_values(["reach_id", "time_sos"]).reset_index(drop=True)
    return sos, pd.DataFrame(audit_rows)


def build_audited_common_support(
    predictions_path: Path,
    stations: pd.DataFrame,
    sos_raw_dir: Path,
    tolerance_minutes: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = collapse_model_predictions(predictions_path)
    station_columns = [
        "station_id", "nearest_node_distance_km", "station_node_distance_bin",
        "mapping_status", "mapping_confidence", "v17b_reach_id", "v16_reach_id",
        "selected_node_count", "selected_reach_count",
    ]
    model = model.merge(
        stations[station_columns].drop_duplicates("station_id"),
        on="station_id", how="left", validate="many_to_one",
    )
    mapped_status = {"stable_same_id_high", "remapped_high", "remapped_medium"}
    model = model[
        model["mapping_status"].isin(mapped_status)
        & model["v16_reach_id"].notna()
        & model["time"].notna()
    ].copy()
    model["v16_reach_id"] = normalize_id(model["v16_reach_id"])
    sos, file_audit = load_mapped_sos(sos_raw_dir, set(model["v16_reach_id"]))
    if sos.empty:
        raise RuntimeError(f"No usable mapped v16 SoS files found in {sos_raw_dir}")

    attached: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for reach_id, left in model.groupby("v16_reach_id", sort=False):
        right = sos[sos["reach_id"].eq(reach_id)]
        if right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("time"),
            right.drop(columns="reach_id").sort_values("time_sos"),
            left_on="time", right_on="time_sos", direction="nearest", tolerance=tolerance,
        )
        attached.append(merged)
    wide = pd.concat(attached, ignore_index=True, copy=False) if attached else pd.DataFrame()
    if wide.empty:
        raise RuntimeError("Mapped v16 SoS files did not match any model overpasses")
    wide["sos_match_delta_seconds"] = (wide["time"] - wide["time_sos"]).abs().dt.total_seconds()

    long_rows: list[pd.DataFrame] = []
    for column, label in SOS_PRODUCTS.items():
        values = pd.to_numeric(wide[column], errors="coerce")
        keep = (
            wide["Q_observed"].gt(0) & wide["Q_model"].gt(0)
            & values.gt(0) & values.lt(1.0e10) & wide["time_sos"].notna()
        )
        selected = wide.loc[keep, [
            "station_id", "time", "time_sos", "sos_match_delta_seconds",
            "validation_scheme", "Q_observed", "Q_model", "holdout_prediction_count",
            *station_columns[1:],
        ]].copy()
        selected["Q_product"] = values.loc[keep].to_numpy(float)
        selected["product_column"] = column
        selected["product_label"] = label
        selected["product_family"] = "L4 SoS V3 (SWORD v16 audited)"
        selected["gauge_constrained"] = False
        long_rows.append(selected)
    observations = pd.concat(long_rows, ignore_index=True, copy=False) if long_rows else pd.DataFrame()
    build_audit = pd.DataFrame([
        {"check": "collapsed_model_rows", "value": len(collapse_model_predictions(predictions_path))},
        {"check": "mapped_model_rows", "value": len(model)},
        {"check": "mapped_model_stations", "value": model["station_id"].nunique()},
        {"check": "sos_files_requested", "value": model["v16_reach_id"].nunique()},
        {"check": "sos_rows_loaded", "value": len(sos)},
        {"check": "model_rows_with_any_time_match", "value": int(wide["time_sos"].notna().sum())},
        {"check": "time_match_tolerance_minutes", "value": tolerance_minutes},
        {"check": "long_positive_common_support_rows", "value": len(observations)},
        {"check": "common_support_stations", "value": observations["station_id"].nunique()},
    ])
    return observations, file_audit, build_audit


def metric_values(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    keep = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[keep], pred[keep]
    if not len(obs):
        return {"pair_count": 0}
    error = pred - obs
    sst = float(np.sum((obs - obs.mean()) ** 2))
    if len(obs) >= 2 and np.std(obs) > 0 and np.std(pred) > 0:
        correlation = float(np.corrcoef(obs, pred)[0, 1])
    else:
        correlation = np.nan
    r2 = correlation * correlation if np.isfinite(correlation) else np.nan
    nse = float(1 - np.sum(error ** 2) / sst) if sst > 0 else np.nan
    if (
        len(obs) >= 2
        and np.isfinite(correlation)
        and np.std(obs, ddof=1) > 0
        and np.mean(obs) != 0
    ):
        alpha = np.std(pred, ddof=1) / np.std(obs, ddof=1)
        beta = np.mean(pred) / np.mean(obs)
        kge = float(1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        kge = np.nan
    relative = np.abs(error) / obs
    ratio = pred / obs
    return {
        "pair_count": int(len(obs)),
        "R2": r2,
        "NSE": nse,
        "KGE": kge,
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "PBIAS": float(100 * np.sum(error) / np.sum(obs)),
        "MdAPE": float(100 * np.median(relative)),
        "factor2_fraction": float(np.mean((ratio >= .5) & (ratio <= 2))),
        "factor_error_median": float(np.exp(np.median(np.abs(np.log(ratio))))),
    }


def station_metrics(
    frame: pd.DataFrame,
    pred_col: str,
    *,
    min_pairs: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for station_id, group in frame.groupby("station_id", sort=False):
        record = metric_values(group["Q_observed"], group[pred_col])
        if record.get("pair_count", 0) < min_pairs:
            continue
        record["station_id"] = station_id
        rows.append(record)
    return pd.DataFrame(rows)


def cohort_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    mapped = frame["mapping_status"].isin(
        ["stable_same_id_high", "remapped_high", "remapped_medium"]
    )
    high = frame["mapping_status"].isin(["stable_same_id_high", "remapped_high"])
    same_id = frame["mapping_status"].eq("stable_same_id_high")
    distance = pd.to_numeric(frame["nearest_node_distance_km"], errors="coerce")
    return {
        "audited_mapped": mapped,
        "audited_high_confidence": high,
        "audited_same_numeric_id": same_id,
        "audited_le5km": mapped & distance.le(5),
        "audited_le2km": mapped & distance.le(2),
        "audited_gt5km": mapped & distance.gt(5),
    }


def bootstrap_median(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    draws = rng.choice(values, size=(iterations, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, .025)), float(np.quantile(medians, .975))


def calculate_comparisons(
    observations: pd.DataFrame,
    *,
    min_pairs: int,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    station_rows: list[pd.DataFrame] = []
    advantage_rows: list[dict] = []
    rng = np.random.default_rng(seed)
    metrics = ["R2", "NSE", "KGE", "RMSE", "MAE", "PBIAS", "MdAPE", "factor2_fraction"]
    direction = {
        "R2": 1, "NSE": 1, "KGE": 1, "factor2_fraction": 1,
        "RMSE": -1, "MAE": -1, "MdAPE": -1,
    }

    for cohort, mask in cohort_masks(observations).items():
        subset = observations.loc[mask].copy()
        for (scheme, product_label, product_column), group in subset.groupby(
            ["validation_scheme", "product_label", "product_column"], sort=False
        ):
            if scheme not in SCHEMES or product_label not in MAIN_PRODUCTS:
                continue
            ours = station_metrics(group, "Q_model", min_pairs=min_pairs)
            official = station_metrics(group, "Q_product", min_pairs=min_pairs)
            if ours.empty or official.empty:
                continue
            ours.insert(1, "method", "our_method")
            official.insert(1, "method", "official_product")
            combined = pd.concat([ours, official], ignore_index=True)
            combined.insert(0, "cohort", cohort)
            combined.insert(1, "validation_scheme", scheme)
            combined.insert(2, "product_label", product_label)
            combined.insert(3, "product_column", product_column)
            station_rows.append(combined)

            eligible_ids = sorted(set(ours["station_id"]) & set(official["station_id"]))
            station_pair_count = (
                group[group["station_id"].isin(eligible_ids)]
                .groupby("station_id").size().sum()
            )
            for method, values in [("our_method", ours), ("official_product", official)]:
                row = {
                    "cohort": cohort,
                    "validation_scheme": scheme,
                    "product_label": product_label,
                    "product_column": product_column,
                    "method": method,
                    "pair_count": int(station_pair_count),
                    "station_count": int(len(eligible_ids)),
                }
                values = values[values["station_id"].isin(eligible_ids)]
                for metric in metrics:
                    numeric = pd.to_numeric(values[metric], errors="coerce").dropna()
                    row[f"station_median_{metric}"] = float(numeric.median()) if len(numeric) else np.nan
                    row[f"station_q25_{metric}"] = float(numeric.quantile(.25)) if len(numeric) else np.nan
                    row[f"station_q75_{metric}"] = float(numeric.quantile(.75)) if len(numeric) else np.nan
                    row[f"station_valid_{metric}_count"] = int(len(numeric))
                summary_rows.append(row)

            ours_i = ours.set_index("station_id")
            official_i = official.set_index("station_id")
            for metric, sign in direction.items():
                pair = pd.concat(
                    [ours_i[metric].rename("ours"), official_i[metric].rename("official")],
                    axis=1, join="inner",
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if pair.empty:
                    continue
                advantage = sign * (pair["ours"] - pair["official"])
                lo, hi = bootstrap_median(advantage.to_numpy(), bootstrap_iterations, rng)
                advantage_rows.append({
                    "cohort": cohort,
                    "validation_scheme": scheme,
                    "product_label": product_label,
                    "product_column": product_column,
                    "metric": metric,
                    "station_count": int(len(pair)),
                    "median_advantage_ours": float(np.median(advantage)),
                    "q25_advantage_ours": float(np.quantile(advantage, .25)),
                    "q75_advantage_ours": float(np.quantile(advantage, .75)),
                    "bootstrap_ci_lower_2p5": lo,
                    "bootstrap_ci_upper_97p5": hi,
                    "ours_better_fraction": float(np.mean(advantage > 0)),
                    "tie_fraction": float(np.mean(advantage == 0)),
                    "advantage_definition": (
                        "ours - official; positive favors ours"
                        if sign == 1 else "official - ours; positive favors ours"
                    ),
                })

    summary = pd.DataFrame(summary_rows)
    station = pd.concat(station_rows, ignore_index=True) if station_rows else pd.DataFrame()
    advantage = pd.DataFrame(advantage_rows)
    return summary, station, advantage


def calculate_fixed_four_product_support(
    observations: pd.DataFrame,
    *,
    min_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every candidate on rows where all four main SoS products exist.

    Unlike the pairwise comparison, this makes the local-model population
    identical for every product within each validation scheme.  It is intended
    for the absolute-metric headline figure; pairwise comparisons remain useful
    for confidence intervals because they retain more observations.
    """
    high = observations[
        observations["mapping_status"].isin(["stable_same_id_high", "remapped_high"])
        & observations["product_label"].isin(MAIN_PRODUCTS)
    ].copy()
    keys = ["station_id", "time", "validation_scheme"]
    base = high.groupby(keys, as_index=False).agg(
        Q_observed=("Q_observed", "first"),
        Q_model=("Q_model", "first"),
        nearest_node_distance_km=("nearest_node_distance_km", "first"),
        selected_node_count=("selected_node_count", "first"),
    )
    products = high.pivot_table(
        index=keys, columns="product_label", values="Q_product", aggfunc="first"
    ).reset_index()
    fixed = base.merge(products, on=keys, how="inner", validate="one_to_one")
    fixed = fixed.dropna(subset=MAIN_PRODUCTS).copy()

    rows: list[dict] = []
    metrics = ["R2", "NSE", "KGE", "RMSE", "MAE", "PBIAS", "MdAPE", "factor2_fraction"]
    for scheme in SCHEMES:
        scheme_rows = fixed[fixed["validation_scheme"].eq(scheme)].copy()
        eligible_ids = set(
            scheme_rows.groupby("station_id").size().loc[lambda value: value >= min_pairs].index
        )
        scheme_rows = scheme_rows[scheme_rows["station_id"].isin(eligible_ids)].copy()
        if scheme_rows.empty:
            continue
        local = station_metrics(scheme_rows, "Q_model", min_pairs=min_pairs)
        for product in MAIN_PRODUCTS:
            official = station_metrics(scheme_rows, product, min_pairs=min_pairs)
            for method, values in [("our_method", local), ("official_product", official)]:
                row = {
                    "cohort": "fixed_four_product_high_confidence",
                    "validation_scheme": scheme,
                    "product_label": product,
                    "product_column": product,
                    "method": method,
                    "pair_count": len(scheme_rows),
                    "station_count": len(eligible_ids),
                }
                for metric in metrics:
                    numeric = pd.to_numeric(values[metric], errors="coerce").dropna()
                    row[f"station_median_{metric}"] = float(numeric.median()) if len(numeric) else np.nan
                    row[f"station_q25_{metric}"] = float(numeric.quantile(.25)) if len(numeric) else np.nan
                    row[f"station_q75_{metric}"] = float(numeric.quantile(.75)) if len(numeric) else np.nan
                    row[f"station_valid_{metric}_count"] = int(len(numeric))
                rows.append(row)
    return pd.DataFrame(rows), fixed


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": LIGHT_GREY,
        "grid.linewidth": .7,
        "grid.alpha": .8,
    })


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spatial_audit(
    crosswalk: pd.DataFrame,
    stations: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axes[0, 0]
    order = [
        "stable_same_id_high", "stable_same_id_medium",
        "remapped_high", "remapped_medium",
        "ambiguous_multiple", "ambiguous_or_unmatched",
        "missing_from_v17b_netcdf",
    ]
    counts = crosswalk["mapping_status"].value_counts().reindex(order).dropna()
    colors = [BLUE if str(x).startswith("stable") else GOLD if str(x).startswith("remapped") else MID_GREY for x in counts.index]
    ax.barh(np.arange(len(counts)), counts.values, color=colors, edgecolor=INK, linewidth=.5)
    ax.set_yticks(np.arange(len(counts)), [x.replace("_", " ") for x in counts.index])
    ax.invert_yaxis()
    ax.set_xlabel("Audited v17b reaches")
    ax.set_title("a  v16-v17b reach mapping status", loc="left")
    for y, value in enumerate(counts.values):
        ax.text(value, y, f"  {int(value):,}", va="center")

    ax = axes[0, 1]
    same = crosswalk[truthy(crosswalk["same_numeric_id"])].copy()
    x = pd.to_numeric(same["p95_bidirectional_km"], errors="coerce")
    y = pd.to_numeric(same["overlap_fraction_1km"], errors="coerce")
    valid = x.notna() & y.notna()
    status_color = same.loc[valid, "mapping_status"].map(
        lambda value: BLUE if str(value).startswith("stable_same_id") else ORANGE
    )
    ax.scatter(x[valid], y[valid], s=16, alpha=.55, c=status_color, edgecolors="none")
    ax.axvline(2, color=INK, linestyle="--", linewidth=1)
    ax.axhline(.8, color=INK, linestyle="--", linewidth=1)
    ax.set_xscale("symlog", linthresh=.1)
    ax.set_xlim(left=0)
    ax.set_ylim(-.03, 1.03)
    ax.set_xlabel("Bidirectional centerline p95 distance (km)")
    ax.set_ylabel("Minimum bidirectional overlap within 1 km")
    ax.set_title("b  Same-ID geometry agreement", loc="left")

    ax = axes[1, 0]
    population_masks = {
        "Calibratable stations": stations.get("eligible_for_calibration", pd.Series(False, index=stations.index)).astype(str).str.lower().eq("true"),
        "Audited SoS common support": stations.get("in_audited_sos_common_support", pd.Series(False, index=stations.index)).fillna(False).astype(bool),
    }
    for label, mask in population_masks.items():
        values = pd.to_numeric(stations.loc[mask, "nearest_node_distance_km"], errors="coerce").dropna().sort_values()
        if len(values):
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"{label} (n={len(values):,})")
    for threshold in [2, 5]:
        ax.axvline(threshold, color=INK, linestyle="--", linewidth=.9)
    ax.set_xscale("log")
    ax.set_xlim(left=.01)
    ax.set_xlabel("Gauge-to-nearest-node distance (km)")
    ax.set_ylabel("Cumulative fraction of stations")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("c  Station-node distance is retained as sensitivity", loc="left")

    ax = axes[1, 1]
    counts = []
    for cohort in ["audited_mapped", "audited_high_confidence", "audited_same_numeric_id", "audited_le5km", "audited_le2km"]:
        part = summary[summary["cohort"].eq(cohort)]
        counts.append({
            "cohort": cohort,
            "WSE": part.loc[part["validation_scheme"].eq("wse_stratified"), "station_count"].max(),
            "Time": part.loc[part["validation_scheme"].eq("temporal_forward"), "station_count"].max(),
        })
    counts = pd.DataFrame(counts).set_index("cohort")
    pos = np.arange(len(counts))
    width = .36
    ax.bar(pos - width / 2, counts["WSE"], width, color=BLUE, label="WSE")
    ax.bar(pos + width / 2, counts["Time"], width, color=ORANGE, label="Time")
    ax.set_xticks(pos, [COHORT_LABELS[x] for x in counts.index], rotation=25, ha="right")
    ax.set_ylabel("Maximum eligible stations among four SoS products")
    ax.legend(frameon=False)
    ax.set_title("d  Comparison support after spatial checks", loc="left")

    fig.suptitle("Spatial audit of v16 SoS and v17b station-node comparisons", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .96])
    save_figure(fig, output_dir, "01_spatial_mapping_audit", dpi)


def _comparison_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    product_short = {
        "SoS V3 consensus": "Consensus",
        "SoS V3 MetroMan": "MetroMan",
        "SoS V3 MOMMA": "MOMMA",
        "SoS V3 SIC4DVar": "SIC4DVar",
    }
    frame["display"] = frame.apply(
        lambda row: f"{SCHEME_LABELS.get(row['validation_scheme'], row['validation_scheme'])} | {product_short.get(row['product_label'], row['product_label'])}",
        axis=1,
    )
    order = [f"{scheme} | {product}" for scheme in ["WSE", "Time"] for product in ["Consensus", "MetroMan", "MOMMA", "SIC4DVar"]]
    return frame, order


def plot_paired_advantage(advantage: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    setup_style()
    data = advantage[advantage["cohort"].eq("audited_high_confidence")].copy()
    data, order = _comparison_labels(data)
    specifications = [
        ("R2", "R2 advantage\n(ours - official)"),
        ("NSE", "NSE advantage\n(ours - official)"),
        ("MdAPE", "MdAPE advantage, percentage points\n(official - ours)"),
        ("factor2_fraction", "Factor-of-two advantage\n(ours - official)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.5), sharey=True)
    for ax, (metric, xlabel) in zip(axes, specifications):
        part = data[data["metric"].eq(metric)].set_index("display").reindex(order)
        y = np.arange(len(order))
        value = pd.to_numeric(part["median_advantage_ours"], errors="coerce").to_numpy()
        lo = pd.to_numeric(part["bootstrap_ci_lower_2p5"], errors="coerce").to_numpy()
        hi = pd.to_numeric(part["bootstrap_ci_upper_97p5"], errors="coerce").to_numpy()
        valid = np.isfinite(value) & np.isfinite(lo) & np.isfinite(hi)
        ax.axvline(0, color=INK, linestyle="--", linewidth=1)
        ax.errorbar(
            value[valid], y[valid],
            xerr=np.vstack((value[valid] - lo[valid], hi[valid] - value[valid])),
            fmt="o", color=BLUE, ecolor=BLUE, capsize=3, markersize=5,
        )
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(metric.replace("factor2_fraction", "Factor-of-two fraction"))
        for index in np.flatnonzero(valid):
            fraction = part.iloc[index]["ours_better_fraction"]
            count = part.iloc[index]["station_count"]
            ax.annotate(
                f"win {fraction:.0%}; n={int(count)}",
                (value[index], y[index]), xytext=(5, 6), textcoords="offset points",
                fontsize=7.5, color=INK,
            )
    fig.suptitle(
        "Paired station-level advantage after v16 reach-ID correction",
        fontsize=15, fontweight="bold",
    )
    fig.text(.5, .01, "Dots are station-median paired advantages; bars are bootstrap 95% intervals. Positive favors the local model.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .95])
    save_figure(fig, output_dir, "02_audited_paired_advantage", dpi)


def plot_headline_metrics(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    *,
    cohort: str = "audited_high_confidence",
    title: str = "Local model and SoS products on audited v16-v17b common support",
    stem: str = "03_audited_headline_metrics",
) -> None:
    setup_style()
    data = summary[summary["cohort"].eq(cohort)].copy()
    data, order = _comparison_labels(data)
    specifications = [
        ("R2", "Station-median R2", "linear"),
        ("NSE", "Station-median NSE", "symlog"),
        ("MdAPE", "Station-median MdAPE (%)", "log"),
        ("factor2_fraction", "Station-median factor-of-two fraction", "linear"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.5), sharey=True)
    for ax, (metric, xlabel, scale) in zip(axes, specifications):
        pivot = data.pivot_table(index="display", columns="method", values=f"station_median_{metric}", aggfunc="first").reindex(order)
        y = np.arange(len(order))
        official = pd.to_numeric(pivot.get("official_product"), errors="coerce").to_numpy()
        ours = pd.to_numeric(pivot.get("our_method"), errors="coerce").to_numpy()
        for index in range(len(order)):
            if np.isfinite(official[index]) and np.isfinite(ours[index]):
                ax.plot([official[index], ours[index]], [y[index], y[index]], color=MID_GREY, linewidth=1)
        ax.scatter(official, y, s=32, marker="s", facecolor="white", edgecolor=ORANGE, linewidth=1.5, label="Official SoS")
        ax.scatter(ours, y, s=34, marker="o", color=BLUE, label="Local model")
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=1)
            ax.axvline(0, color=INK, linestyle="--", linewidth=.9)
        elif scale == "log":
            ax.set_xscale("log")
        elif metric in {"R2", "factor2_fraction"}:
            ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(metric.replace("factor2_fraction", "Factor-of-two fraction"))
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
                   bbox_to_anchor=(.5, .945))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95])
    save_figure(fig, output_dir, stem, dpi)


def plot_spatial_sensitivity(advantage: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    setup_style()
    data = advantage[advantage["cohort"].isin([
        "audited_mapped", "audited_high_confidence", "audited_le5km", "audited_le2km"
    ])].copy()
    data, order = _comparison_labels(data)
    cohorts = ["audited_mapped", "audited_high_confidence", "audited_le5km", "audited_le2km"]
    specifications = [
        ("R2", "R2"), ("NSE", "NSE"), ("MdAPE", "MdAPE"),
        ("factor2_fraction", "Factor-of-two")
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.5), sharey=True, layout="constrained")
    for ax, (metric, title) in zip(axes, specifications):
        part = data[data["metric"].eq(metric)]
        pivot = part.pivot_table(index="display", columns="cohort", values="ours_better_fraction", aggfunc="first").reindex(index=order, columns=cohorts)
        counts = part.pivot_table(index="display", columns="cohort", values="station_count", aggfunc="first").reindex(index=order, columns=cohorts)
        image = ax.imshow(pivot.to_numpy(float), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for row in range(len(order)):
            for col in range(len(cohorts)):
                value = pivot.iloc[row, col]
                count = counts.iloc[row, col]
                if np.isfinite(value):
                    color = "white" if value >= .65 else INK
                    ax.text(col, row, f"{value:.2f}\n(n={int(count)})", ha="center", va="center", fontsize=7.2, color=color)
        ax.set_xticks(np.arange(len(cohorts)), [COHORT_LABELS[x] for x in cohorts], rotation=28, ha="right")
        ax.set_yticks(np.arange(len(order)), order)
        ax.set_title(title)
        ax.grid(False)
    colorbar = fig.colorbar(image, ax=axes, fraction=.025, pad=.02, shrink=.88)
    colorbar.set_label("Fraction of stations where local model is better")
    fig.suptitle("Sensitivity of paired product comparison to spatial screening", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "04_spatial_screening_sensitivity", dpi)


def legacy_correction_audit(
    legacy_path: Path,
    corrected: pd.DataFrame,
    *,
    min_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = pd.read_csv(legacy_path, dtype={"station_id": "string"}, low_memory=False)
    legacy = legacy[legacy["product_label"].isin(MAIN_PRODUCTS)].copy()
    legacy["station_id"] = normalize_id(legacy["station_id"])
    legacy["time"] = pd.to_datetime(legacy["time"], errors="coerce", utc=True)
    corrected_keys = corrected[corrected["product_label"].isin(MAIN_PRODUCTS)][[
        "station_id", "time", "validation_scheme", "product_label",
        "Q_observed", "Q_product",
    ]].copy()
    intersection = legacy[[
        "station_id", "time", "validation_scheme", "product_label", "Q_product",
    ]].merge(
        corrected_keys,
        on=["station_id", "time", "validation_scheme", "product_label"],
        how="inner", validate="one_to_one", suffixes=("_legacy", "_corrected"),
    )
    intersection["absolute_log10_change"] = np.abs(
        np.log10(intersection["Q_product_corrected"] / intersection["Q_product_legacy"])
    )
    rows: list[dict] = []
    for (scheme, product), group in intersection.groupby(
        ["validation_scheme", "product_label"], sort=False
    ):
        record = {
            "validation_scheme": scheme,
            "product_label": product,
            "exact_common_pair_count": len(group),
            "exact_common_station_count": group["station_id"].nunique(),
            "unchanged_value_fraction": float(np.isclose(
                group["Q_product_legacy"], group["Q_product_corrected"],
                rtol=0, atol=1.0e-9,
            ).mean()),
            "changed_more_than_20pct_fraction": float(
                (group["absolute_log10_change"] > np.log10(1.2)).mean()
            ),
            "median_factor_change": float(
                10 ** group["absolute_log10_change"].median()
            ),
        }
        for name, column in [
            ("legacy_id_only", "Q_product_legacy"),
            ("corrected_v16", "Q_product_corrected"),
        ]:
            metrics = station_metrics(group, column, min_pairs=min_pairs)
            record[f"{name}_eligible_station_count"] = len(metrics)
            for metric in ["R2", "NSE", "KGE", "MdAPE", "factor2_fraction"]:
                values = pd.to_numeric(metrics.get(metric), errors="coerce").dropna()
                record[f"{name}_station_median_{metric}"] = (
                    float(values.median()) if len(values) else np.nan
                )
        rows.append(record)
    return pd.DataFrame(rows), intersection


def plot_legacy_correction_effect(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    setup_style()
    data, order = _comparison_labels(summary)
    data = data.set_index("display").reindex(order)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.5), sharey=True)

    ax = axes[0]
    changed = pd.to_numeric(data["changed_more_than_20pct_fraction"], errors="coerce")
    ax.barh(y, changed, color=GOLD, edgecolor=INK, linewidth=.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of SoS values changed >20%")
    ax.set_title("Value sensitivity")
    ax.set_yticks(y, order)
    ax.invert_yaxis()

    specifications = [
        ("R2", "Station-median R2", "linear"),
        ("NSE", "Station-median NSE", "symlog"),
        ("MdAPE", "Station-median MdAPE (%)", "log"),
    ]
    for ax, (metric, xlabel, scale) in zip(axes[1:], specifications):
        old = pd.to_numeric(data[f"legacy_id_only_station_median_{metric}"], errors="coerce").to_numpy()
        new = pd.to_numeric(data[f"corrected_v16_station_median_{metric}"], errors="coerce").to_numpy()
        for index in range(len(order)):
            if np.isfinite(old[index]) and np.isfinite(new[index]):
                ax.plot([old[index], new[index]], [y[index], y[index]], color=MID_GREY, linewidth=1)
        ax.scatter(old, y, s=32, marker="s", facecolor="white", edgecolor=ORANGE, linewidth=1.4, label="Legacy ID-only")
        ax.scatter(new, y, s=34, marker="o", color=BLUE, label="Audited v16 ID")
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=1)
            ax.axvline(0, color=INK, linestyle="--", linewidth=.9)
        elif scale == "log":
            ax.set_xscale("log")
        else:
            ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(metric)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Effect of v16 reach-ID correction on identical station-overpass rows",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, .95])
    save_figure(fig, output_dir, "05_sos_id_correction_effect", dpi)


def write_report(
    output_dir: Path,
    crosswalk: pd.DataFrame,
    station_audit: pd.DataFrame,
    summary: pd.DataFrame,
    advantage: pd.DataFrame,
) -> None:
    status = crosswalk["mapping_status"].value_counts(dropna=False)
    primary = advantage[
        advantage["cohort"].eq("audited_high_confidence")
        & advantage["metric"].isin(["R2", "NSE", "MdAPE", "factor2_fraction"])
    ].copy()
    primary["scheme"] = primary["validation_scheme"].map(SCHEME_LABELS)
    lines = [
        "# SWORD v16-v17b spatial audit and SoS comparison",
        "",
        "## Audit scope",
        "",
        "The local hydraulic model remains based on the existing v17b station-node inventory and the three-node aggregation. The audit changes only the eligibility and provenance of the v16 SoS comparison.",
        "",
        "## Reach mapping status",
        "",
        "| Status | Reaches |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {int(value):,} |" for key, value in status.items())
    lines.extend([
        "",
        "## Primary paired advantages after high-confidence v16 ID correction",
        "",
        "| Validation | Product | Metric | Stations | Median advantage | 95% interval | Local win fraction |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for _, row in primary.sort_values(["validation_scheme", "product_label", "metric"]).iterrows():
        lines.append(
            f"| {row['scheme']} | {row['product_label']} | {row['metric']} | {int(row['station_count'])} | "
            f"{row['median_advantage_ours']:.3g} | [{row['bootstrap_ci_lower_2p5']:.3g}, {row['bootstrap_ci_upper_97p5']:.3g}] | "
            f"{row['ours_better_fraction']:.1%} |"
        )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Positive paired advantage always favors the local model.",
        "- Geometry-verified comparisons retain the existing nearest-plus-adjacent-node model; they do not replace it with a single-node model.",
        "- Distance-threshold cohorts are sensitivity analyses. They should be reported alongside, not silently substituted for, the full verified cohort.",
        "- High- and medium-confidence remapped reaches use SoS files queried by the audited v16 ID. Ambiguous reaches remain excluded.",
    ])
    (output_dir / "README_SPATIAL_AUDIT_AND_SOS_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sos-raw-dir", type=Path, required=True)
    parser.add_argument("--legacy-observations", type=Path)
    parser.add_argument("--station-audit", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-tolerance-minutes", type=float, default=30.0)
    parser.add_argument("--min-station-pairs", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(
        args.station_audit,
        dtype={"station_id": "string", "v17b_reach_id": "string", "v16_reach_id": "string"},
        low_memory=False,
    )
    stations["station_id"] = normalize_id(stations["station_id"])
    stations["v16_reach_id"] = normalize_id(stations["v16_reach_id"])
    observations, sos_file_audit, build_audit = build_audited_common_support(
        args.predictions, stations, args.sos_raw_dir, args.time_tolerance_minutes,
    )
    stations["in_audited_sos_common_support"] = stations["station_id"].isin(
        set(observations["station_id"])
    )

    summary, station_metrics_frame, advantage = calculate_comparisons(
        observations,
        min_pairs=args.min_station_pairs,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    fixed_summary, fixed_observations = calculate_fixed_four_product_support(
        observations,
        min_pairs=args.min_station_pairs,
    )
    observations.to_csv(args.output_dir / "audited_common_support_observations.csv", index=False, encoding="utf-8-sig")
    sos_file_audit.to_csv(args.output_dir / "audited_sos_file_quality.csv", index=False, encoding="utf-8-sig")
    build_audit.to_csv(args.output_dir / "audited_common_support_build_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "audited_common_support_summary.csv", index=False, encoding="utf-8-sig")
    station_metrics_frame.to_csv(args.output_dir / "audited_station_metrics.csv", index=False, encoding="utf-8-sig")
    advantage.to_csv(args.output_dir / "audited_paired_station_advantage.csv", index=False, encoding="utf-8-sig")
    fixed_summary.to_csv(
        args.output_dir / "fixed_four_product_common_support_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    fixed_observations.to_csv(
        args.output_dir / "fixed_four_product_common_support_observations.csv",
        index=False, encoding="utf-8-sig",
    )

    crosswalk = pd.read_csv(args.crosswalk, dtype={"v17b_reach_id": "string", "v16_reach_id": "string"}, low_memory=False)
    plot_spatial_audit(crosswalk, stations, summary, args.output_dir, args.dpi)
    plot_paired_advantage(advantage, args.output_dir, args.dpi)
    plot_headline_metrics(
        fixed_summary, args.output_dir, args.dpi,
        cohort="fixed_four_product_high_confidence",
        title="Local model and SoS on fixed four-product common support",
        stem="03_audited_headline_metrics",
    )
    plot_headline_metrics(
        summary, args.output_dir, args.dpi,
        cohort="audited_high_confidence",
        title="Pairwise product-specific common support (sample composition varies)",
        stem="03b_pairwise_common_support_headline_metrics",
    )
    plot_spatial_sensitivity(advantage, args.output_dir, args.dpi)
    if args.legacy_observations:
        correction_summary, correction_rows = legacy_correction_audit(
            args.legacy_observations, observations,
            min_pairs=args.min_station_pairs,
        )
        correction_summary.to_csv(
            args.output_dir / "sos_id_correction_exact_support_summary.csv",
            index=False, encoding="utf-8-sig",
        )
        correction_rows.to_csv(
            args.output_dir / "sos_id_correction_exact_support_observations.csv",
            index=False, encoding="utf-8-sig",
        )
        plot_legacy_correction_effect(correction_summary, args.output_dir, args.dpi)
    write_report(args.output_dir, crosswalk, stations, summary, advantage)

    configuration = {
        "predictions": str(args.predictions),
        "sos_raw_dir": str(args.sos_raw_dir),
        "legacy_observations": str(args.legacy_observations) if args.legacy_observations else None,
        "station_audit": str(args.station_audit),
        "crosswalk": str(args.crosswalk),
        "output_dir": str(args.output_dir),
        "time_tolerance_minutes": args.time_tolerance_minutes,
        "min_station_pairs": args.min_station_pairs,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "primary_cohort": "audited_high_confidence",
        "sos_sword_version": "16",
        "local_model_sword_version": "17b",
        "three_node_aggregation_changed": False,
    }
    (args.output_dir / "comparison_configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(configuration, ensure_ascii=False, indent=2))
    print("\nPrimary audited paired advantages:")
    print(
        advantage[
            advantage["cohort"].eq("audited_high_confidence")
            & advantage["metric"].isin(["R2", "NSE", "MdAPE", "factor2_fraction"])
        ][[
            "validation_scheme", "product_label", "metric", "station_count",
            "median_advantage_ours", "bootstrap_ci_lower_2p5",
            "bootstrap_ci_upper_97p5", "ours_better_fraction",
        ]].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
