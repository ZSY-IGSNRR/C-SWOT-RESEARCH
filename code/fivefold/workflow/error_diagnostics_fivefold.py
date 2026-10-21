"""Additional diagnostics from saved fivefold predictions; no model fitting.

Outputs contain aggregate or derived scores, never measured or predicted Q.
Factor error is max(Qpred/Qobs, Qobs/Qpred), evaluated on positive finite
predictions. Nonpositive predictions are counted explicitly as unscorable
factor errors and remain in NSE/RMSE calculations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from _paths import work_root
import platform

import numpy as np
import pandas as pd

from compare_products_fivefold import json_safe, reduce_scores, save_aggregate, source_hash, write_json


BIN_EDGES = [0, .25, .5, 1, np.inf]
BIN_LABELS = ["(0,0.25]", "(0.25,0.5]", "(0.5,1]", "(1,infinity)"]
BOUND_FLAGS = {
    "n": ("n_at_lower_bound", "n_at_upper_bound"),
    "beta": ("beta_at_lower_bound", "beta_at_upper_bound"),
    "depth0": ("depth_at_lower_bound", "depth_at_upper_bound"),
    "width0": ("width0_at_lower_bound", "width0_at_upper_bound"),
}


def boolean(series):
    tokens = series.astype("string").str.strip().str.lower()
    if not tokens.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError(f"Unknown or missing boolean values in {series.name}")
    return tokens.isin(["true", "1", "1.0"]).to_numpy(bool)


def read_predictions(path):
    columns = ["station_id", "obs_id", "method", "fold_id", "repeat_id", "validation_scheme", "WSE", "Q_observed", "Q_calculated",
               "train_wse_min", "train_wse_max", "calibration_wse_span", "domain", "normalized_extrapolation_distance"]
    data = pd.read_csv(path, usecols=columns, dtype={"station_id": "string", "obs_id": "string"})
    if not data.validation_scheme.eq("wse_stratified_5fold").all() or not data.repeat_id.eq(0).all():
        raise ValueError("Input is not the new standard fivefold CV")
    if data.duplicated(["station_id", "obs_id", "method"]).any():
        raise ValueError("Repeated observations within a method's OOF predictions")
    if not (np.isfinite(data.Q_observed) & data.Q_observed.gt(0) & np.isfinite(data.Q_calculated)).all():
        raise ValueError("Nonfinite prediction or invalid measured Q in saved successful predictions")
    return data


def factor_error_tables(full):
    outside = full.loc[full.domain.isin(["above", "below"])].copy()
    expected = np.maximum(outside.train_wse_min - outside.WSE, np.maximum(outside.WSE - outside.train_wse_max, 0)) / outside.calibration_wse_span
    if not np.allclose(outside.normalized_extrapolation_distance, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise ValueError("Saved normalized extrapolation distance disagrees with fold-specific WSE bounds")
    usable_distance = np.isfinite(outside.normalized_extrapolation_distance) & outside.normalized_extrapolation_distance.gt(0)
    dropped_distance = len(outside) - int(usable_distance.sum())
    outside = outside.loc[usable_distance].copy()
    outside["distance_bin"] = pd.cut(outside.normalized_extrapolation_distance, BIN_EDGES, labels=BIN_LABELS, right=True).astype("string")
    positive = outside.Q_calculated.gt(0)
    outside["factor_error"] = np.nan
    outside.loc[positive, "factor_error"] = np.maximum(outside.loc[positive, "Q_calculated"] / outside.loc[positive, "Q_observed"],
                                                     outside.loc[positive, "Q_observed"] / outside.loc[positive, "Q_calculated"])
    outside["factor_error_valid"] = np.isfinite(outside.factor_error)
    by_gauge, balanced, by_fold = [], [], []
    for side in ["above", "below"]:
        for distance_bin in BIN_LABELS:
            subset = outside.loc[outside.domain.eq(side) & outside.distance_bin.eq(distance_bin)]
            gauges = []
            for sid, group in subset.groupby("station_id", sort=False):
                valid = group.loc[group.factor_error_valid, "factor_error"]
                median = float(valid.median()) if len(valid) else np.nan
                by_gauge.append({"domain": side, "distance_bin": distance_bin, "station_id": sid,
                                 "OOF_observations": len(group), "finite_factor_error_observations": len(valid),
                                 "nonpositive_prediction_count": int(group.Q_calculated.le(0).sum()),
                                 "gauge_median_factor_error": median})
                if len(valid):
                    gauges.append(median)
            balanced.append({"aggregation": "OOF_equal_station", "domain": side, "distance_bin": distance_bin,
                             "observations": len(subset), "gauges_with_observations": subset.station_id.nunique(),
                             "finite_factor_error_observations": int(subset.factor_error_valid.sum()),
                             "nonpositive_prediction_count": int(subset.Q_calculated.le(0).sum()), **reduce_scores(gauges)})
            for fold in range(5):
                f = subset.loc[subset.fold_id.eq(fold)]
                medians = f.loc[f.factor_error_valid].groupby("station_id").factor_error.median()
                by_fold.append({"domain": side, "distance_bin": distance_bin, "fold_id": fold,
                                "observations": len(f), "gauges_with_observations": f.station_id.nunique(),
                                "nonpositive_prediction_count": int(f.Q_calculated.le(0).sum()), **reduce_scores(medians)})
    by_fold = pd.DataFrame(by_fold)
    fold_summary = []
    for (side, distance_bin), group in by_fold.groupby(["domain", "distance_bin"], sort=False):
        fold_summary.append({"aggregation": "median_of_fold_equal_station_medians", "domain": side, "distance_bin": distance_bin,
                             **reduce_scores(group["median"])})
    return {"extrapolation_factor_error_OOF_gauge_metrics.csv": pd.DataFrame(by_gauge),
            "extrapolation_factor_error_OOF_summary.csv": pd.DataFrame(balanced),
            "extrapolation_factor_error_by_fold.csv": by_fold,
            "extrapolation_factor_error_fold_summary.csv": pd.DataFrame(fold_summary)}, dropped_distance


def parameter_tables(parameters):
    full = parameters.loc[parameters.method.eq("full")].copy()
    if full.duplicated(["station_id", "fold_id"]).any():
        raise ValueError("Duplicate full-model station-fold parameter vectors")
    rows, fold_rows, gauge_rows = [], [], []
    flags = []
    for parameter, (low, high) in BOUND_FLAGS.items():
        full[low], full[high] = boolean(full[low]), boolean(full[high])
        any_name = parameter + "_any_bound"
        full[any_name] = full[low] | full[high]
        flags.append(any_name)
        for side, column in [("lower", low), ("upper", high), ("either", any_name)]:
            counts = full.groupby("station_id")[column].sum()
            rows.append({"parameter": parameter, "bound": side, "fitted_station_fold_units": len(full),
                         "boundary_hits": int(full[column].sum()), "fraction_of_fits": float(full[column].mean()),
                         "gauges": len(counts), "gauges_with_any_hit": int(counts.gt(0).sum()),
                         "fraction_of_gauges_with_any_hit": float(counts.gt(0).mean()),
                         "median_hit_folds_per_gauge": float(counts.median())})
            for fold, group in full.groupby("fold_id"):
                fold_rows.append({"parameter": parameter, "bound": side, "fold_id": fold,
                                  "fitted_gauges": len(group), "boundary_hits": int(group[column].sum()),
                                  "fraction_of_fits": float(group[column].mean())})
        for sid, group in full.groupby("station_id", sort=False):
            gauge_rows.append({"station_id": sid, "parameter": parameter, "fitted_folds": len(group),
                               "lower_hit_folds": int(group[low].sum()), "upper_hit_folds": int(group[high].sum()),
                               "either_hit_folds": int(group[any_name].sum()), "median_parameter": float(group[parameter].median())})
    full["any_parameter_bound"] = full[flags].any(axis=1)
    bound_count = {"full_model_parameter_vectors": len(full), "full_model_gauges": full.station_id.nunique(),
                   "any_parameter_bound_vectors": int(full.any_parameter_bound.sum()),
                   "fraction_vectors_any_parameter_bound": float(full.any_parameter_bound.mean()),
                   "gauges_any_bound_in_at_least_one_fold": int(full.groupby("station_id").any_parameter_bound.any().sum())}
    return {"parameter_boundary_summary.csv": pd.DataFrame(rows), "parameter_boundary_by_fold.csv": pd.DataFrame(fold_rows),
            "parameter_boundary_by_gauge.csv": pd.DataFrame(gauge_rows)}, bound_count


def station_NSE_sensitivity(predictions):
    gauge_rows = []
    for (method, sid), group in predictions.groupby(["method", "station_id"], sort=False):
        observed, predicted = group.Q_observed.to_numpy(float), group.Q_calculated.to_numpy(float)
        sst = float(np.square(observed - observed.mean()).sum())
        sse = float(np.square(observed - predicted).sum())
        eligible_math = len(group) >= 5 and sst > 0
        nonconstant_prediction = bool(np.std(predicted) > 0)
        gauge_rows.append({"method": method, "station_id": sid, "nobs": len(group),
                           "OOF_NSE_mathematical": 1 - sse / sst if eligible_math else np.nan,
                           "OOF_NSE_legacy": 1 - sse / sst if eligible_math and nonconstant_prediction else np.nan,
                           "OOF_RMSE": float(np.sqrt(sse / len(group))), "observed_nonconstant": sst > 0,
                           "predicted_nonconstant": nonconstant_prediction, "eligible_mathematical": eligible_math,
                           "eligible_legacy": eligible_math and nonconstant_prediction})
    gauges = pd.DataFrame(gauge_rows)
    summary = []
    for method, group in gauges.groupby("method", sort=False):
        for label, column in [("mathematical_NSE", "OOF_NSE_mathematical"), ("legacy_nonconstant_prediction_required", "OOF_NSE_legacy")]:
            eligible = group.loc[np.isfinite(group[column])]
            summary.append({"method": method, "definition": label, "total_gauges": len(group),
                            "constant_prediction_gauges": int((~group.predicted_nonconstant).sum()),
                            "positive_NSE_fraction": float(eligible[column].gt(0).mean()) if len(eligible) else np.nan,
                            "negative_NSE_fraction": float(eligible[column].lt(0).mean()) if len(eligible) else np.nan,
                            **reduce_scores(eligible[column])})
    return {"station_OOF_NSE_definition_sensitivity.csv": pd.DataFrame(summary),
            "station_OOF_NSE_definition_gauge_metrics.csv": gauges}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=work_root())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    predpath, parpath = root / "private/validation_predictions_RESTRICTED.csv", root / "private/parameter_estimates_RESTRICTED.csv"
    predictions = read_predictions(predpath)
    parameters = pd.read_csv(parpath, dtype={"station_id": "string"})
    outputs, invalid_distance = factor_error_tables(predictions.loc[predictions.method.eq("full")])
    boundary_outputs, counts = parameter_tables(parameters)
    outputs.update(boundary_outputs)
    outputs.update(station_NSE_sensitivity(predictions))
    for name, frame in outputs.items():
        save_aggregate(tables / name, frame)
    report = {"status": "PASS", "sources": [source_hash(predpath), source_hash(parpath)],
              "analysis_environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "validation_scheme": "wse_stratified_5fold", "full_model_OOF_rows": int(predictions.method.eq("full").sum()),
              "extrapolation_records_without_positive_finite_normalized_distance": invalid_distance,
              "factor_error": "max(Qpred/Qobs, Qobs/Qpred), finite positive predicted and observed discharge only; nonpositive predictions are counted explicitly and are retained in NSE/RMSE.",
              "distance_bins": BIN_LABELS, "bin_boundaries": "Right closed. A distance of exactly 0.25 is in (0,0.25], exactly 0.5 is in (0.25,0.5], exactly 1 is in (0.5,1].",
              "station_balancing": "For each above/below distance bin, first compute the median factor error for each gauge from its unique OOF predictions, then summarize equally weighted gauge medians. Per-fold summaries are additional diagnostics with potentially different contributing gauges.",
              "boundary_definition": "Use the original V15 saved bound-hit flags: within 1% of the optimization-coordinate bound span. The coordinates are log(n), log(beta), log(depth0), and the linear width0/minimum-training-width ratio. WSE0 = minimum training WSE - depth0, so depth0 flags diagnose the unobserved vertical geometry parameter.",
              "boundary_interpretation": "Near-bound flags describe fitted station-fold vectors. A gauge with at least one hit over five folds is a different count. Parameter bounds differ across gauges for geometry; this audit does not interpret a hit as proof of optimizer failure.",
              "NSE_sensitivity": "Both station definitions require >=5 OOF pairs and positive observed SST. Mathematical NSE permits a constant prediction; the legacy manuscript eligibility additionally requires predicted variation. Predictions remain unique per station-observation-method.",
              **counts, "outputs_contain_observed_or_predicted_Q": False, "models_refitted": False}
    write_json(tables / "error_diagnostics_method.json", report)
    print(json.dumps(json_safe(report), ensure_ascii=False), flush=True)
    print(outputs["extrapolation_factor_error_OOF_summary.csv"].to_string(index=False), flush=True)
    print(outputs["parameter_boundary_summary.csv"].to_string(index=False), flush=True)
    print(outputs["station_OOF_NSE_definition_sensitivity.csv"].loc[lambda x: x.method.eq("full")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
