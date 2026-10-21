"""Match standard fivefold OOF predictions to original, audited SoS observations.

Run with the same NumPy/Pandas environment as the fivefold experiment:
    python code/compare_products_fivefold.py --root /path/to/run

No old held-out intersection is used to define the new comparison sample. The
measured Q already paired with each local OOF observation is retained, including
near-midnight matches. Restricted observation-level data are written only into
private/. Files in tables/ contain aggregate or derived statistics, never Q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from _paths import work_root
import shutil

import numpy as np
import pandas as pd


SCHEME = "wse_stratified_5fold"
KEY = ["station_id", "obs_id"]
PRODUCTS = {
    "sos_consensus_q": "Consensus",
    "sos_metroman_q": "MetroMan",
    "sos_momma_q": "MOMMA",
    "sos_sic4dvar_q": "SIC4DVar",
}
DOMAINS = ["all", "within", "above", "below"]
DEFAULT_RAW = None
DEFAULT_AUDIT = None
OLD_NATIVE = None


def normalize(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def source_hash(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_aggregate(path: Path, frame: pd.DataFrame):
    forbidden = {"Q_observed", "Q_calculated", "Q_model", *PRODUCTS}
    if forbidden.intersection(frame.columns):
        raise ValueError(f"Raw discharge values cannot be saved in aggregate table {path}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def score(frame: pd.DataFrame, prediction: str, gauge_eligibility: bool = False) -> dict:
    observed = pd.to_numeric(frame.Q_observed, errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(frame[prediction], errors="coerce").to_numpy(float)
    keep = np.isfinite(observed) & (observed > 0) & np.isfinite(predicted)
    observed, predicted = observed[keep], predicted[keep]
    n = len(observed)
    result = {"nobs": n, "NSE": np.nan, "RMSE": np.nan, "SSE": np.nan, "SST": np.nan,
              "observed_sd": np.nan, "predicted_sd": np.nan}
    if not n:
        return result
    sse = float(np.sum(np.square(predicted - observed)))
    sst = float(np.sum(np.square(observed - observed.mean())))
    result.update(SSE=sse, SST=sst, RMSE=float(np.sqrt(sse / n)),
                  observed_sd=float(np.std(observed)), predicted_sd=float(np.std(predicted)))
    eligible = n >= (5 if gauge_eligibility else 2) and sst > 0
    if gauge_eligibility:
        eligible = eligible and np.std(predicted) > 0
    if eligible:
        result["NSE"] = 1.0 - sse / sst
    return result


def reduce_scores(values) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"N": 0, "median": np.nan, "q25": np.nan, "q75": np.nan, "half_IQR": np.nan}
    q25, q75 = np.quantile(array, [0.25, 0.75], method="linear")
    return {"N": len(array), "median": float(np.median(array)), "q25": float(q25),
            "q75": float(q75), "half_IQR": float((q75 - q25) / 2)}


def read_full_predictions(path: Path) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, dtype={"station_id": "string", "obs_id": "string", "node_id": "string", "reach_id": "string"}, chunksize=100_000):
        keep = chunk.method.eq("full")
        if "validation_scheme" in chunk:
            keep &= chunk.validation_scheme.eq(SCHEME)
        if "sample_role" in chunk:
            keep &= chunk.sample_role.eq("validation")
        parts.append(chunk.loc[keep].copy())
    frame = pd.concat(parts, ignore_index=True)
    if frame.empty:
        raise ValueError("No full-model standard fivefold validation predictions")
    required = {*KEY, "fold_id", "repeat_id", "time", "WSE", "Q_observed", "Q_calculated"}
    if missing := required.difference(frame):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    for column in KEY:
        frame[column] = normalize(frame[column])
        if frame[column].isna().any():
            raise ValueError(f"Missing {column}")
    if frame.duplicated(KEY).any():
        raise ValueError("An observation was validated more than once; input is not standard fivefold OOF")
    if not frame.repeat_id.eq(0).all() or not frame.fold_id.isin(range(5)).all():
        raise ValueError("Expected repeat_id = 0 and fold_id in 0..4")
    frame["time"] = pd.to_datetime(frame.time, utc=True, errors="coerce")
    if frame.time.isna().any():
        raise ValueError("Invalid prediction time")
    # Use each full-model fold's calibration domain, never the old ten splits.
    if "domain" not in frame:
        bounds = next(((lo, hi) for lo, hi in [("train_wse_min", "train_wse_max"),
                                              ("width_curve_wse_min", "width_curve_wse_max")]
                       if lo in frame and hi in frame), None)
        if bounds is None:
            raise ValueError("Missing fold-specific calibration WSE bounds/domain")
        low, high = bounds
        frame["domain"] = np.where(frame.WSE < frame[low], "below", np.where(frame.WSE > frame[high], "above", "within"))
    if not frame.domain.isin(DOMAINS[1:]).all():
        raise ValueError("Unknown or missing full-model calibration WSE domain")
    finite = np.isfinite(frame.Q_observed) & frame.Q_observed.gt(0) & np.isfinite(frame.Q_calculated)
    if not finite.all():
        raise ValueError("Full-model prediction input contains invalid Q; failure rows must be audited separately")
    return frame


def match_raw_products(full: pd.DataFrame, audit_path: Path, raw_dir: Path):
    audit = pd.read_csv(audit_path, dtype={"station_id": "string", "v16_reach_id": "string", "v17b_reach_id": "string", "nearest_v17b_node_id": "string"})
    for column in ["station_id", "v16_reach_id", "v17b_reach_id", "nearest_v17b_node_id"]:
        audit[column] = normalize(audit[column])
    if not audit.station_id.is_unique:
        raise ValueError("SWORD crosswalk station IDs are not unique")
    audit_columns = ["station_id", "mapping_status", "mapping_confidence", "v16_reach_id", "v17b_reach_id", "nearest_v17b_node_id"]
    frame = full.merge(audit[audit_columns], on="station_id", how="left", validate="many_to_one")
    identity_checks = {}
    for prediction, crosswalk in [("node_id", "nearest_v17b_node_id"), ("reach_id", "v17b_reach_id")]:
        if prediction in frame:
            known = frame[prediction].notna() & frame[crosswalk].notna()
            if not normalize(frame.loc[known, prediction]).eq(normalize(frame.loc[known, crosswalk])).all():
                raise ValueError(f"Static SWORD crosswalk disagrees with current {prediction}")
            identity_checks[prediction] = {"status": "verified", "rows_checked": int(known.sum())}
        else:
            identity_checks[prediction] = {"status": "not_in_prediction_table", "rows_checked": 0}
    statuses = ["stable_same_id_high", "remapped_high"]
    high = frame.mapping_confidence.eq("high") & frame.mapping_status.isin(statuses) & frame.v16_reach_id.notna()
    mapped = frame.loc[high].copy()
    output, files = [], []
    for reach, group in mapped.groupby("v16_reach_id", sort=True):
        path = raw_dir / f"{reach}.csv"
        info = {"reach_id": reach, "local_observations": len(group), "local_gauges": group.station_id.nunique()}
        if not path.exists():
            files.append({**info, "status": "missing_file"})
            continue
        provenance = source_hash(path)
        raw = pd.read_csv(path, dtype={"reach_id": "string", "sword_version": "string"})
        if not {"reach_id", "time_str"}.issubset(raw):
            raise ValueError(f"Malformed SoS file {path}")
        raw["time_sos"] = pd.to_datetime(raw.time_str, utc=True, errors="coerce")
        version = normalize(raw.get("sword_version", pd.Series(pd.NA, index=raw.index, dtype="string")))
        valid = normalize(raw.reach_id).eq(reach) & (version.isna() | version.eq("16")) & raw.time_sos.notna()
        raw = raw.loc[valid].copy()
        valid_before_window = len(raw)
        lower, upper = group.time.min() - pd.Timedelta(minutes=30), group.time.max() + pd.Timedelta(minutes=30)
        raw = raw.loc[raw.time_sos.between(lower, upper)].copy()
        invalid_counts = {}
        for column, name in PRODUCTS.items():
            raw[column] = pd.to_numeric(raw[column], errors="coerce") if column in raw else np.nan
            invalid_product = ~np.isfinite(raw[column]) | raw[column].le(0) | raw[column].ge(1e10)
            invalid_counts[f"{name}_invalid_or_missing"] = int(invalid_product.sum())
            raw.loc[invalid_product, column] = np.nan
        raw = raw[["time_sos", *PRODUCTS]].drop_duplicates()
        if raw.duplicated("time_sos").any():
            raise ValueError(f"Conflicting SoS estimates for the same reach/time in {path}")
        if raw.empty:
            files.append({**info, "status": "no_rows_in_matching_window", "valid_rows_before_window": valid_before_window, **provenance})
            continue
        joined = pd.merge_asof(group.sort_values("time"), raw.sort_values("time_sos"),
                               left_on="time", right_on="time_sos", direction="nearest", tolerance=pd.Timedelta(minutes=30))
        joined["time_offset_seconds"] = (joined.time - joined.time_sos).abs().dt.total_seconds()
        output.append(joined)
        files.append({**info, "status": "read", "valid_rows_before_window": valid_before_window,
                      "rows_in_matching_window": len(raw), "matched_local_observations": int(joined.time_sos.notna().sum()),
                      **invalid_counts, **provenance})
        if source_hash(path)["sha256"] != provenance["sha256"]:
            raise ValueError(f"SoS source file changed during analysis: {path}")
    if not output:
        raise ValueError("No high-confidence raw SoS matches available")
    matched = pd.concat(output, ignore_index=True)
    if matched.duplicated(KEY).any():
        raise ValueError("Product matching created duplicate OOF observations")
    if not matched.time_offset_seconds.dropna().le(1800).all():
        raise ValueError("Product match exceeded 30 minutes")
    initial_coverage = {
        "input_full_model_OOF_observations": len(full), "input_full_model_gauges": full.station_id.nunique(),
        "high_confidence_mapped_observations": len(mapped), "high_confidence_mapped_gauges": mapped.station_id.nunique(),
        "observations_in_read_source_reaches": len(matched), "time_matched_observations": int(matched.time_sos.notna().sum()),
        "any_product_available": int(matched[list(PRODUCTS)].notna().any(axis=1).sum()),
        "all_four_products_available": int(matched[list(PRODUCTS)].notna().all(axis=1).sum()),
        "crosswalk_identity_checks": identity_checks,
        "mapping_status_counts": frame.mapping_status.fillna("missing").value_counts().to_dict(),
    }
    return matched, pd.DataFrame(files), initial_coverage


def summarize_comparisons(matched: pd.DataFrame):
    supports = [("product_specific", name, matched.dropna(subset=[column]), {"Local": "Q_calculated", name: column})
                for column, name in PRODUCTS.items()]
    supports.append(("common_four", "All_four_products", matched.dropna(subset=list(PRODUCTS)),
                     {"Local": "Q_calculated", **{name: column for column, name in PRODUCTS.items()}}))
    by_fold, oof_scores, gauge_rows, coverage = [], [], [], []
    for support, product, base, models in supports:
        for domain in DOMAINS:
            subset = base if domain == "all" else base.loc[base.domain.eq(domain)]
            info = {"support": support, "product": product, "domain": domain}
            coverage.append({**info, "observations": len(subset), "gauges": subset.station_id.nunique(),
                             "nonempty_folds": subset.fold_id.nunique()})
            for model, prediction in models.items():
                oof_scores.append({**info, "model": model, "aggregation": "each_observation_once_OOF",
                                   "ngauges": subset.station_id.nunique(), **score(subset, prediction)})
            for fold in range(5):
                fold_data = subset.loc[subset.fold_id.eq(fold)]
                for model, prediction in models.items():
                    by_fold.append({**info, "fold_id": fold, "model": model, "ngauges": fold_data.station_id.nunique(), **score(fold_data, prediction)})
            for sid, group in subset.groupby("station_id", sort=False):
                scores = {model: score(group, column, gauge_eligibility=True) for model, column in models.items()}
                # Compare identical gauge series and require all competitors to
                # satisfy the original >=5-pair, nonconstant-series definition.
                if all(np.isfinite(value["NSE"]) for value in scores.values()):
                    for model, values in scores.items():
                        gauge_rows.append({**info, "station_id": sid, "model": model, **values})
    per_fold = pd.DataFrame(by_fold)
    group_columns = ["support", "product", "domain", "model"]
    summaries = []
    for key, group in per_fold.groupby(group_columns, sort=False):
        info = dict(zip(group_columns, key))
        for metric in ["NSE", "RMSE", "nobs", "ngauges", "SSE", "SST"]:
            summaries.append({**info, "metric": metric, **reduce_scores(group[metric])})
    gains = []
    for key, group in per_fold.groupby(["support", "product", "domain"], sort=False):
        pivot = group.pivot(index="fold_id", columns="model", values=["NSE", "RMSE", "nobs", "SST"])
        for model in pivot["NSE"].columns:
            if model == "Local":
                continue
            for metric in ["nobs", "SST"]:
                if not np.allclose(pivot[metric]["Local"], pivot[metric][model], equal_nan=True):
                    raise ValueError("Competing models were evaluated on different observations")
            for metric in ["NSE", "RMSE"]:
                gains.append({**dict(zip(["support", "product", "domain"], key)), "comparison": f"Local-minus-{model}",
                              "metric": metric, **reduce_scores(pivot[metric]["Local"] - pivot[metric][model])})
    gauge = pd.DataFrame(gauge_rows, columns=[*group_columns, "station_id", "nobs", "NSE", "RMSE", "SSE", "SST", "observed_sd", "predicted_sd"])
    gauge_summary = []
    for key, group in gauge.groupby(group_columns, sort=False):
        for metric in ["NSE", "RMSE"]:
            gauge_summary.append({**dict(zip(group_columns, key)), "metric": metric,
                                  "positive_NSE_fraction": float(group.NSE.gt(0).mean()),
                                  "negative_NSE_fraction": float(group.NSE.lt(0).mean()), **reduce_scores(group[metric])})
    return {
        "products_metrics_by_fold.csv": per_fold,
        "products_fold_summary.csv": pd.DataFrame(summaries),
        "products_paired_fold_gains.csv": pd.DataFrame(gains),
        "products_OOF_metrics.csv": pd.DataFrame(oof_scores),
        "products_OOF_gauge_metrics.csv": gauge,
        "products_OOF_gauge_summary.csv": pd.DataFrame(gauge_summary),
        "products_coverage.csv": pd.DataFrame(coverage),
    }


def copy_native_summaries(old: Path, tables: Path):
    copied = []
    for name in ["native_official_pooled.csv", "native_official_coverage.csv", "native_official_station_summary.csv"]:
        source = old / name
        if not source.exists():
            continue
        target = tables / ("products_reused_" + name)
        shutil.copy2(source, target)
        copied.append({"source": source_hash(source), "output": str(target)})
    method = old / "native_official_method_and_validation.json"
    if method.exists():
        target = tables / "products_reused_native_official_method_and_validation.json"
        shutil.copy2(method, target)
        copied.append({"source": source_hash(method), "output": str(target)})
    return copied


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=work_root())
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--station-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--sos-raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--old-native-dir", type=Path, default=OLD_NATIVE)
    parser.add_argument("--skip-native-copy", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    private, tables = root / "private", root / "tables"
    private.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    predictions = args.predictions or private / "validation_predictions_RESTRICTED.csv"
    full = read_full_predictions(predictions)
    matched, file_audit, coverage = match_raw_products(full, args.station_audit, args.sos_raw_dir)
    matched.to_csv(private / "products_matched_observations_RESTRICTED.csv", index=False, encoding="utf-8-sig")
    save_aggregate(tables / "products_source_file_audit.csv", file_audit)
    outputs = summarize_comparisons(matched)
    for name, frame in outputs.items():
        save_aggregate(tables / name, frame)
    reused = [] if args.skip_native_copy else copy_native_summaries(args.old_native_dir, tables)
    timed = matched.loc[matched.time_sos.notna()]
    midnight = timed.time.dt.floor("D").ne(timed.time_sos.dt.floor("D"))
    reused_product = timed.duplicated(["station_id", "time_sos"], keep=False)
    report = {
        "status": "PASS", "validation_scheme": SCHEME, **coverage,
        "sources": [source_hash(predictions), source_hash(args.station_audit)],
        "raw_sos_directory": str(args.sos_raw_dir),
        "matching": "Nearest original SoS acquisition within 30 minutes, SWORD v16 crosswalk with high-confidence stable_same_id_high/remapped_high status; conflicting same-reach/time products raise an error.",
        "time_tolerance_seconds": 1800,
        "maximum_time_offset_seconds": timed.time_offset_seconds.max(),
        "cross_midnight_matches": int(midnight.sum()),
        "observed_Q_rule": "Retain Q_observed from each local held-out prediction and its original matched gauge date. Do not rematch Q using the SoS UTC date.",
        "official_validity": "Finite positive Q < 1e10; nonpositive values and the -999999999999 fill value are unavailable. Local finite nonpositive predictions are retained as prediction errors.",
        "same_product_time_reused_for_distinct_local_observations": int(reused_product.sum()),
        "primary_support": "common_four: all four official products plus the local model on identical observations. Product-specific comparisons are additional, separately labelled supports.",
        "national_NSE": "Compute NSE and RMSE across gauges separately in each held-out fold, then report the median and half the IQR of up to five fold scores. This is a descriptive fold distribution, not a confidence interval. No gauge-level pair-count screen is applied to national scores.",
        "OOF_NSE": "Concatenate the five mutually exclusive held-out folds and evaluate each station-obs_id once; this differs from the median of five fold scores.",
        "domains": "Each observation uses its full-model fold-specific calibration WSE bounds; exact boundaries are within. Product domain comparisons inherit the local-model domain.",
        "gauge_diagnostic": "Each station is evaluated once from OOF data, with >=5 pairs and nonconstant observed AND predicted discharge for all compared models on the same support. Report equal-station distributions separately from national NSE.",
        "constraints": "Local estimates used measured discharge for calibration. The archived upstream selection identifies these four SoS columns as unconstrained. The saved raw tables do not independently provide gauge-assimilation status.",
        "native_official_reused": reused,
        "native_reuse_scope": "Copied pre-existing broad native-product evaluations are independent of the new CV and use the frozen original 542-gauge cohort and each product's own available observations. They are not recalculated on, or restricted to, the new fivefold-eligible gauges. Their original method/source audit is copied alongside the aggregate tables.",
        "product_version": "SWOT L4 SoS River Discharge V3 stored under the Hydrocron RiverSP 2.0 collection request, SWORD16. Collection version 2.0 is not an SoS algorithm Version 2 claim.",
        "restricted_output": str(private / "products_matched_observations_RESTRICTED.csv"),
        "aggregate_outputs_contain_observed_or_predicted_Q": False,
    }
    write_json(tables / "products_method_and_validation.json", report)
    main_summary = outputs["products_fold_summary.csv"]
    selected = main_summary.support.eq("common_four") & main_summary.domain.eq("all") & main_summary.metric.isin(["NSE", "RMSE"])
    print(main_summary.loc[selected].to_string(index=False), flush=True)
    print(json.dumps(json_safe(coverage), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
