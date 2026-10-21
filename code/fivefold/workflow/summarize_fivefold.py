"""Aggregate the V15 standard five-fold predictions without exporting gauge Q.

This script never fits a model and never substitutes archived parameters for a
new fold.  New predictions remain in private/.  Output CSVs contain aggregate
scores only.  All default paths are derived from this script and existing audit
metadata; --old-predictions can override an archived input path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import work_root
ROOT = work_root()
WORK = ROOT.parent
from _paths import external_path
OLD_LOCAL = external_path("CSWOT_LEGACY_LOCAL")
OLD_CONTROLS = external_path("CSWOT_LEGACY_CONTROLS")
METHODS = [
    "full", "beta_fixed_1", "width_train_only", "no_initial_width_guard",
    "no_width_growth_guard", "monotonic_only", "no_hampel", "no_monotonic",
    "fixed_reference_n", "n_only_fixed_geometry", "empirical_rating_curve",
    "observed_fixed", "corrected_fixed",
]
DOMAINS = ["all", "within", "above", "below", "outside"]
OBSKEY = ["station_id", "obs_id"]
SPLITKEY = ["repeat_id", "fold_id"]
KEY = OBSKEY + SPLITKEY
FORBIDDEN = {"Q_observed", "Q_calculated", "Q_model", "obs_id", "time"}
METRICS = ["nobs", "ngauges", "n_unique_observations", "SSE", "SST", "NSE",
           "RMSE", "within_gauge_SST", "between_gauge_SST", "between_gauge_SST_fraction"]


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": h.hexdigest()}


def read(path, **kwargs):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                       dtype={"station_id": "string", "obs_id": "string"}, **kwargs)


def save_csv(path, frame):
    if FORBIDDEN & set(frame.columns):
        raise AssertionError(f"Observation-level fields cannot leave private/: {path}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(N=0, median=np.nan, q25=np.nan, q75=np.nan, half_IQR=np.nan,
                    minimum=np.nan, maximum=np.nan)
    q25, median, q75 = np.quantile(a, [.25, .5, .75], method="linear")
    return dict(N=len(a), median=float(median), q25=float(q25), q75=float(q75),
                half_IQR=float((q75 - q25) / 2), minimum=float(a.min()), maximum=float(a.max()))


def metrics(g, minimum_pairs=2, require_prediction_variation=False):
    # Inputs have already passed one common finite/positive-observation filter.
    o = g.Q_observed.to_numpy(float)
    p = g.Q_calculated.to_numpy(float)
    n = len(g)
    ngauges = int(g.station_id.nunique())
    out = dict(nobs=n, ngauges=ngauges,
               n_unique_observations=int(g.obs_id.nunique()) if ngauges <= 1 else len(g[OBSKEY].drop_duplicates()))
    out.update({k: np.nan for k in METRICS if k not in out})
    if not n:
        return out
    err = p - o
    sse = float(np.dot(err, err))
    centered = o - o.mean()
    sst = float(np.dot(centered, centered))
    if ngauges == 1:
        within = sst
    else:
        station_centered = o - g.groupby("station_id", observed=True).Q_observed.transform("mean").to_numpy()
        within = float(np.dot(station_centered, station_centered))
    between = max(0., sst - within) if sst - within > -1e-8 * max(sst, 1) else sst - within
    out.update(SSE=sse, SST=sst, RMSE=float(np.sqrt(sse / n)), within_gauge_SST=within,
               between_gauge_SST=between,
               between_gauge_SST_fraction=between / sst if sst > 0 else np.nan)
    if n >= minimum_pairs and sst > 0 and (not require_prediction_variation or np.std(p) > 0):
        out["NSE"] = 1 - sse / sst
    return out


def mask_domain(d, domain):
    if domain == "all":
        return np.ones(len(d), dtype=bool)
    if domain == "outside":
        return d.domain.isin(["above", "below"])
    return d.domain.eq(domain)


def read_predictions(path, scheme, parameters_path=None, validation_scheme=None):
    header = read(path, nrows=0).columns
    wanted = KEY + ["method", "sample_role", "validation_scheme", "scheme", "WSE",
                    "Q_observed", "Q_calculated", "train_wse_min", "train_wse_max",
                    "width_curve_wse_min", "width_curve_wse_max", "domain"]
    d = read(path, usecols=[x for x in wanted if x in header])
    if "sample_role" in d:
        d = d[d.sample_role.eq("validation")].copy()
    if validation_scheme is not None:
        col = "validation_scheme" if "validation_scheme" in d else "scheme"
        d = d[d[col].eq(validation_scheme)].copy()
    for key in OBSKEY:
        d[key] = d[key].str.strip().str.replace(r"\.0$", "", regex=True)
    if d[OBSKEY].isna().any().any():
        raise AssertionError("Missing station or observation identifier")
    for c in SPLITKEY:
        if c not in d:
            d[c] = 0
        d[c] = d[c].astype(int)
    if d.duplicated(["method"] + KEY).any():
        raise AssertionError("Duplicate prediction within a method/split/observation")
    spread = d.groupby(OBSKEY, observed=True).Q_observed.agg(["min", "max"])
    if not np.allclose(spread["min"], spread["max"], rtol=1e-10, atol=1e-10, equal_nan=True):
        raise AssertionError("Observed Q disagrees across saved method predictions")
    source_count = len(d)
    valid = np.isfinite(d.Q_observed) & d.Q_observed.gt(0) & np.isfinite(d.Q_calculated)
    exclusions = d.assign(valid=valid).groupby("method", observed=True).valid.agg(["size", "sum"]).reset_index()
    exclusions = exclusions.rename(columns={"size": "source_rows", "sum": "valid_rows"})
    exclusions["scheme"] = scheme
    d = d[valid].copy()
    needs_bounds = not {"train_wse_min", "train_wse_max"}.issubset(d.columns)
    if not needs_bounds:
        needs_bounds = d[["train_wse_min", "train_wse_max"]].isna().any().any()
    if needs_bounds and parameters_path is not None and Path(parameters_path).exists():
        p = read(parameters_path)
        if validation_scheme and "validation_scheme" in p:
            p = p[p.validation_scheme.eq(validation_scheme)]
        for c in SPLITKEY:
            if c not in p:
                p[c] = 0
            p[c] = p[c].astype(int)
        p.station_id = p.station_id.str.strip().str.replace(r"\.0$", "", regex=True)
        pkey = ["station_id", "method"] + SPLITKEY
        p = p[pkey + ["train_wse_min", "train_wse_max"]]
        if p.duplicated(pkey).any():
            raise AssertionError("Duplicate method/fit parameter rows")
        d = d.drop(columns=[c for c in ["train_wse_min", "train_wse_max"] if c in d])
        d = d.merge(p, on=pkey, how="left", validate="many_to_one")
    if not {"train_wse_min", "train_wse_max"}.issubset(d):
        # Compatibility only: old fixed-control exports carry the exact same
        # calibration boundaries as their saved training width curve.
        for side in ["min", "max"]:
            d[f"train_wse_{side}"] = d.get(f"width_curve_wse_{side}", np.nan)
    bounds = d[["WSE", "train_wse_min", "train_wse_max"]].to_numpy(float)
    if not np.isfinite(bounds).all() or np.any(d.train_wse_max < d.train_wse_min):
        raise AssertionError("Finite per-method training WSE boundaries are required")
    expected = np.select([d.WSE.lt(d.train_wse_min), d.WSE.gt(d.train_wse_max)],
                         ["below", "above"], default="within")
    if "domain" in d and d.domain.notna().any():
        have = d.domain.notna()
        if not np.array_equal(d.loc[have, "domain"].to_numpy(), expected[have]):
            raise AssertionError("Saved domain differs from this method's training WSE boundaries")
    d["domain"] = expected
    d["analysis_scheme"] = scheme
    return d, exclusions, source_count


def shared(d, methods):
    z = d[d.method.isin(methods)].copy()
    if not set(methods).issubset(set(z.method.unique())):
        return z.iloc[:0].copy()
    count = z.groupby(KEY, observed=True).method.nunique()
    idx = count[count.eq(len(methods))].index
    z = z.set_index(KEY).loc[idx].reset_index()
    # Comparisons use the full model's domain on the exact same records. Native
    # tables retain each method's own domain. Both definitions are explicit.
    full = z[z.method.eq("full")][KEY + ["domain"]].rename(columns={"domain": "full_domain"})
    if "full" in methods:
        z = z.merge(full, on=KEY, validate="many_to_one")
        z["domain"] = z.pop("full_domain")
    return z


def support_sets(d):
    present = [x for x in METHODS if x in set(d.method)]
    yield "native", "native_method_observations", d
    yield "all_methods_common", "all_13_methods", shared(d, METHODS)
    yield "requested_three_common", "width_correction_and_joint_calibration", shared(d, ["full", "observed_fixed", "corrected_fixed"])
    yield "hydraulic_ablation_common", "hydraulic_parameter_ablations", shared(d, ["full", "n_only_fixed_geometry", "fixed_reference_n", "beta_fixed_1"])
    for method in present:
        if method != "full":
            yield "pairwise_common", f"full_vs_{method}", shared(d, ["full", method])


def aggregate_scheme(d, scheme, is_fivefold, allow_incomplete=False):
    score_rows, oof_rows, station_rows, station_split_rows, trim_rows, coverage = [], [], [], [], [], []
    expected_splits = [(0, f) for f in range(5)] if is_fivefold else [(r, 0) for r in range(10)]
    if allow_incomplete:
        expected_splits = sorted({tuple(x) for x in d[SPLITKEY].to_numpy()})
    for support, comparison, base in support_sets(d):
        if base.empty:
            continue
        available_methods = sorted(base.method.unique())
        for domain in DOMAINS:
            z = base[mask_domain(base, domain)]
            for method in available_methods:
                a = z[z.method.eq(method)]
                meta = dict(scheme=scheme, support=support, comparison=comparison, domain=domain, method=method)
                coverage.append({**meta, "prediction_rows": len(a), "unique_observations": len(a[OBSKEY].drop_duplicates()),
                                 "gauges": a.station_id.nunique(), "available_splits": len(a[SPLITKEY].drop_duplicates())})
                for repeat, fold in expected_splits:
                    g = a[a.repeat_id.eq(repeat) & a.fold_id.eq(fold)]
                    score_rows.append({**meta, "repeat_id": repeat, "fold_id": fold, **metrics(g)})
                    # Residual-based removal is retrospective, never an input-QC rule.
                    if support in ["native", "requested_three_common"]:
                        nremove = int(math.ceil(.01 * len(g))) if len(g) else 0
                        residual = (g.Q_calculated - g.Q_observed).abs().to_numpy()
                        order = np.argsort(-residual, kind="stable")
                        baseline = metrics(g)
                        kept = g.iloc[order[nremove:]]
                        removed_sse = float(np.sum(residual[order[:nremove]] ** 2))
                        trim_rows.append({**meta, "repeat_id": repeat, "fold_id": fold,
                            "residual_rule": "remove_exact_ceil_1_percent_largest_absolute_residuals_without_refitting",
                            "n_removed": nremove, "removed_fraction": nremove / len(g) if len(g) else np.nan,
                            "removed_SSE_fraction": removed_sse / baseline["SSE"] if baseline["SSE"] > 0 else np.nan,
                            "baseline_NSE": baseline["NSE"], "baseline_RMSE": baseline["RMSE"], **metrics(kept)})
                if is_fivefold:
                    if a.duplicated(OBSKEY).any():
                        raise AssertionError("A five-fold observation was validated more than once")
                    unique = a
                    reducer = "single_out_of_fold_prediction"
                else:
                    unique = a.groupby(OBSKEY, as_index=False, observed=True).agg(
                        Q_observed=("Q_observed", "first"), Q_calculated=("Q_calculated", "median"))
                    reducer = "median_of_available_repeated_validation_predictions_sensitivity"
                oof_rows.append({**meta, "aggregation": reducer, **metrics(unique)})
                if not is_fivefold:
                    oof_rows.append({**meta, "aggregation": "literal_stacked_repeated_predictions_sensitivity", **metrics(a)})
                if support in ["native", "all_methods_common", "requested_three_common", "hydraulic_ablation_common"]:
                    for sid, g in unique.groupby("station_id", observed=True, sort=False):
                        station_rows.append({**meta, "station_id": sid, "aggregation": reducer,
                            "NSE_rule": "at_least_5_pairs_and_nonconstant_observed_and_predicted_Q",
                            **metrics(g, minimum_pairs=5, require_prediction_variation=True)})
                    for (sid, repeat, fold), g in a.groupby(["station_id"] + SPLITKEY, observed=True, sort=False):
                        station_split_rows.append({**meta, "station_id": sid, "repeat_id": repeat, "fold_id": fold,
                            "NSE_rule": "legacy_at_least_5_pairs_and_nonconstant_observed_and_predicted_Q",
                            **metrics(g, minimum_pairs=5, require_prediction_variation=True)})
    return {"cross_gauge_metrics_by_split": pd.DataFrame(score_rows),
            "unique_observation_and_stacked_metrics": pd.DataFrame(oof_rows),
            "station_unique_observation_metrics": pd.DataFrame(station_rows),
            "station_metrics_by_split_legacy": pd.DataFrame(station_split_rows),
            "extreme_residual_metrics_by_split": pd.DataFrame(trim_rows), "comparison_coverage": pd.DataFrame(coverage)}


def summarize(frame, group, columns):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for key, g in frame.groupby(group, observed=True, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group, key))
        for col in columns:
            row.update({f"{col}_{k}": v for k, v in distribution(g[col]).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_station_medians(frame):
    keys = ["scheme", "support", "comparison", "domain", "method", "station_id"]
    out = []
    for key, g in frame.groupby(keys, observed=True, dropna=False, sort=False):
        out.append({**dict(zip(keys, key)), "NSE": g.NSE.median(), "RMSE": g.RMSE.median(),
                    "valid_NSE_splits": int(g.NSE.notna().sum()), "prediction_rows": int(g.nobs.sum()),
                    "aggregation": "station_median_of_eligible_split_scores_legacy"})
    return pd.DataFrame(out)


def station_distributions(frame, legacy=False):
    keys = ["scheme", "support", "comparison", "domain", "method"]
    if "aggregation" in frame:
        keys.append("aggregation")
    rows = []
    for key, g in frame.groupby(keys, observed=True, dropna=False, sort=False):
        row = dict(zip(keys, key))
        valid = g.NSE[np.isfinite(g.NSE)]
        row.update(stations_with_predictions=len(g), eligible_NSE_stations=len(valid),
                   positive_NSE_fraction=float(valid.gt(0).mean()) if len(valid) else np.nan,
                   negative_NSE_fraction=float(valid.lt(0).mean()) if len(valid) else np.nan)
        for metric in ["NSE", "RMSE"]:
            row.update({f"{metric}_{k}": v for k, v in distribution(g[metric]).items()})
        if legacy:
            row["median_eligible_splits_per_station"] = float(g.valid_NSE_splits.median())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gains(frame):
    pairs = [("observed_fixed", "corrected_fixed", "width_correction_without_Q_calibration"),
             ("corrected_fixed", "full", "joint_calibration_after_width_correction"),
             ("observed_fixed", "full", "combined_changes")]
    out = []
    sub = frame[frame.support.eq("requested_three_common")]
    for (scheme, domain), g in sub.groupby(["scheme", "domain"], observed=True):
        for before, after, label in pairs:
            b = g[g.method.eq(before)]
            a = g[g.method.eq(after)]
            both = b.merge(a, on=SPLITKEY, suffixes=("_before", "_after"), validate="one_to_one")
            if not np.array_equal(both.nobs_before.to_numpy(), both.nobs_after.to_numpy()):
                raise AssertionError("Paired improvement requires identical observations")
            for metric, values in [("NSE_gain", both.NSE_after - both.NSE_before),
                ("RMSE_reduction_percent", 100 * (1 - both.RMSE_after / both.RMSE_before))]:
                out.append(dict(scheme=scheme, domain=domain, comparison=label, metric=metric, **distribution(values)))
    return pd.DataFrame(out)


def intersection_sensitivity(new, old):
    """Compare designs on common station sets, and separately common observations.

    Domain boundaries remain design-specific.  Equal station support does not
    make the 80/20 and 70/30 training experiments statistically equivalent.
    """
    rows, counts = [], []
    for method in sorted(set(new.method) & set(old.method)):
        n = new[new.method.eq(method)]
        o = old[old.method.eq(method)]
        stations = set(n.station_id) & set(o.station_id)
        observation_keys = pd.MultiIndex.from_frame(n[OBSKEY].drop_duplicates()).intersection(
            pd.MultiIndex.from_frame(o[OBSKEY].drop_duplicates()))
        for support in ["old_new_common_stations", "old_new_common_observations"]:
            for scheme, data in [("wse_stratified_5fold", n), ("wse_stratified_10_repeats", o)]:
                if support.endswith("stations"):
                    data = data[data.station_id.isin(stations)]
                else:
                    keep = pd.MultiIndex.from_frame(data[OBSKEY]).isin(observation_keys)
                    data = data[keep]
                for domain in DOMAINS:
                    z = data[mask_domain(data, domain)]
                    counts.append(dict(scheme=scheme, support=support, method=method, domain=domain,
                        ngauges=z.station_id.nunique(), n_unique_observations=len(z[OBSKEY].drop_duplicates()),
                        n_prediction_rows=len(z)))
                    expected = [(0, f) for f in range(5)] if scheme.endswith("5fold") else [(r, 0) for r in range(10)]
                    for repeat, fold in expected:
                        g = z[z.repeat_id.eq(repeat) & z.fold_id.eq(fold)]
                        rows.append(dict(scheme=scheme, support=support, method=method, domain=domain,
                                         repeat_id=repeat, fold_id=fold, **metrics(g)))
    return pd.DataFrame(rows), pd.DataFrame(counts)


def validate_assignments(path, d, allow_incomplete=False):
    result = {"path": str(path), "available": Path(path).exists()}
    if not result["available"]:
        return result
    a = read(path)
    if "validation_scheme" in a:
        a = a[a.validation_scheme.eq("wse_stratified_5fold")]
    for key in OBSKEY:
        a[key] = a[key].str.strip().str.replace(r"\.0$", "", regex=True)
    if "repeat_id" not in a:
        a["repeat_id"] = 0
    for c in SPLITKEY:
        a[c] = a[c].astype(int)
    role = "role" if "role" in a else "sample_role" if "sample_role" in a else None
    # New runner stores each observation once with its validation-fold label.
    # Older assignment tables explicitly repeat training/validation role rows.
    val = a[a[role].eq("validation")].copy() if role else a.copy()
    if val.duplicated(OBSKEY).any():
        raise AssertionError("Assignments validate an observation in multiple folds")
    if a.duplicated(KEY + ([role] if role else [])).any():
        raise AssertionError("Duplicate assignment rows")
    if role:
        leak = a.pivot_table(index=KEY, columns=role, aggfunc="size", fill_value=0)
        if {"training", "validation"}.issubset(leak.columns):
            if ((leak.training > 0) & (leak.validation > 0)).any():
                raise AssertionError("Training/validation leakage within a fold")
    membership = val[KEY].drop_duplicates().assign(_assigned=True)
    check = d[KEY].drop_duplicates().merge(membership, on=KEY, how="left", validate="one_to_one")
    if check._assigned.isna().any():
        raise AssertionError("Predictions lack matching validation assignment")
    qcpass = val[val.qc_group.eq("pass")].copy() if "qc_group" in val else val.copy()
    station_folds = qcpass.groupby("station_id", observed=True).fold_id.nunique()
    if not allow_incomplete and len(station_folds) and not station_folds.eq(5).all():
        raise AssertionError("An assigned station lacks one of the five validation folds")
    expected = qcpass[KEY].drop_duplicates()
    full = d[d.method.eq("full")][KEY].drop_duplicates()
    coverage = expected.merge(full.assign(_predicted=True), on=KEY, how="left", validate="one_to_one")
    missing = coverage[coverage._predicted.isna()]
    extra = full.merge(expected.assign(_expected=True), on=KEY, how="left", validate="one_to_one")
    unexpected = extra[extra._expected.isna()]
    missing_groups = missing.groupby(["station_id"] + SPLITKEY, observed=True).size().reset_index(name="missing_full_validation_predictions")
    result.update(validation_observations=len(val), validation_stations=val.station_id.nunique(),
                  unique_validation_assignment=True, no_within_fold_overlap=True,
                  all_prediction_keys_assigned=True, expected_full_QC_pass_predictions=len(expected),
                  observed_full_predictions=len(full), missing_full_predictions=len(missing),
                  unexpected_full_predictions=len(unexpected),
                  full_model_complete_coverage=not len(missing) and not len(unexpected),
                  missing_full_by_station_fold=missing_groups.to_dict("records"))
    return safe(result)


def definitions():
    return {
        "primary": "Compute NSE/RMSE across all gauge validation pairs separately for each of five folds. Report median +/- IQR/2 across the five scores using linear-interpolation quartiles. No >=5-pair gauge exclusion is applied to these national scores.",
        "new_unique_observation": "A standard five-fold OOF prediction is used once per station/observation/method. Joining the five folds produces one national OOF score and one OOF score per gauge. These single scores are distinct from the median of five national fold scores.",
        "station_OOF_eligibility": "At least five valid OOF pairs with variation in both observed and predicted Q, matching the original manuscript's gauge eligibility rule. Each station has equal weight in the distribution of station scores. Excluding constant predictions is a comparability convention, not a mathematical requirement of NSE.",
        "legacy_station_eligibility": "At least five finite pairs in an individual split, with both observed and predicted variation, followed by a per-station median of eligible split scores. With 20% validation, this rule omits many gauges; it is a sensitivity rather than the principal standard five-fold station summary.",
        "finite_filter": "Finite observed and predicted Q with observed Q > 0. Negative finite predictions are retained, following the old national-score rule. NSE additionally needs >=2 pairs and positive observed SST.",
        "native_domain": "Below/above are strictly below/above each method's training WSE min/max. Exact boundaries belong to within. Outside is the union of above and below.",
        "shared_domain": "Pairwise/all-method/requested-three/hydraulic comparisons first require identical observation keys and use the full model training WSE boundaries for the common domain.",
        "extreme_residual": "Retrospective deletion of exactly ceil(1% x n) largest absolute residuals per method/domain/split, stable original-order tie breaking, without refitting. This requires observed discharge and is not an operational QC screen.",
        "old_unique_sensitivity": "Old 10-repeat predictions may recur for one overpass. The median available validation prediction per observation is an ensemble sensitivity and is not equivalent to a single out-of-fold prediction. Domain selection precedes the median reduction.",
        "comparison_limit": "Five-fold 80/20 and old ten-repeat 70/30 CV differ in training fraction, allocation and observation coverage. Score differences do not isolate repetition count or prove improvement. Common-station and common-observation supports are additional sensitivities.",
        "uncertainty": "IQR/2 summarizes variation of split scores; it is neither a confidence interval nor generally the distance from the median to either actual quartile.",
        "privacy": "Prediction inputs contain restricted gauge Q and remain private. This script writes no observation identifiers, observed/predicted Q rows, or times into aggregate output CSVs.",
    }


def render_readout(tables, output, old_note, status="PASS"):
    summary = tables["cross_gauge_summary"]
    station = tables["station_unique_observation_summary"]
    lines = ["# V15 标准五折 WSE 分层交叉验证", "", f"计算与完整性检查状态：**{status}**。", "",
        "每个站点的观测分为五个互斥验证折，每次约80%率定、20%验证，每条观测最多进入一次验证。下表为各折跨站NSE/RMSE的中位数 ± IQR/2，不是站点NSE的中位数。", "",
        "|方法|分析域|有效折数|验证对数中位数|NSE|RMSE (m³ s⁻¹)|", "|---|---|---:|---:|---:|---:|"]
    selected = summary[summary.scheme.eq("wse_stratified_5fold") & summary.support.eq("native") & summary.domain.isin(["all", "within", "above", "below"])]
    for method in METHODS:
        for _, r in selected[selected.method.eq(method)].iterrows():
            lines.append(f"|{method}|{r.domain}|{int(r.NSE_N)}|{r.nobs_median:.0f}|{r.NSE_median:.3f} ± {r.NSE_half_IQR:.3f}|{r.RMSE_median:.1f} ± {r.RMSE_half_IQR:.1f}|")
    lines += ["", "上述原生样本表每种方法可用的观测可能不同。直接比较改正、率定或消融收益时，应使用requested_three_common、hydraulic_ablation_common或pairwise_common共享样本表。", "",
              "## 站点表现", "", "主站点诊断把每站五折的独立验证预测拼接后只计算一次NSE，沿用论文资格要求：至少五个有效观测，且实测与预测Q均存在变化。每折先计算站点NSE再取站点中位数的旧口径另存；20%验证使这一旧口径更容易缺少五对观测。", ""]
    s = station[station.scheme.eq("wse_stratified_5fold") & station.support.eq("native") & station.domain.eq("all")]
    for _, r in s.iterrows():
        lines.append(f"- {r.method}: {int(r.eligible_NSE_stations)} 个站点的OOF NSE中位数为 {r.NSE_median:.3f} ± {r.NSE_half_IQR:.3f}，NSE>0比例为 {100*r.positive_NSE_fraction:.1f}%。")
    lines += ["", "## 解读边界", "", "- 水位域反映相对于各折率定范围的水力内插/外推，不能据此声称已进行按日期前后划分的未来事件验证。", "- 删除最大1%残差需要已知实测Q，仅用于诊断误差集中程度。", "- 新五折与旧十次划分同时改变了率定比例和观测分配；差异不能仅归因于折数。", f"- 旧结果对比状态：{old_note}", "", "所有公开汇总表均排除逐观测实测Q、预测Q、时间和观测ID。论文未在本步骤更新。", ""]
    (output / "readout.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "private/validation_predictions_RESTRICTED.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "tables")
    parser.add_argument("--old-predictions", type=Path)
    parser.add_argument("--skip-old", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="Smoke/subset runs only; mark output incomplete")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    new, exclusions, count = read_predictions(args.predictions, "wse_stratified_5fold",
        ROOT / "tables/parameter_estimates.csv", "wse_stratified_5fold")
    if not args.allow_incomplete:
        if not set(new.method).issubset(set(METHODS)):
            raise AssertionError(f"Unexpected methods: {set(new.method)-set(METHODS)}")
        if set(new.repeat_id) != {0} or not set(new.fold_id).issubset(set(range(5))):
            raise AssertionError("New prediction identifiers are not standard five-fold")
    if new.duplicated(["method"] + OBSKEY).any():
        raise AssertionError("New five-fold prediction repeats an observation across folds")
    assignment_path = ROOT / "private/split_assignments_RESTRICTED.csv"
    if not assignment_path.exists():
        assignment_path = ROOT / "tables/split_assignments.csv"
    assignment_check = validate_assignments(assignment_path, new, args.allow_incomplete)
    all_parts = [aggregate_scheme(new, "wse_stratified_5fold", True, args.allow_incomplete)]
    sources = [file_hash(args.predictions)]
    old_note = "未请求旧结果对比。"
    old = None
    if not args.skip_old:
        oldpath = args.old_predictions
        if oldpath is None and OLD_LOCAL is not None and (OLD_LOCAL / "QA.json").exists():
            oldpath = Path(json.loads((OLD_LOCAL / "QA.json").read_text(encoding="utf-8-sig"))["source"])
        if oldpath is not None and oldpath.exists():
            old, e, _ = read_predictions(oldpath, "wse_stratified_10_repeats", oldpath.parent / "parameter_estimates.csv", "wse_stratified")
            sources.append(file_hash(oldpath))
            extras = OLD_CONTROLS / "private/new_controls_validation_predictions_RESTRICTED.csv" if OLD_CONTROLS is not None else None
            if extras is not None and extras.exists():
                control, e2, _ = read_predictions(extras, "wse_stratified_10_repeats",
                    OLD_CONTROLS / "tables/fixed_parameter_configurations.csv", "wse_stratified")
                old = pd.concat([old, control], ignore_index=True)
                e = pd.concat([e, e2], ignore_index=True)
                sources.append(file_hash(extras))
            exclusions = pd.concat([exclusions, e], ignore_index=True)
            all_parts.append(aggregate_scheme(old, "wse_stratified_10_repeats", False))
            old_note = "已按同一汇总脚本重新计算旧十次划分，并提供相同站点及相同观测的敏感性比较。"
        else:
            old_note = "旧逐观测输入不可访问，未重算旧结果及站点交集；此前聚合表仍保留原处。"
    tables = {name: pd.concat([part[name] for part in all_parts], ignore_index=True) for name in all_parts[0]}
    group = ["scheme", "support", "comparison", "domain", "method"]
    tables["cross_gauge_summary"] = summarize(tables["cross_gauge_metrics_by_split"], group, METRICS)
    tables["station_unique_observation_summary"] = station_distributions(tables["station_unique_observation_metrics"])
    tables["station_median_metrics_legacy"] = legacy_station_medians(tables["station_metrics_by_split_legacy"])
    tables["station_median_summary_legacy"] = station_distributions(tables["station_median_metrics_legacy"], legacy=True)
    tables["extreme_residual_summary"] = summarize(tables["extreme_residual_metrics_by_split"],
        group + ["residual_rule"], ["n_removed", "removed_fraction", "removed_SSE_fraction", "baseline_NSE", "baseline_RMSE", "NSE", "RMSE"])
    tables["paired_gains"] = paired_gains(tables["cross_gauge_metrics_by_split"])
    tables["finite_pair_support"] = exclusions
    tables["missing_full_predictions_by_station_fold"] = pd.DataFrame(
        assignment_check.get("missing_full_by_station_fold", []),
        columns=["station_id", "repeat_id", "fold_id", "missing_full_validation_predictions"])
    if old is not None:
        inter, cover = intersection_sensitivity(new, old)
        tables["old_new_intersection_metrics_by_split"] = inter
        tables["old_new_intersection_coverage"] = cover
        tables["old_new_intersection_summary"] = summarize(inter, ["scheme", "support", "method", "domain"], METRICS)
    for name, frame in tables.items():
        save_csv(args.output / f"{name}.csv", frame)
    rr = tables["cross_gauge_metrics_by_split"]
    populated = rr.nobs.gt(0)
    identities = {
        "RMSE_squared_equals_SSE_div_n": bool(np.allclose(rr.loc[populated, "RMSE"] ** 2, rr.loc[populated, "SSE"] / rr.loc[populated, "nobs"], equal_nan=True)),
        "SST_decomposition": bool(np.allclose(rr.SST, rr.within_gauge_SST + rr.between_gauge_SST, equal_nan=True)),
        "new_each_observation_validated_at_most_once_per_method": True,
        "aggregate_outputs_exclude_observation_Q_and_keys": True,
    }
    nse_eligible = rr.nobs.ge(2) & rr.SST.gt(0)
    identities["NSE_equals_1_minus_SSE_div_SST"] = bool(np.allclose(
        rr.loc[nse_eligible, "NSE"], 1 - rr.loc[nse_eligible, "SSE"] / rr.loc[nse_eligible, "SST"], equal_nan=True))
    if not all(identities.values()):
        raise AssertionError(f"Metric identities failed: {identities}")
    complete_full = assignment_check.get("available", False) and assignment_check.get("full_model_complete_coverage", False)
    missing_methods = sorted(set(METHODS) - set(new.method))
    failure_path = ROOT / "checks/failures.csv"
    failed_fits = 0
    if failure_path.exists() and failure_path.stat().st_size > 5:
        try:
            failed_fits = len(read(failure_path))
        except pd.errors.EmptyDataError:
            pass
    status = "INCOMPLETE_SUBSET" if args.allow_incomplete else (
        "INCOMPLETE_FULL_MODEL_COVERAGE" if not complete_full else
        "INCOMPLETE_METHOD_COVERAGE" if missing_methods or failed_fits else "PASS")
    report = dict(status=status,
                  source_rows_before_finite_filter=count, new_valid_prediction_rows=len(new),
                  new_methods=sorted(new.method.unique()), missing_methods=missing_methods,
                  recorded_failed_fits=failed_fits, sources=sources,
                  definitions=definitions(), checks=identities, assignments=assignment_check,
                  old_comparison=old_note, output_files=sorted(f"{name}.csv" for name in tables))
    (args.output / "summary.json").write_text(json.dumps(safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    render_readout(tables, args.output, old_note, status=report["status"])
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "new_valid_prediction_rows": len(new), "methods": len(new.method.unique()),
                      "old_comparison": old_note}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
