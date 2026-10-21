"""Independent exact-equivalence checks; never writes run checkpoints or Q rows."""
from pathlib import Path
import copy
import hashlib
import json
import math
import sys
import time

import run_fivefold as r
np, pd = r.np, r.pd
ROOT = r.ROOT


def exact(a, b, path="root"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            exact(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype, path
        assert np.array_equal(a, b, equal_nan=True), path
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for j, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f"{path}[{j}]")
    elif isinstance(a, (float, np.floating)) and np.isnan(a):
        assert isinstance(b, (float, np.floating)) and np.isnan(b), path
    else:
        assert a == b, f"Mismatch at {path}"


def collapse_check(x, y, label):
    original = r._original_collapse(x, y)
    wrapped = r.cached_collapse(x, y)
    exact(original, wrapped, label)
    for output in wrapped:
        if len(output):
            output[0] = -1234567.
    protected = r.cached_collapse(x, y)
    exact(original, protected, label + ".mutation_protection")
    return {"label": label, "input_rows": len(x), "anchor_rows": len(original[0]),
            "exact": True, "cached_return_mutation_protected": True}


def main():
    started = time.monotonic()
    original_code_sha = r.digest(Path(r.__file__))
    report = {"status": "RUNNING", "runner_sha256": original_code_sha,
              "no_run_checkpoints_written": True,
              "no_observation_level_discharge_exported": True}
    checks = []
    synthetic = [
        (np.array([3., 1., 2., 1., 2., 3.]), np.array([9., 2., 5., 4., 7., 11.]), "duplicate_WSE"),
        (np.array([1., np.nan, 2., np.inf, -np.inf, 2.]), np.array([0., 4., -3., 5., 6., np.nan]), "nonfinite_and_width_floor"),
        (np.array([], dtype=float), np.array([], dtype=float), "empty"),
        (pd.Series([1., 2., 1.], index=[10, 20, 30]), pd.Series([3., 4., 5.], index=[10, 20, 30]), "aligned_nondefault_series_index"),
        (pd.Series([1., 2.], index=[10, 20]), pd.Series([3., 4.], index=[20, 10]), "misaligned_series_fallback"),
        (pd.Series([1., 2., 1.], index=[10, 10, 30]), pd.Series([3., 4., 5.], index=[10, 10, 30]), "aligned_duplicate_series_index"),
    ]
    for x, y, label in synthetic:
        checks.append(collapse_check(x, y, label))
    chosen = None
    for sid in ["401050", "502300", "705700"]:
        source = Path(r.CFG.data_dir) / f"{sid}.csv"
        base, _, _, _ = r.v15.load_station_with_global_qc(source, r.CFG.aggregate_nodes_by_overpass)
        assignment = r.split_station(base, sid)
        assert assignment is not None
        for fold in range(5):
            trids = assignment.loc[assignment.fold_id.ne(fold), "obs_id"].tolist()
            teids = assignment.loc[assignment.fold_id.eq(fold), "obs_id"].tolist()
            pairs, _ = r.v15.make_split_frame_versions(base, trids, teids)
            for method in ["full", "no_monotonic", "no_hampel", "width_train_only", "monotonic_only"]:
                train, _ = pairs[method]
                checks.append(collapse_check(train.wse_model, train.width_model,
                    f"station_{sid}_fold_{fold}_{method}"))
            if sid == "502300" and fold == 0:
                chosen = (sid, fold, source, pairs, assignment)
    report["collapse_checks"] = checks
    assert chosen is not None
    sid, fold, source, pairs, assignment = chosen
    train, validation = pairs["full"]
    metadata = r.core.metadata_lookup(Path(r.CFG.metadata_file))
    n_ref = r.v15.pipeline.finite_number(metadata.get(sid, {}).get("N_River"))
    if not np.isfinite(n_ref) or n_ref <= 0:
        valid = [r.v15.pipeline.finite_number(v.get("N_River")) for v in metadata.values()]
        valid = [v for v in valid if np.isfinite(v) and v > 0]
        n_ref = np.median(valid) if valid else .035
    n_ref = float(np.clip(n_ref, r.core.N_MIN, r.core.N_MAX))
    seed = r.v15.pipeline.stable_seed(sid, "wse_5fold_fit", fold, base=r.CFG.seed)
    results = {}
    for enabled in [False, True]:
        r.core.collapse_wse_width = r.cached_collapse if enabled else r._original_collapse
        r._cached_collapse.cache_clear()
        t0 = time.monotonic()
        fit = r.v15.fit_method("full", train, r.CFG, seed, n_ref)
        assert fit is not None
        prediction = r.v15.pipeline.predict_method("full", fit, validation, r.SCHEME, fold, 0, "validation")
        results[enabled] = (fit, prediction, time.monotonic() - t0)
        print(json.dumps({"cache_enabled": enabled, "elapsed_s": results[enabled][2],
            "training_records": len(train), "validation_records": len(validation)}), flush=True)
    exact(results[False][0], results[True][0], "full_fit_all_return_fields")
    assert results[False][1].equals(results[True][1]), "Full prediction frames differ"
    full = results[True][0]
    x0 = r.fitting_input(pairs["full"][0])
    equivalent = []
    for method in sorted(r.EQUIVALENT_JOINT - {"full"}):
        tr, te = pairs[method]
        x1 = r.fitting_input(tr)
        same = x0[0].equals(x1[0]) and x0[1].equals(x1[1])
        item = {"method": method, "exact_training_inputs_equal_full": bool(same)}
        if same:
            reused = copy.deepcopy(full)
            reused.update(r.quality.curve_metadata(tr))
            reused["parameter_source"] = "station_joint_n_beta_WSE0_width0_" + method
            # Metadata updates may not alter any numerical fitted parameter.
            for key in ["n", "beta", "WSE0", "width0", "depth0", "A0", "anchor_wse", "anchor_width"]:
                exact(full[key], reused[key], f"reuse_{method}.{key}")
            item["numerical_parameters_unchanged"] = True
        equivalent.append(item)
    report.update(status="PASS", collapse_cases=len(checks),
        full_fit_cache_off_on_exact=True, all_fit_return_fields_equal=True,
        entire_prediction_dataframe_exact=True,
        tested_fit={"station_id": sid, "fold_id": fold, "training_n": len(train),
            "validation_n": len(validation), "fit_seed": seed,
            "optimizer_maxiter": r.CFG.optimizer_maxiter,
            "optimizer_popsize": r.CFG.optimizer_popsize,
            "validation_multistart_count": r.CFG.validation_multistart_count,
            "cache_off_seconds": results[False][2], "cache_on_seconds": results[True][2],
            "source_sha256": r.digest(source)},
        equivalent_training_methods=equivalent,
        cache_details="Only collapse_wse_width preprocessing cached; original function called on miss; independent output copies; actual objective and optimizer unmodified.",
        execution_seconds=time.monotonic()-started)
    assert original_code_sha == r.digest(Path(r.__file__)), "Runner changed during this audit"
    r.dump_json(ROOT / "checks/computation_equivalence.json", report)
    print(json.dumps({"status": "PASS", "collapse_cases": len(checks),
        "full_fit_and_predictions_exact": True, "output": str(ROOT / "checks/computation_equivalence.json")}), flush=True)


if __name__ == "__main__":
    main()
