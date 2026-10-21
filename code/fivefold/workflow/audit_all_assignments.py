"""Read-only audit of all input stations and expected standard fivefold rows.

Reuses the active runner's V15 loader and split_station without fitting any
model, invoking initialize(), or reading/writing its running checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import run_fivefold as run

np, pd = run.np, run.pd
ROOT = run.ROOT
from _paths import external_path
OLD_SUMMARY = external_path("CSWOT_LEGACY_QC_SUMMARY")
OLD_FLAGS = external_path("CSWOT_LEGACY_QC_FLAGS")
FLAGS = ["global_hampel_wse", "global_hampel_width", "global_hampel_log_slope", "global_invalid_slope", "global_qc_pass"]


def normalized_flags(frame):
    """Parse explicit boolean tokens; astype(bool) treats 'False' as True."""
    output = pd.DataFrame(index=frame.index)
    for column in FLAGS:
        tokens = frame[column].astype("string").str.strip().str.lower()
        valid = tokens.isin(["true", "false", "1", "0", "1.0", "0.0"])
        if not valid.all():
            raise ValueError(f"Missing or unrecognized QC boolean tokens in {column}: {tokens.loc[~valid].value_counts(dropna=False).to_dict()}")
        output[column] = tokens.isin(["true", "1", "1.0"]).to_numpy(dtype=bool)
    return output


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"N": 0}
    return {"N": len(values), "min": float(values.min()), "q25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, .75)),
            "max": float(values.max()), "mean": float(values.mean())}


def write_table(path, frame):
    if {"Q", "Q_observed", "Q_calculated", "obs_id", "time", "time_dt"}.intersection(frame.columns):
        raise ValueError(f"Restricted observation fields in aggregate audit table: {path}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    started = time.monotonic()
    checks = ROOT / "checks"
    private = ROOT / "private"
    checks.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    old_summary = pd.read_csv(OLD_SUMMARY, dtype={"station_id": "string"}) if OLD_SUMMARY is not None else pd.DataFrame(columns=["station_id"])
    old_flags = pd.read_csv(OLD_FLAGS, dtype={"station_id": "string", "obs_id": "string"},
                            usecols=["station_id", "obs_id", *FLAGS]) if OLD_FLAGS is not None else pd.DataFrame(columns=["station_id", "obs_id", *FLAGS])
    old_flag_dtypes = {column: str(old_flags[column].dtype) for column in FLAGS}
    old_flags.loc[:, FLAGS] = normalized_flags(old_flags)
    old_summary["station_id"] = old_summary.station_id.map(run.core.normalize_station_id)
    old_flags["station_id"] = old_flags.station_id.map(run.core.normalize_station_id)
    assert old_summary.station_id.is_unique
    assert not old_flags.duplicated(["station_id", "obs_id"]).any()
    old_by_station = {sid: x.set_index("obs_id") for sid, x in old_flags.groupby("station_id", sort=False)}
    old_counts = old_summary.set_index("station_id")
    files = sorted(Path(run.CFG.data_dir).glob("*.csv"))
    station_rows, fold_rows, stratum_rows, expected = [], [], [], []
    mismatches, errors = [], []
    overlap_record_count = 0
    for number, path in enumerate(files, start=1):
        sid = run.core.normalize_station_id(path.stem)
        try:
            base, audit, _, overpass = run.v15.load_station_with_global_qc(path, run.CFG.aggregate_nodes_by_overpass)
            assert not base.obs_id.duplicated().any()
            counts = {"complete_observations": len(base), "retained_observations": int(base.global_qc_pass.sum())}
            counts.update({flag + "_count": int(base[flag].sum()) for flag in FLAGS[:-1]})
            old_present = sid in old_counts.index
            summary_equal = None
            if old_present:
                summary_equal = all(counts[c] == int(old_counts.loc[sid, c]) for c in counts)
                if not summary_equal:
                    mismatches.append({"station_id": sid, "kind": "station_count", "new": counts,
                                       "old": {c: int(old_counts.loc[sid, c]) for c in counts}})
            old_records = old_by_station.get(sid)
            flags_equal = None
            overlap = 0
            if old_records is not None:
                current = base.set_index("obs_id")
                overlap = len(current.index.intersection(old_records.index))
                overlap_record_count += overlap
                same_keys = set(current.index) == set(old_records.index)
                mismatch_by_flag = {}
                if same_keys:
                    current_boolean = normalized_flags(current.loc[old_records.index, FLAGS])
                    old_boolean = normalized_flags(old_records[FLAGS])
                    mismatch_by_flag = {column: int(np.count_nonzero(current_boolean[column].to_numpy() != old_boolean[column].to_numpy())) for column in FLAGS}
                flags_equal = same_keys and sum(mismatch_by_flag.values()) == 0
                if not flags_equal:
                    mismatches.append({"station_id": sid, "kind": "observation_QC_flags", "identical_observation_keys": same_keys,
                                       "current_records": len(current), "old_records": len(old_records), "overlap_records": overlap,
                                       "mismatched_values_per_QC_flag": mismatch_by_flag})
            assigned = run.split_station(base, sid)
            eligible = assigned is not None
            missing_old_explanation = "" if old_present else ("No complete paired overpass records in either input reconstruction or the old station summary" if not len(base) else "Current station has complete paired overpasses but no row in the old summary; inspect old run cohort/audit availability")
            station_rows.append({"station_id": sid, "raw_input_rows": audit["raw_row_count"], **counts,
                                 "qc_fail_observations": len(base) - counts["retained_observations"], "eligible_fivefold": eligible,
                                 "old_summary_present": old_present, "old_summary_counts_equal": summary_equal,
                                 "old_QC_records_present": old_records is not None, "old_QC_flags_equal": flags_equal,
                                 "old_QC_overlap_records": overlap, "missing_old_explanation": missing_old_explanation})
            if not eligible:
                continue
            assigned["station_id"] = sid
            assigned["validation_scheme"] = run.SCHEME
            assigned["repeat_id"] = 0
            assert len(assigned) == len(base) and assigned.obs_id.nunique() == len(base)
            assert set(assigned.obs_id) == set(base.obs_id)
            assert assigned.fold_id.isin(range(5)).all()
            expected.append(assigned)
            qpassed = assigned.loc[assigned.qc_group.eq("pass")]
            c = qpassed.fold_id.value_counts().reindex(range(5), fill_value=0)
            assert c.max() - c.min() <= 1 and c.min() >= 1
            for group, stratum in assigned.groupby(["qc_group", "wse_stratum"], sort=False):
                sc = stratum.fold_id.value_counts().reindex(range(5), fill_value=0)
                assert sc.max() - sc.min() <= 1
                stratum_rows.append({"station_id": sid, "qc_group": group[0], "wse_stratum": group[1],
                                     "stratum_observations": len(stratum), "min_validation_per_fold": int(sc.min()),
                                     "max_validation_per_fold": int(sc.max()), "fold_count_difference": int(sc.max() - sc.min())})
            joined = base.merge(assigned[["obs_id", "fold_id"]], on="obs_id", validate="one_to_one")
            # Verify full-method training exactly as the runner does, without
            # building other methods' curves or invoking any optimizer.
            for fold in range(5):
                raw_train, raw_test = joined.loc[joined.fold_id.ne(fold)], joined.loc[joined.fold_id.eq(fold)]
                assert not set(raw_train.obs_id).intersection(raw_test.obs_id)
                train, test = run.quality.retain(raw_train), run.quality.retain(raw_test)
                curve = run.quality.curve(train)
                corrected = run.quality.apply_curve(train, curve, training=True)
                prepared, _ = run.core.prepare_training(corrected, True)
                assert prepared is not None and len(prepared) >= 5
                assert len(test) >= 1
                assert len(train) + len(test) == len(qpassed)
                fold_rows.append({"station_id": sid, "fold_id": fold, "qc_pass_total": len(qpassed),
                                  "full_train_records": len(train), "full_prepared_train_records": len(prepared),
                                  "full_validation_records": len(test), "full_training_fraction": len(train) / len(qpassed),
                                  "raw_train_records": len(raw_train), "raw_validation_records": len(raw_test),
                                  "no_hampel_train_records": int((~raw_train.global_invalid_slope).sum()),
                                  "no_hampel_validation_records": int((~raw_test.global_invalid_slope).sum()),
                                  "training_WSE_unique": train.wse_model.nunique()})
        except Exception as exc:
            errors.append({"station_id": sid, "reason": str(exc)})
        if number % 100 == 0 or number == len(files):
            print(json.dumps({"processed_files": number, "total_files": len(files), "eligible_stations": len(expected),
                              "elapsed_s": round(time.monotonic() - started, 1), "errors": len(errors)}), flush=True)
    stations = pd.DataFrame(station_rows)
    folds = pd.DataFrame(fold_rows)
    strata = pd.DataFrame(stratum_rows)
    assignments = pd.concat(expected, ignore_index=True)
    assert not assignments.duplicated(["station_id", "obs_id"]).any()
    assignments.to_csv(private / "expected_fold_assignments_RESTRICTED.csv", index=False, encoding="utf-8-sig")
    write_table(checks / "all_station_assignment_eligibility.csv", stations)
    write_table(checks / "all_station_expected_fold_counts.csv", folds)
    write_table(checks / "all_station_stratum_balance.csv", strata)
    pd.DataFrame(errors, columns=["station_id", "reason"]).to_csv(checks / "all_assignment_load_errors.csv", index=False, encoding="utf-8-sig")
    (checks / "all_assignment_QC_mismatches.json").write_text(json.dumps(safe(mismatches), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    eligible = stations.loc[stations.eligible_fivefold]
    counts_by_fold = folds.groupby("fold_id").agg(expected_full_validation_rows=("full_validation_records", "sum"),
                                                    expected_full_train_rows=("full_train_records", "sum"),
                                                    expected_raw_validation_rows=("raw_validation_records", "sum"),
                                                    expected_no_hampel_validation_rows=("no_hampel_validation_records", "sum"),
                                                    contributing_stations=("station_id", "nunique")).reset_index()
    write_table(checks / "all_station_expected_totals_by_fold.csv", counts_by_fold)
    old_ids = set(old_summary.station_id)
    new_ids = set(stations.station_id)
    report = {
        "status": "PASS" if not errors and not mismatches else "REVIEW_REQUIRED",
        "validation_scheme": run.SCHEME, "input_files": len(files), "successfully_loaded_stations": len(stations),
        "source_input_directory": str(run.CFG.data_dir), "loader_and_splitter": str(Path(run.__file__)),
        "eligible_stations": len(eligible), "eligible_QC_pass_observations": int(eligible.retained_observations.sum()),
        "eligible_raw_complete_observations": int(eligible.complete_observations.sum()),
        "all_stations_complete_observations": int(stations.complete_observations.sum()),
        "all_stations_QC_pass_observations": int(stations.retained_observations.sum()),
        "expected_station_fold_units": len(folds), "expected_full_OOF_rows": int(folds.full_validation_records.sum()),
        "expected_no_hampel_OOF_rows": int(folds.no_hampel_validation_records.sum()),
        "expected_full_OOF_rows_equal_total_eligible_QC_pass": int(folds.full_validation_records.sum()) == int(eligible.retained_observations.sum()),
        "expected_validation_totals_by_fold": counts_by_fold.to_dict("records"),
        "full_train_records_distribution": distribution(folds.full_train_records),
        "full_prepared_train_records_distribution": distribution(folds.full_prepared_train_records),
        "full_validation_records_distribution": distribution(folds.full_validation_records),
        "full_training_fraction_distribution": distribution(folds.full_training_fraction),
        "QC_pass_observations_per_eligible_station_distribution": distribution(eligible.retained_observations),
        "all_eligible_training_sets_prepare_successfully_with_at_least_5_records": len(folds) == 5 * len(eligible) and folds.full_prepared_train_records.ge(5).all(),
        "every_eligible_raw_observation_assigned_to_exactly_one_validation_fold": len(assignments) == int(eligible.complete_observations.sum()),
        "per_station_QC_pass_validation_fold_size_difference_at_most_1": True,
        "within_each_QC_group_WSE_stratum_fold_size_difference_at_most_1": bool(strata.fold_count_difference.le(1).all()),
        "old_QC_station_summary_stations": len(old_summary), "old_QC_station_overlap": len(old_ids.intersection(new_ids)),
        "old_QC_station_summary_absent_from_current_loaded_stations": sorted(old_ids - new_ids),
        "current_stations_absent_from_old_summary": int((~stations.old_summary_present).sum()),
        "current_nonempty_stations_absent_from_old_summary": int((~stations.old_summary_present & stations.complete_observations.gt(0)).sum()),
        "current_empty_stations_absent_from_old_summary": int((~stations.old_summary_present & stations.complete_observations.eq(0)).sum()),
        "old_QC_overlap_observations": overlap_record_count,
        "old_QC_source_flag_dtypes_before_normalization": old_flag_dtypes,
        "QC_flag_comparison": "Explicit case-insensitive true/false or 1/0 token normalization; no direct astype(bool) conversion of strings. Equal observation keys are aligned before comparing boolean values.",
        "old_QC_mismatch_count": len(mismatches), "input_load_or_split_errors": len(errors),
        "old_QC_sources": [str(p) for p in [OLD_SUMMARY, OLD_FLAGS] if p is not None],
        "restricted_expected_assignments": str(private / "expected_fold_assignments_RESTRICTED.csv"),
        "audit_outputs_contain_observed_Q": False, "optimizer_run": False, "running_checkpoints_touched": False,
        "elapsed_s": time.monotonic() - started,
        "interpretation": "Expected counts describe input/split eligibility, not successful fitted predictions. Failed fits must be counted separately when reconciling the completed experiment. QC-fail records are independently assigned for no-Hampel sensitivity; primary full-model records are QC-pass only. With small station samples, exact fractions vary around 80/20.",
    }
    report = safe(report)
    (checks / "all_assignment_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
