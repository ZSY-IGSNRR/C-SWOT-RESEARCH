from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_write_csv, normalize_id_series, read_id_file
from .config import output_path
from .quality import hampel_flags


def _remove_stale_csvs(directory: Path, active_names: set[str]) -> int:
    """Remove derived station CSVs that do not belong to the current run."""
    removed = 0
    if not directory.exists():
        return removed
    for path in directory.glob("*.csv"):
        if path.name not in active_names:
            path.unlink()
            removed += 1
    return removed


def _asof_attach(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_time: str,
    right_time: str,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    left = left.sort_values(left_time).copy()
    right = right.sort_values(right_time).copy()
    return pd.merge_asof(
        left, right, left_on=left_time, right_on=right_time,
        direction="nearest", tolerance=tolerance,
    )


def _attach_reach_slope_fallback(
    merged: pd.DataFrame,
    reach: pd.DataFrame,
    *,
    tolerance: pd.Timedelta | None,
) -> pd.DataFrame:
    """Fill only missing/nonpositive reach slope from a wider time window.

    All other reach variables stay governed by the strict same-overpass match.
    This is important because copying an older official ``dschg_*`` estimate
    onto a new node observation would create a false contemporaneous product
    comparison.  The source time and time offset of every used slope are kept.
    """
    result = merged.copy()
    strict = pd.to_numeric(result.get("reach_slope"), errors="coerce")
    strict_valid = np.isfinite(strict) & strict.gt(0)
    result["reach_slope_strict"] = strict

    source_time = pd.to_datetime(
        result.get("reach_reach_time_utc"), errors="coerce", utc=True
    )
    result["reach_slope_source_time_utc"] = source_time.where(strict_valid)
    result["reach_slope_match_delta_seconds"] = pd.to_numeric(
        result.get("reach_match_delta_seconds"), errors="coerce"
    ).where(strict_valid)
    result["reach_slope_fallback_used"] = False

    if "reach_slope" not in reach or (
        tolerance is not None and tolerance <= pd.Timedelta(0)
    ):
        result["reach_slope_source"] = np.where(
            strict_valid, "strict_same_overpass", "missing"
        )
        return result

    fallback = reach[["reach_reach_time_utc", "reach_slope"]].copy()
    fallback["reach_slope_fallback"] = pd.to_numeric(
        fallback.pop("reach_slope"), errors="coerce"
    )
    fallback = fallback[
        fallback["reach_reach_time_utc"].notna()
        & np.isfinite(fallback["reach_slope_fallback"])
        & fallback["reach_slope_fallback"].gt(0)
    ].rename(columns={
        "reach_reach_time_utc": "reach_slope_fallback_time_utc"
    })
    if fallback.empty:
        result["reach_slope_source"] = np.where(
            strict_valid, "strict_same_overpass", "missing"
        )
        return result

    result["_slope_fallback_order"] = np.arange(len(result), dtype=int)
    result = _asof_attach(
        result,
        fallback,
        left_time="time_utc",
        right_time="reach_slope_fallback_time_utc",
        tolerance=tolerance,
    )
    candidate = pd.to_numeric(result["reach_slope_fallback"], errors="coerce")
    candidate_valid = np.isfinite(candidate) & candidate.gt(0)
    use_fallback = ~strict_valid.to_numpy() & candidate_valid.to_numpy()
    fallback_delta = (
        result["time_utc"] - result["reach_slope_fallback_time_utc"]
    ).abs().dt.total_seconds()
    result.loc[use_fallback, "reach_slope"] = candidate.loc[use_fallback]
    result.loc[use_fallback, "reach_slope_source_time_utc"] = result.loc[
        use_fallback, "reach_slope_fallback_time_utc"
    ]
    result.loc[use_fallback, "reach_slope_match_delta_seconds"] = fallback_delta.loc[
        use_fallback
    ]
    result.loc[use_fallback, "reach_slope_fallback_used"] = True
    result["reach_slope_source"] = np.select(
        [strict_valid.to_numpy(), use_fallback],
        ["strict_same_overpass", "nearest_reach_slope_fallback"],
        default="missing",
    )
    return result.sort_values("_slope_fallback_order").drop(
        columns=[
            "_slope_fallback_order",
            "reach_slope_fallback",
            "reach_slope_fallback_time_utc",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def attach_reach_to_nodes(config: dict) -> pd.DataFrame:
    """Attach only temporally close reach observations; never use unlimited nearest dates."""
    node_dir = output_path(config, "qc", "node_d")
    reach_dir = output_path(config, "qc", "reach_d")
    sos_dir = output_path(config, "qc", "reach_sos_v2")
    output_dir = output_path(config, "fused", "node_reach")
    output_dir.mkdir(parents=True, exist_ok=True)
    node_ids = read_id_file(
        output_path(config, "inventory", "node_ids.csv"), "node_id"
    )
    active_names = {f"{node_id}.csv" for node_id in node_ids}
    _remove_stale_csvs(output_dir, active_names)
    tolerance = pd.Timedelta(minutes=float(config["fusion"]["reach_node_time_tolerance_minutes"]))
    slope_fallback_days = config["fusion"].get(
        "reach_node_slope_fallback_tolerance_days", 0
    )
    slope_fallback_tolerance = (
        None
        if slope_fallback_days is None
        else pd.Timedelta(days=float(slope_fallback_days))
    )
    records: list[dict] = []

    for node_id in node_ids:
        node_file = node_dir / f"{node_id}.csv"
        if not node_file.exists():
            records.append({"node_id": node_id, "status": "missing_node_file"})
            continue
        try:
            node = pd.read_csv(node_file, dtype={"node_id": "string", "reach_id": "string"}, low_memory=False)
            if node.empty or "reach_id" not in node:
                records.append({"node_id": node_file.stem, "status": "empty_node"})
                continue
            node["node_id"] = normalize_id_series(node["node_id"])
            node["reach_id"] = normalize_id_series(node["reach_id"])
            node["time_utc"] = pd.to_datetime(node.get("time_str"), errors="coerce", utc=True)
            reach_id = str(node["reach_id"].dropna().iloc[0])
            reach_file = reach_dir / f"{reach_id}.csv"
            if not reach_file.exists():
                records.append({
                    "node_id": node_file.stem, "reach_id": reach_id,
                    "status": "missing_reach_file", "node_rows": len(node),
                })
                continue
            reach = pd.read_csv(reach_file, dtype={"reach_id": "string"}, low_memory=False)
            reach["reach_id"] = normalize_id_series(reach["reach_id"])
            reach["reach_time_utc"] = pd.to_datetime(reach.get("time_str"), errors="coerce", utc=True)
            reach = reach.dropna(subset=["reach_time_utc"])
            prefix_exceptions = {"reach_id"}
            reach = reach.rename(columns={
                column: f"reach_{column}"
                for column in reach.columns if column not in prefix_exceptions
            })
            merged = _asof_attach(
                node.dropna(subset=["time_utc"]), reach,
                left_time="time_utc", right_time="reach_reach_time_utc", tolerance=tolerance,
            )
            merged["reach_match_delta_seconds"] = (
                merged["time_utc"] - merged["reach_reach_time_utc"]
            ).abs().dt.total_seconds()
            merged["reach_temporal_match"] = merged["reach_reach_time_utc"].notna()
            if "sword_version" in merged and "reach_sword_version" in merged:
                node_version = merged["sword_version"].astype("string").str.strip()
                reach_version = merged["reach_sword_version"].astype("string").str.strip()
                merged["qc_node_reach_sword_version_exact_match"] = (
                    node_version.notna() & reach_version.notna() & node_version.eq(reach_version)
                )

            merged = _attach_reach_slope_fallback(
                merged,
                reach,
                tolerance=slope_fallback_tolerance,
            )

            # Optional Level-4 SoS layer is explicitly prefixed and keeps both
            # version fields. IDs are never declared equivalent silently.
            sos_file = sos_dir / f"{reach_id}.csv"
            if config["download"]["include_sos_v2"] and sos_file.exists():
                sos = pd.read_csv(sos_file, dtype={"reach_id": "string"}, low_memory=False)
                source_time = sos.pop("time_utc") if "time_utc" in sos else sos.get("time_str")
                sos["sos_v2_time_utc"] = pd.to_datetime(source_time, errors="coerce", utc=True)
                sos = sos.dropna(subset=["sos_v2_time_utc"])
                sos = sos.rename(columns={
                    column: f"sos_v2_{column}"
                    for column in sos.columns
                    if column not in {"reach_id", "sos_v2_time_utc"}
                })
                merged = _asof_attach(
                    merged, sos.drop(columns=["reach_id"], errors="ignore"),
                    left_time="time_utc", right_time="sos_v2_time_utc", tolerance=tolerance,
                )
                merged["sos_v2_match_delta_seconds"] = (
                    merged["time_utc"] - merged["sos_v2_time_utc"]
                ).abs().dt.total_seconds()
                merged["sos_v2_cross_version_id_assumption"] = merged["sos_v2_time_utc"].notna()

            atomic_write_csv(merged, output_dir / node_file.name)
            records.append({
                "node_id": node_file.stem, "reach_id": reach_id, "status": "ok",
                "node_rows": len(node), "output_rows": len(merged),
                "reach_matched_rows": int(merged["reach_temporal_match"].sum()),
                "max_match_delta_seconds": merged["reach_match_delta_seconds"].max(),
                "slope_available_rows": int(
                    pd.to_numeric(merged.get("reach_slope"), errors="coerce")
                    .gt(0).sum()
                ),
                "slope_fallback_rows": int(
                    merged.get(
                        "reach_slope_fallback_used",
                        pd.Series(False, index=merged.index),
                    ).fillna(False).sum()
                ),
                "max_slope_match_delta_seconds": pd.to_numeric(
                    merged.get("reach_slope_match_delta_seconds"), errors="coerce"
                ).max(),
                "node_sword_versions": "|".join(sorted(node.get("sword_version", pd.Series(dtype=str)).dropna().astype(str).unique())),
                "reach_sword_versions": "|".join(sorted(reach.get("reach_sword_version", pd.Series(dtype=str)).dropna().astype(str).unique())),
                "exact_sword_version_match_rows": int(merged.get("qc_node_reach_sword_version_exact_match", pd.Series(False, index=merged.index)).sum()),
            })
        except Exception as exc:
            records.append({
                "node_id": node_file.stem, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    audit = pd.DataFrame(records)
    atomic_write_csv(audit, output_path(config, "manifests", "node_reach_fusion.csv"))
    return audit


def _aggregate_overpasses(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["swot_date"] = pd.to_datetime(frame["time_utc"], errors="coerce", utc=True).dt.floor("D")
    cycle_column = "cycle_id" if "cycle_id" in frame else "reach_cycle_id"
    pass_column = "pass_id" if "pass_id" in frame else "reach_pass_id"
    group_columns = ["swot_date"]
    for column in [cycle_column, pass_column]:
        if column in frame and column not in group_columns:
            group_columns.append(column)

    identifier_columns = {
        "node_id", "reach_id", "cycle_id", "pass_id", "station_id",
        "station_name", "station_lon", "station_lat", "time_str", "time_utc",
        "swot_date", "reach_cycle_id", "reach_pass_id",
    }
    aggregation: dict[str, str] = {}
    for column in frame.columns:
        if column in group_columns:
            continue
        if column in identifier_columns or not pd.api.types.is_numeric_dtype(frame[column]):
            aggregation[column] = "first"
        else:
            aggregation[column] = "median"
    grouped = frame.groupby(group_columns, dropna=False).agg(aggregation).reset_index()
    counts = frame.groupby(group_columns, dropna=False).agg(
        node_record_count=("node_id", "size"),
        unique_node_count=("node_id", "nunique"),
    ).reset_index()
    return grouped.merge(counts, on=group_columns, how="left", validate="one_to_one")


def _filter_station_overpass_hampel(
    overpasses: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove station-level WSE/width spikes after robust node aggregation.

    Per-node temporal Hampel filtering is already applied in quality.py.  This
    second pass is necessary because an overpass-wide disturbance can affect
    all selected nodes and therefore survive the per-node screen.
    """
    frame = overpasses.copy()
    if frame.empty:
        return frame, {
            "overpasses_before_station_hampel": 0,
            "overpasses_station_hampel_flagged": 0,
            "overpasses_after_station_hampel": 0,
        }

    time_column = "time_utc" if "time_utc" in frame else "swot_date"
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    frame = frame.sort_values(time_column).reset_index(drop=True)
    pass_column = "pass_id" if "pass_id" in frame else "reach_pass_id" if "reach_pass_id" in frame else None
    groups = frame.groupby(pass_column, dropna=False).groups if pass_column else {"all": frame.index}

    flagged_any = pd.Series(False, index=frame.index)
    for column in ["wse", "width"]:
        flags = pd.Series(False, index=frame.index)
        if column in frame:
            for indices in groups.values():
                flags.loc[indices] = hampel_flags(
                    frame.loc[indices, column],
                    int(config["quality"]["hampel_window"]),
                    float(config["quality"]["hampel_sigma"]),
                )
        frame[f"qc_station_hampel_{column}"] = flags.fillna(False)
        flagged_any |= flags.fillna(False)

    frame["qc_station_hampel_any"] = flagged_any
    enabled = bool(config["fusion"].get("station_overpass_hampel", True))
    exclude = bool(config["quality"].get("exclude_hampel_outliers", True))
    frame["qc_station_hampel_keep"] = ~(flagged_any & enabled & exclude)
    audit = {
        "overpasses_before_station_hampel": len(frame),
        "overpasses_station_hampel_flagged": int(flagged_any.sum()),
        "overpasses_after_station_hampel": int(frame["qc_station_hampel_keep"].sum()),
    }
    return frame.loc[frame["qc_station_hampel_keep"]].copy(), audit


def _limit_station_nodes(station_map: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Defensively enforce the configured maximum even with a stale inventory."""
    limit_each_side = int(config["inventory"]["node_neighbors_each_side"])
    maximum_nodes = 1 + 2 * limit_each_side
    mapping = station_map.copy()
    if "neighbor_offset" in mapping:
        offsets = pd.to_numeric(mapping["neighbor_offset"], errors="coerce")
        mapping = mapping[offsets.abs().le(limit_each_side)].copy()
        mapping["_abs_offset"] = offsets.loc[mapping.index].abs()
        mapping = mapping.sort_values(["_abs_offset", "neighbor_offset"]).drop(columns="_abs_offset")
    elif "selection_role" in mapping:
        mapping["_nearest_first"] = mapping["selection_role"].ne("nearest")
        mapping = mapping.sort_values("_nearest_first").drop(columns="_nearest_first")
    return mapping.drop_duplicates("node_id").head(maximum_nodes).copy()


def _read_daily_gauge(path: Path, config: dict) -> pd.DataFrame:
    gauge = pd.read_csv(path, low_memory=False)
    date_column = config["columns"]["gauge_date"]
    if date_column in gauge:
        gauge["gauge_date"] = pd.to_datetime(gauge[date_column], errors="coerce", utc=True).dt.floor("D")
    elif {"year", "mon", "day"}.issubset(gauge.columns):
        gauge["gauge_date"] = pd.to_datetime(
            dict(year=gauge["year"], month=gauge["mon"], day=gauge["day"]),
            errors="coerce", utc=True,
        )
    else:
        raise ValueError(f"Gauge file has neither {date_column!r} nor year/mon/day")
    gauge = gauge.dropna(subset=["gauge_date"]).copy()
    for column in gauge.columns:
        if column != "gauge_date":
            converted = pd.to_numeric(gauge[column], errors="coerce")
            original_nonnull = gauge[column].notna()
            if converted[original_nonnull].notna().all():
                gauge[column] = converted
    numeric = [column for column in gauge.select_dtypes(include=[np.number]).columns if column != "gauge_date"]
    other = [column for column in gauge.columns if column not in numeric + ["gauge_date"]]
    method = config["fusion"]["gauge_daily_aggregation"]
    numeric_agg = "median" if method == "median" else "mean"
    agg = {**{column: numeric_agg for column in numeric}, **{column: "first" for column in other}}
    daily = gauge.groupby("gauge_date", as_index=False).agg(agg)
    counts = gauge.groupby("gauge_date").size().rename("gauge_daily_source_count").reset_index()
    daily = daily.merge(counts, on="gauge_date", how="left")
    return daily.rename(columns={column: f"gauge_{column}" for column in daily.columns if column != "gauge_date"})


def _match_gauge(overpasses: pd.DataFrame, gauge: pd.DataFrame, tolerance_days: int) -> pd.DataFrame:
    if tolerance_days == 0:
        merged = overpasses.merge(
            gauge, left_on="swot_date", right_on="gauge_date",
            how="left", validate="many_to_one",
        )
    else:
        merged = pd.merge_asof(
            overpasses.sort_values("swot_date"), gauge.sort_values("gauge_date"),
            left_on="swot_date", right_on="gauge_date", direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
        )
    merged["gauge_match_delta_days_signed"] = (
        merged["gauge_date"] - merged["swot_date"]
    ).dt.total_seconds() / 86400.0
    merged["gauge_match_delta_days"] = merged[
        "gauge_match_delta_days_signed"
    ].abs()
    delta = merged["gauge_match_delta_days"]
    merged["gauge_match_class"] = np.select(
        [
            delta.eq(0),
            delta.le(1),
            delta.le(3),
            delta.le(7),
            delta.notna(),
        ],
        ["exact_day", "within_1_day", "within_3_days", "within_7_days", "beyond_7_days"],
        default="unmatched",
    )
    merged["gauge_exact_day_match"] = delta.eq(0)
    return merged


def fuse_stations(config: dict) -> pd.DataFrame:
    map_path = output_path(config, "inventory", "station_feature_map.csv")
    if not map_path.exists():
        raise FileNotFoundError("station_feature_map.csv is required; inventory.scope cannot be 'all' for gauge fusion")
    mapping = pd.read_csv(map_path, dtype={"station_id": "string", "node_id": "string", "reach_id": "string"})
    mapping = mapping[mapping["within_max_distance"] & mapping["node_id"].notna()].copy()
    node_reach_dir = output_path(config, "fused", "node_reach")
    node_records_dir = output_path(config, "fused", "station_node_records")
    overpass_dir = output_path(config, "fused", "station_overpasses")
    node_records_dir.mkdir(parents=True, exist_ok=True)
    overpass_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    active_station_files: set[str] = set()

    for station_id, station_map in mapping.groupby("station_id", sort=True):
        station_map = _limit_station_nodes(station_map, config)
        frames: list[pd.DataFrame] = []
        for _, item in station_map.iterrows():
            path = node_reach_dir / f"{item['node_id']}.csv"
            if path.exists():
                frame = pd.read_csv(path, dtype={"node_id": "string", "reach_id": "string"}, low_memory=False)
                frame["time_utc"] = pd.to_datetime(frame.get("time_utc", frame.get("time_str")), errors="coerce", utc=True)
                frames.append(frame)
        if not frames:
            records.append({"station_id": station_id, "status": "no_node_reach_files"})
            continue
        combined = pd.concat(frames, ignore_index=True, copy=False)
        first = station_map.iloc[0]
        station_columns = [
            "station_id", "station_name", "station_lon", "station_lat"
        ]
        station_metadata = pd.DataFrame(
            {
                "station_id": station_id,
                "station_name": first.get("station_name", ""),
                "station_lon": first.get("station_lon", np.nan),
                "station_lat": first.get("station_lat", np.nan),
            },
            index=combined.index,
        )
        # The node/reach table is intentionally very wide.  Adding the four
        # metadata columns one at a time fragments its internal pandas blocks
        # and emits a PerformanceWarning for every station.  Concatenate them
        # in one operation and consolidate once before subsequent processing.
        combined = pd.concat(
            [
                combined.drop(columns=station_columns, errors="ignore"),
                station_metadata,
            ],
            axis=1,
            copy=False,
        ).copy()
        atomic_write_csv(combined, node_records_dir / f"{station_id}.csv")
        overpasses = _aggregate_overpasses(combined.dropna(subset=["time_utc"]))
        overpasses, station_hampel_audit = _filter_station_overpass_hampel(overpasses, config)
        gauge_path = Path(config["paths"]["gauge_dir"]) / f"{station_id}.csv"
        if gauge_path.exists():
            gauge = _read_daily_gauge(gauge_path, config)
            overpasses = _match_gauge(
                overpasses, gauge, int(config["fusion"]["gauge_match_tolerance_days"])
            )
            gauge_q = f"gauge_{config['columns']['gauge_discharge']}"
            gauge_valid = pd.to_numeric(overpasses.get(gauge_q), errors="coerce").notna()
            gauge_matches = int(gauge_valid.sum())
            gauge_exact_matches = int(
                (gauge_valid & overpasses["gauge_exact_day_match"]).sum()
            )
            gauge_offset_matches = gauge_matches - gauge_exact_matches
            status = "ok"
        else:
            gauge_matches = 0
            gauge_exact_matches = 0
            gauge_offset_matches = 0
            status = "missing_gauge_file"
        atomic_write_csv(overpasses, overpass_dir / f"{station_id}.csv")
        active_station_files.add(f"{station_id}.csv")
        records.append({
            "station_id": station_id, "status": status,
            "selected_nodes": station_map["node_id"].nunique(),
            "node_records": len(combined), "overpasses": len(overpasses),
            "gauge_q_matches": gauge_matches,
            "gauge_q_exact_day_matches": gauge_exact_matches,
            "gauge_q_offset_day_matches": gauge_offset_matches,
            "gauge_match_tolerance_days": int(
                config["fusion"]["gauge_match_tolerance_days"]
            ),
            **station_hampel_audit,
        })
    # A changed node-neighbour rule can make a station unavailable in the
    # current run.  Do not leave an older derived file behind, because the
    # experiment runner discovers inputs by directory listing.
    _remove_stale_csvs(node_records_dir, active_station_files)
    _remove_stale_csvs(overpass_dir, active_station_files)
    audit = pd.DataFrame(records)
    atomic_write_csv(audit, output_path(config, "manifests", "station_fusion.csv"))
    return audit


def robust_datum_shift(config: dict) -> pd.DataFrame:
    input_dir = output_path(config, "fused", "station_overpasses")
    output_dir = output_path(config, "fused", "station_overpasses_datum_adjusted")
    output_dir.mkdir(parents=True, exist_ok=True)
    wse_column = "wse"
    gauge_wse_column = f"gauge_{config['columns']['gauge_wse']}"
    minimum = int(config["datum_shift"]["minimum_pairs"])
    sigma = float(config["datum_shift"]["mad_sigma"])
    max_mad = float(config["datum_shift"]["maximum_residual_mad_m"])
    records: list[dict] = []
    active_output_files: set[str] = set()

    for path in sorted(input_dir.glob("*.csv")) if input_dir.exists() else []:
        frame = pd.read_csv(path, low_memory=False)
        if wse_column not in frame or gauge_wse_column not in frame:
            records.append({"station_id": path.stem, "status": "missing_wse_column"})
            atomic_write_csv(frame, output_dir / path.name)
            active_output_files.add(path.name)
            continue
        satellite = pd.to_numeric(frame[wse_column], errors="coerce")
        gauge = pd.to_numeric(frame[gauge_wse_column], errors="coerce")
        difference = gauge - satellite
        valid = difference.dropna()
        if len(valid) < minimum:
            shift = np.nan
            apply_shift = False
            status = "insufficient_pairs"
            inliers = valid
            residual_mad = np.nan
        else:
            initial = float(valid.median())
            absolute_deviation = (valid - initial).abs()
            mad = float(absolute_deviation.median())
            if mad > 0:
                inliers = valid[absolute_deviation <= sigma * 1.4826 * mad]
            else:
                inliers = valid
            shift = float(inliers.median())
            residual_mad = float((inliers - shift).abs().median())
            apply_shift = len(inliers) >= minimum and residual_mad <= max_mad
            status = "applied" if apply_shift else "unstable_not_applied"
        frame["wse_datum_shift_m"] = shift
        frame["wse_datum_adjusted"] = satellite + shift if apply_shift else satellite
        frame["wse_datum_shift_applied"] = apply_shift

        before = gauge - satellite
        after = gauge - frame["wse_datum_adjusted"]
        records.append({
            "station_id": path.stem, "status": status, "pairs": len(valid),
            "inlier_pairs": len(inliers), "shift_m": shift,
            "residual_mad_m": residual_mad,
            "bias_before_m": before.mean(), "mae_before_m": before.abs().mean(),
            "rmse_before_m": np.sqrt((before**2).mean()),
            "bias_after_m": after.mean(), "mae_after_m": after.abs().mean(),
            "rmse_after_m": np.sqrt((after**2).mean()),
        })
        atomic_write_csv(frame, output_dir / path.name)
        active_output_files.add(path.name)
    _remove_stale_csvs(output_dir, active_output_files)
    report = pd.DataFrame(records)
    atomic_write_csv(report, output_path(config, "manifests", "datum_shift.csv"))
    return report
