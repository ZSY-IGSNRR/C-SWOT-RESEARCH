#!/usr/bin/env python3
"""Geometry-based audit of SWORD v16 and v17b reaches used in this study.

The current local model uses a v17b node inventory.  Hydrocron exposes SoS V3
discharge through the Version-2.0/v16 reach collection.  This script verifies
whether the same numeric reach ID still describes the same physical centerline,
and searches nearby v16 reaches when it does not.

No discharge or gauge value is used to choose a mapping.  Mapping is based only
on SWORD coordinates, centerline proximity, reach length and ID provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EARTH_RADIUS_KM = 6371.0088


def normalize_id(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _filled(variable, dtype) -> np.ndarray:
    values = np.ma.asarray(variable[:], dtype=dtype)
    if np.ma.isMaskedArray(values):
        fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else -999
        values = values.filled(fill)
    return np.asarray(values, dtype=dtype)


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.deg2rad(np.asarray(lon, dtype=float))
    lat_r = np.deg2rad(np.asarray(lat, dtype=float))
    cos_lat = np.cos(lat_r)
    return np.column_stack((
        cos_lat * np.cos(lon_r),
        cos_lat * np.sin(lon_r),
        np.sin(lat_r),
    ))


def chord_to_km(chord: np.ndarray | float) -> np.ndarray:
    chord = np.asarray(chord, dtype=float)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))


def read_reaches(path: Path, version: str) -> pd.DataFrame:
    with netCDF4.Dataset(path, "r") as root:
        group = root.groups["reaches"]
        reach_count = len(group.dimensions["num_reaches"])

        def optional(name: str) -> np.ndarray:
            if name not in group.variables:
                return np.full(reach_count, np.nan)
            return _filled(group.variables[name], float)

        frame = pd.DataFrame({
            "reach_id": _filled(group.variables["reach_id"], np.int64),
            "lon": _filled(group.variables["x"], float),
            "lat": _filled(group.variables["y"], float),
            "reach_length_m": _filled(group.variables["reach_length"], float),
            "n_nodes": _filled(group.variables["n_nodes"], float),
            "network": optional("network"),
            "stream_order": optional("stream_order"),
            "trib_flag": _filled(group.variables["trib_flag"], float),
        })
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame[
        frame["lon"].between(-180, 180)
        & frame["lat"].between(-90, 90)
        & frame["reach_id"].gt(0)
    ].drop_duplicates("reach_id")
    frame["sword_version"] = version
    return frame.reset_index(drop=True)


def extract_centerline_points(
    path: Path,
    reach_ids: set[int],
    *,
    chunk_size: int = 1_000_000,
) -> dict[int, np.ndarray]:
    wanted = np.asarray(sorted(reach_ids), dtype=np.int64)
    pieces: dict[int, list[np.ndarray]] = {int(value): [] for value in wanted}
    if not len(wanted):
        return {}
    with netCDF4.Dataset(path, "r") as root:
        group = root.groups["centerlines"]
        reach_var = group.variables["reach_id"]
        x_var = group.variables["x"]
        y_var = group.variables["y"]
        # ``centerlines/reach_id`` is shaped (num_domains, num_points).
        # Domain 0 is the primary reach owning each centerline point; the
        # remaining rows only list neighbouring domains at junctions.  Using
        # every row would duplicate junction points and produce a 2-D mask.
        size = len(x_var)
        for start in range(0, size, chunk_size):
            stop = min(size, start + chunk_size)
            # netCDF4 slices are arrays rather than Variables; handle masks here.
            reach = reach_var[0, start:stop]
            if np.ma.isMaskedArray(reach):
                reach = reach.filled(-999)
            reach = np.asarray(reach, dtype=np.int64)
            mask = np.isin(reach, wanted)
            if not mask.any():
                continue
            x = x_var[start:stop]
            y = y_var[start:stop]
            if np.ma.isMaskedArray(x):
                x = x.filled(np.nan)
            if np.ma.isMaskedArray(y):
                y = y.filled(np.nan)
            x = np.asarray(x, dtype=float)[mask]
            y = np.asarray(y, dtype=float)[mask]
            selected_reach = reach[mask]
            finite = np.isfinite(x) & np.isfinite(y)
            x, y, selected_reach = x[finite], y[finite], selected_reach[finite]
            for reach_id in np.unique(selected_reach):
                local = selected_reach == reach_id
                pieces[int(reach_id)].append(np.column_stack((x[local], y[local])))
    return {
        reach_id: np.concatenate(parts, axis=0)
        for reach_id, parts in pieces.items()
        if parts
    }


def centerline_metrics(a: np.ndarray | None, b: np.ndarray | None) -> dict[str, float]:
    empty = {
        "centerline_points_v17b": 0,
        "centerline_points_v16": 0,
        "median_bidirectional_km": np.nan,
        "p95_bidirectional_km": np.nan,
        "overlap_fraction_1km": np.nan,
        "overlap_fraction_2km": np.nan,
        "overlap_fraction_5km": np.nan,
    }
    if a is None or b is None or not len(a) or not len(b):
        return empty
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    if not len(a) or not len(b):
        return empty
    a_xyz = lonlat_to_xyz(a[:, 0], a[:, 1])
    b_xyz = lonlat_to_xyz(b[:, 0], b[:, 1])
    d_ab = chord_to_km(cKDTree(b_xyz).query(a_xyz, k=1)[0])
    d_ba = chord_to_km(cKDTree(a_xyz).query(b_xyz, k=1)[0])
    return {
        "centerline_points_v17b": int(len(a)),
        "centerline_points_v16": int(len(b)),
        "median_bidirectional_km": float(max(np.median(d_ab), np.median(d_ba))),
        "p95_bidirectional_km": float(max(np.quantile(d_ab, .95), np.quantile(d_ba, .95))),
        "overlap_fraction_1km": float(min(np.mean(d_ab <= 1), np.mean(d_ba <= 1))),
        "overlap_fraction_2km": float(min(np.mean(d_ab <= 2), np.mean(d_ba <= 2))),
        "overlap_fraction_5km": float(min(np.mean(d_ab <= 5), np.mean(d_ba <= 5))),
    }


def mapping_level(row: pd.Series) -> str:
    length_ratio = row["length_similarity"]
    if (
        row["overlap_fraction_1km"] >= .80
        and row["p95_bidirectional_km"] <= 2
        and length_ratio >= .50
    ):
        return "high"
    if (
        row["overlap_fraction_2km"] >= .70
        and row["p95_bidirectional_km"] <= 5
        and length_ratio >= .33
    ):
        return "medium"
    return "low"


def build_crosswalk(
    v16_path: Path,
    v17b_path: Path,
    target_ids: set[int],
    *,
    candidate_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v16 = read_reaches(v16_path, "v16").set_index("reach_id", drop=False)
    v17b = read_reaches(v17b_path, "v17b").set_index("reach_id", drop=False)
    target = v17b.loc[v17b.index.intersection(target_ids)].copy()
    missing_target = sorted(target_ids - set(target.index.astype(int)))

    tree = cKDTree(lonlat_to_xyz(v16["lon"].to_numpy(), v16["lat"].to_numpy()))
    k = min(candidate_count, len(v16))
    _, candidate_index = tree.query(
        lonlat_to_xyz(target["lon"].to_numpy(), target["lat"].to_numpy()), k=k
    )
    if k == 1:
        candidate_index = candidate_index[:, None]

    candidate_map: dict[int, set[int]] = {}
    v16_ids_array = v16["reach_id"].to_numpy(dtype=np.int64)
    for target_id, indexes in zip(target["reach_id"].astype(int), candidate_index):
        candidates = set(v16_ids_array[np.atleast_1d(indexes)].astype(int))
        if target_id in v16.index:
            candidates.add(int(target_id))
        candidate_map[int(target_id)] = candidates

    v17_points = extract_centerline_points(v17b_path, set(candidate_map))
    v16_candidate_ids = set().union(*candidate_map.values()) if candidate_map else set()
    v16_points = extract_centerline_points(v16_path, v16_candidate_ids)

    rows: list[dict] = []
    for target_id, candidates in candidate_map.items():
        a = v17b.loc[target_id]
        a_xyz = lonlat_to_xyz(np.asarray([a["lon"]]), np.asarray([a["lat"]]))[0]
        for candidate_id in candidates:
            b = v16.loc[candidate_id]
            b_xyz = lonlat_to_xyz(np.asarray([b["lon"]]), np.asarray([b["lat"]]))[0]
            representative_distance = float(chord_to_km(np.linalg.norm(a_xyz - b_xyz)))
            length_a = float(a["reach_length_m"])
            length_b = float(b["reach_length_m"])
            length_similarity = (
                min(length_a, length_b) / max(length_a, length_b)
                if np.isfinite(length_a) and np.isfinite(length_b) and min(length_a, length_b) > 0
                else np.nan
            )
            row = {
                "v17b_reach_id": str(target_id),
                "v16_reach_id": str(candidate_id),
                "same_numeric_id": target_id == candidate_id,
                "representative_distance_km": representative_distance,
                "v17b_lon": float(a["lon"]),
                "v17b_lat": float(a["lat"]),
                "v16_lon": float(b["lon"]),
                "v16_lat": float(b["lat"]),
                "v17b_length_m": length_a,
                "v16_length_m": length_b,
                "length_similarity": length_similarity,
                "v17b_network": a["network"],
                "v16_network": b["network"],
            }
            row.update(centerline_metrics(v17_points.get(target_id), v16_points.get(candidate_id)))
            rows.append(row)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates, pd.DataFrame()
    candidates["mapping_level"] = candidates.apply(mapping_level, axis=1)
    level_rank = {"high": 2, "medium": 1, "low": 0}
    candidates["_level_rank"] = candidates["mapping_level"].map(level_rank)
    candidates["_same_rank"] = candidates["same_numeric_id"].astype(int)
    candidates["_overlap"] = candidates["overlap_fraction_1km"].fillna(-1)
    candidates["_p95"] = candidates["p95_bidirectional_km"].fillna(np.inf)

    selected: list[dict] = []
    for reach_id, group in candidates.groupby("v17b_reach_id", sort=False):
        same = group[group["same_numeric_id"]]
        same_good = same[same["mapping_level"].isin(["high", "medium"])]
        if not same_good.empty:
            chosen = same_good.sort_values(
                ["_level_rank", "_overlap", "_p95"],
                ascending=[False, False, True],
            ).iloc[0]
        else:
            chosen = group.sort_values(
                ["_level_rank", "_overlap", "_p95", "representative_distance_km"],
                ascending=[False, False, True, True],
            ).iloc[0]

        good = group[group["mapping_level"].isin(["high", "medium"])].sort_values(
            ["_level_rank", "_overlap", "_p95"], ascending=[False, False, True]
        )
        ambiguous_multiple = False
        if not bool(chosen["same_numeric_id"]) and len(good) >= 2:
            top = good.iloc[0]
            second = good.iloc[1]
            ambiguous_multiple = (
                top["mapping_level"] == second["mapping_level"]
                and abs(top["overlap_fraction_1km"] - second["overlap_fraction_1km"]) < .10
            )

        if ambiguous_multiple:
            status = "ambiguous_multiple"
            confidence = "low"
        elif chosen["mapping_level"] == "low":
            status = "ambiguous_or_unmatched"
            confidence = "low"
        elif bool(chosen["same_numeric_id"]):
            status = f"stable_same_id_{chosen['mapping_level']}"
            confidence = chosen["mapping_level"]
        else:
            status = f"remapped_{chosen['mapping_level']}"
            confidence = chosen["mapping_level"]

        record = chosen.drop(labels=[c for c in chosen.index if c.startswith("_")]).to_dict()
        record.update({
            "mapping_status": status,
            "mapping_confidence": confidence,
            "candidate_count_tested": int(len(group)),
            "good_candidate_count": int(len(good)),
            "usable_with_current_same_id_sos": status.startswith("stable_same_id_"),
            "requires_different_v16_sos_download": status.startswith("remapped_"),
        })
        selected.append(record)

    for target_id in missing_target:
        selected.append({
            "v17b_reach_id": str(target_id),
            "v16_reach_id": "",
            "same_numeric_id": False,
            "mapping_status": "missing_from_v17b_netcdf",
            "mapping_confidence": "low",
            "candidate_count_tested": 0,
            "good_candidate_count": 0,
            "usable_with_current_same_id_sos": False,
            "requires_different_v16_sos_download": False,
        })

    crosswalk = pd.DataFrame(selected)
    candidates.drop(columns=["_level_rank", "_same_rank", "_overlap", "_p95"]).to_csv(
        Path("reach_mapping_candidates.csv"), index=False
    ) if False else None
    return crosswalk, candidates.drop(
        columns=["_level_rank", "_same_rank", "_overlap", "_p95"]
    )


def distance_bin(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    for upper, label in [
        (1, "<=1 km"), (2, "1-2 km"), (5, "2-5 km"),
        (10, "5-10 km"), (20, "10-20 km"), (50, "20-50 km"),
    ]:
        if value <= upper:
            return label
    return ">50 km"


def build_station_audit(
    station_map_path: Path,
    crosswalk: pd.DataFrame,
    station_stats_path: Path | None,
    common_support_path: Path | None,
) -> pd.DataFrame:
    mapping = pd.read_csv(
        station_map_path,
        dtype={"station_id": "string", "node_id": "string", "reach_id": "string"},
        low_memory=False,
    )
    mapping["station_id"] = normalize_id(mapping["station_id"])
    mapping["reach_id"] = normalize_id(mapping["reach_id"])
    nearest = mapping[mapping["selection_role"].eq("nearest")].copy()
    if nearest.empty:
        nearest = mapping.sort_values("nearest_node_distance_km").drop_duplicates("station_id")
    counts = mapping.groupby("station_id").agg(
        selected_node_count=("node_id", "nunique"),
        selected_reach_count=("reach_id", "nunique"),
    ).reset_index()
    station = nearest[[
        "station_id", "station_name", "station_lon", "station_lat",
        "nearest_node_distance_km", "node_id", "reach_id",
    ]].drop_duplicates("station_id")
    station = station.merge(counts, on="station_id", how="left", validate="one_to_one")
    station = station.rename(columns={
        "node_id": "nearest_v17b_node_id",
        "reach_id": "v17b_reach_id",
    })
    station["v17b_reach_id"] = normalize_id(station["v17b_reach_id"])
    crosswalk = crosswalk.copy()
    crosswalk["v17b_reach_id"] = normalize_id(crosswalk["v17b_reach_id"])
    station = station.merge(crosswalk, on="v17b_reach_id", how="left", validate="many_to_one")
    station["nearest_node_distance_km"] = pd.to_numeric(
        station["nearest_node_distance_km"], errors="coerce"
    )
    station["station_node_distance_bin"] = station["nearest_node_distance_km"].map(distance_bin)

    if station_stats_path and station_stats_path.exists():
        stats = pd.read_csv(station_stats_path, dtype={"station_id": "string"}, low_memory=False)
        stats["station_id"] = normalize_id(stats["station_id"])
        eligible = stats[["station_id", "eligible_for_calibration"]].drop_duplicates("station_id")
        station = station.merge(eligible, on="station_id", how="left", validate="one_to_one")
    if common_support_path and common_support_path.exists():
        common = pd.read_csv(common_support_path, usecols=["station_id"], dtype={"station_id": "string"})
        common["station_id"] = normalize_id(common["station_id"])
        common_ids = set(common["station_id"].dropna())
        station["in_current_sos_common_support"] = station["station_id"].isin(common_ids)
    return station


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16", type=Path, required=True)
    parser.add_argument("--v17b", type=Path, required=True)
    parser.add_argument("--station-map", type=Path, required=True)
    parser.add_argument("--station-stats", type=Path)
    parser.add_argument("--common-support", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    station_map = pd.read_csv(
        args.station_map, usecols=["reach_id"], dtype={"reach_id": "string"}
    )
    target_ids = set(
        pd.to_numeric(normalize_id(station_map["reach_id"]), errors="coerce")
        .dropna().astype(np.int64)
    )
    crosswalk, candidates = build_crosswalk(
        args.v16, args.v17b, target_ids, candidate_count=args.candidate_count
    )
    crosswalk_path = args.output_dir / "sword_reach_crosswalk_v16_v17b.csv"
    candidate_path = args.output_dir / "sword_reach_mapping_candidates.csv"
    crosswalk.to_csv(crosswalk_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    station = build_station_audit(
        args.station_map, crosswalk, args.station_stats, args.common_support
    )
    station_path = args.output_dir / "station_spatial_audit.csv"
    station.to_csv(station_path, index=False, encoding="utf-8-sig")

    reach_status = (
        crosswalk.groupby(["mapping_status", "mapping_confidence"], dropna=False)
        .size().rename("reach_count").reset_index()
    )
    station_status = (
        station.groupby(["mapping_status", "mapping_confidence"], dropna=False)
        .size().rename("station_count").reset_index()
    )
    distance_summary = []
    for population, mask in {
        "all_mapped_stations": pd.Series(True, index=station.index),
        "calibratable_stations": station.get("eligible_for_calibration", pd.Series(False, index=station.index)).astype(str).str.lower().eq("true"),
        "current_sos_common_support": station.get("in_current_sos_common_support", pd.Series(False, index=station.index)).fillna(False).astype(bool),
    }.items():
        values = station.loc[mask, "nearest_node_distance_km"].dropna()
        distance_summary.append({
            "population": population,
            "station_count": int(mask.sum()),
            "median_km": float(values.median()) if len(values) else np.nan,
            "p90_km": float(values.quantile(.90)) if len(values) else np.nan,
            "p95_km": float(values.quantile(.95)) if len(values) else np.nan,
            "max_km": float(values.max()) if len(values) else np.nan,
            "within_1km": int((values <= 1).sum()),
            "within_2km": int((values <= 2).sum()),
            "within_5km": int((values <= 5).sum()),
        })
    distance_summary = pd.DataFrame(distance_summary)
    reach_status.to_csv(args.output_dir / "reach_mapping_status_summary.csv", index=False)
    station_status.to_csv(args.output_dir / "station_mapping_status_summary.csv", index=False)
    distance_summary.to_csv(args.output_dir / "station_node_distance_summary.csv", index=False)

    audit = {
        "v16_netcdf": str(args.v16),
        "v17b_netcdf": str(args.v17b),
        "target_v17b_reach_count": int(len(target_ids)),
        "audited_reach_count": int(len(crosswalk)),
        "mapping_status_counts": crosswalk["mapping_status"].value_counts(dropna=False).to_dict(),
        "station_count": int(len(station)),
        "current_same_id_geometry_verified_station_count": int(
            station["usable_with_current_same_id_sos"].fillna(False).astype(bool).sum()
        ),
        "thresholds": {
            "high": "min bidirectional point overlap within 1 km >= 0.80; p95 <= 2 km; length similarity >= 0.50",
            "medium": "min bidirectional point overlap within 2 km >= 0.70; p95 <= 5 km; length similarity >= 0.33",
        },
        "note": "Discharge and gauge observations are not used in the spatial crosswalk.",
    }
    (args.output_dir / "sword_spatial_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("\nReach mapping status:\n", reach_status.to_string(index=False))
    print("\nStation-node distance summary:\n", distance_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
