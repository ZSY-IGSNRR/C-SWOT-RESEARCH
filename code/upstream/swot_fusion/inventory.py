from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .common import (
    atomic_write_csv,
    chord_to_km,
    lonlat_to_unit_xyz,
    normalize_id_series,
    require_columns,
)
from .config import output_path


def read_unique_nodes(config: dict) -> pd.DataFrame:
    """Read only the four needed columns from a potentially very large SWORD CSV."""
    columns = config["columns"]
    wanted = [
        columns["node_id"], columns["reach_id"],
        columns["node_lon"], columns["node_lat"],
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        config["paths"]["node_index_csv"],
        usecols=wanted,
        chunksize=int(config["inventory"]["chunksize"]),
        low_memory=False,
    ):
        chunk = chunk.drop_duplicates(subset=[columns["node_id"]])
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=wanted)
    nodes = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=[columns["node_id"]])
    nodes[columns["node_id"]] = normalize_id_series(nodes[columns["node_id"]])
    nodes[columns["reach_id"]] = normalize_id_series(nodes[columns["reach_id"]])
    for column in [columns["node_lon"], columns["node_lat"]]:
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce")
    nodes = nodes.dropna(subset=wanted)
    nodes = nodes[
        nodes[columns["node_lon"]].between(-180, 180)
        & nodes[columns["node_lat"]].between(-90, 90)
    ].copy()
    return nodes.reset_index(drop=True)


def _station_neighborhood_map(config: dict, nodes: pd.DataFrame) -> pd.DataFrame:
    columns = config["columns"]
    stations = pd.read_csv(config["paths"]["station_csv"], low_memory=False)
    required = [
        columns["station_id"], columns["station_lon"], columns["station_lat"],
    ]
    require_columns(stations, required, config["paths"]["station_csv"])
    stations[columns["station_id"]] = normalize_id_series(stations[columns["station_id"]])
    for column in [columns["station_lon"], columns["station_lat"]]:
        stations[column] = pd.to_numeric(stations[column], errors="coerce")
    stations = stations.dropna(subset=required).copy()

    matching_mode = config["inventory"].get("station_matching_mode", "same_reach")
    if matching_mode == "vc_legacy":
        # Exact spatial rule used by 6.1.R-N-S.py: Euclidean distance in
        # longitude/latitude degrees.  cKDTree only accelerates the identical
        # nearest-neighbour calculation.
        tree = cKDTree(nodes[[columns["node_lon"], columns["node_lat"]]].to_numpy(dtype=float))
        _, nearest_index = tree.query(
            stations[[columns["station_lon"], columns["station_lat"]]].to_numpy(dtype=float),
            k=1,
        )
        station_xyz = lonlat_to_unit_xyz(
            stations[columns["station_lon"]], stations[columns["station_lat"]]
        )
        nearest_xyz = lonlat_to_unit_xyz(
            nodes.iloc[nearest_index][columns["node_lon"]],
            nodes.iloc[nearest_index][columns["node_lat"]],
        )
        chord = np.linalg.norm(station_xyz - nearest_xyz, axis=1)
    else:
        tree = cKDTree(
            lonlat_to_unit_xyz(nodes[columns["node_lon"]], nodes[columns["node_lat"]])
        )
        chord, nearest_index = tree.query(
            lonlat_to_unit_xyz(stations[columns["station_lon"]], stations[columns["station_lat"]]),
            k=1,
        )
    stations["nearest_node_row"] = nearest_index
    stations["nearest_node_distance_km"] = chord_to_km(chord)
    max_distance = config["inventory"].get("max_station_node_distance_km")
    stations["within_max_distance"] = (
        True
        if max_distance is None
        else stations["nearest_node_distance_km"] <= float(max_distance)
    )

    node_id = columns["node_id"]
    reach_id = columns["reach_id"]
    neighbor_count = int(config["inventory"]["node_neighbors_each_side"])
    reach_positions: dict[str, tuple[pd.DataFrame, dict[str, int]]] = {}
    for rid, group in nodes.groupby(reach_id, sort=False):
        ordered = group.copy()
        ordered["_order"] = pd.to_numeric(ordered[node_id], errors="coerce")
        ordered = ordered.sort_values(["_order", node_id]).drop(columns="_order").reset_index(drop=True)
        reach_positions[str(rid)] = (ordered, {str(v): i for i, v in enumerate(ordered[node_id])})
    node_lookup = {
        str(value): index for index, value in enumerate(nodes[node_id].astype(str))
    }

    records: list[dict] = []
    for _, station in stations.iterrows():
        nearest = nodes.iloc[int(station["nearest_node_row"])]
        if not bool(station["within_max_distance"]):
            records.append({
                "station_id": station[columns["station_id"]],
                "station_name": station.get(columns["station_name"], ""),
                "station_lon": station[columns["station_lon"]],
                "station_lat": station[columns["station_lat"]],
                "nearest_node_distance_km": station["nearest_node_distance_km"],
                "within_max_distance": False,
                "node_id": "",
                "reach_id": "",
                "selection_role": "unmatched_distance",
            })
            continue
        rid = str(nearest[reach_id])
        nid = str(nearest[node_id])
        if matching_mode == "vc_legacy":
            try:
                center_id = int(nid)
                candidates = [
                    (str(center_id), 0, "nearest"),
                    (str(center_id + 10), 1, "legacy_node_id_neighbor"),
                    (str(center_id - 10), -1, "legacy_node_id_neighbor"),
                ]
            except ValueError:
                candidates = [(nid, 0, "nearest")]
            for candidate_id, offset, role in candidates:
                row_index = node_lookup.get(candidate_id)
                if row_index is None:
                    continue
                candidate = nodes.iloc[row_index]
                records.append({
                    "station_id": station[columns["station_id"]],
                    "station_name": station.get(columns["station_name"], ""),
                    "station_lon": station[columns["station_lon"]],
                    "station_lat": station[columns["station_lat"]],
                    "nearest_node_distance_km": station["nearest_node_distance_km"],
                    "within_max_distance": True,
                    "node_id": str(candidate[node_id]),
                    "reach_id": str(candidate[reach_id]),
                    "node_lon": candidate[columns["node_lon"]],
                    "node_lat": candidate[columns["node_lat"]],
                    "selection_role": role,
                    "neighbor_offset": offset,
                })
            continue
        ordered, positions = reach_positions[rid]
        center = positions[nid]
        lo = max(0, center - neighbor_count)
        hi = min(len(ordered), center + neighbor_count + 1)
        for position in range(lo, hi):
            candidate = ordered.iloc[position]
            records.append({
                "station_id": station[columns["station_id"]],
                "station_name": station.get(columns["station_name"], ""),
                "station_lon": station[columns["station_lon"]],
                "station_lat": station[columns["station_lat"]],
                "nearest_node_distance_km": station["nearest_node_distance_km"],
                "within_max_distance": True,
                "node_id": str(candidate[node_id]),
                "reach_id": str(candidate[reach_id]),
                "node_lon": candidate[columns["node_lon"]],
                "node_lat": candidate[columns["node_lat"]],
                "selection_role": "nearest" if position == center else "same_reach_neighbor",
                "neighbor_offset": position - center,
            })
    return pd.DataFrame.from_records(records)


def build_inventory(config: dict) -> dict[str, Path]:
    nodes = read_unique_nodes(config)
    root = output_path(config, "inventory")
    root.mkdir(parents=True, exist_ok=True)
    all_nodes_path = root / "all_unique_nodes.csv"
    atomic_write_csv(nodes.rename(columns={
        config["columns"]["node_id"]: "node_id",
        config["columns"]["reach_id"]: "reach_id",
        config["columns"]["node_lon"]: "node_lon",
        config["columns"]["node_lat"]: "node_lat",
    }), all_nodes_path)

    if config["inventory"]["scope"] == "all":
        mapped = pd.DataFrame()
        selected = pd.read_csv(all_nodes_path, dtype={"node_id": "string", "reach_id": "string"})
    else:
        mapped = _station_neighborhood_map(config, nodes)
        atomic_write_csv(mapped, root / "station_feature_map.csv")
        selected = mapped[mapped["within_max_distance"]].copy()

    node_ids = pd.DataFrame({"node_id": sorted(selected["node_id"].dropna().astype(str).unique())})
    reach_ids = pd.DataFrame({"reach_id": sorted(selected["reach_id"].dropna().astype(str).unique())})
    node_ids = node_ids[node_ids["node_id"] != ""]
    reach_ids = reach_ids[reach_ids["reach_id"] != ""]
    atomic_write_csv(node_ids, root / "node_ids.csv")
    atomic_write_csv(reach_ids, root / "reach_ids.csv")

    summary = pd.DataFrame([{
        "unique_nodes_in_index": len(nodes),
        "stations": mapped["station_id"].nunique() if not mapped.empty else np.nan,
        "matched_stations": mapped.loc[mapped["within_max_distance"], "station_id"].nunique() if not mapped.empty else np.nan,
        "download_node_ids": len(node_ids),
        "download_reach_ids": len(reach_ids),
        "scope": config["inventory"]["scope"],
    }])
    atomic_write_csv(summary, root / "inventory_summary.csv")
    return {
        "all_nodes": all_nodes_path,
        "station_map": root / "station_feature_map.csv",
        "node_ids": root / "node_ids.csv",
        "reach_ids": root / "reach_ids.csv",
        "summary": root / "inventory_summary.csv",
    }
