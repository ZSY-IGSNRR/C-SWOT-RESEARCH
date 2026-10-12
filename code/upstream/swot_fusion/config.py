from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "station_csv": 'inputs/Station_info.csv',
        "node_index_csv": 'inputs/as_sword_nodes_v17b.csv',
        "gauge_dir": 'inputs/merged_river',
        "roughness_nc": 'inputs/Merged_N_River.nc',
        "output_root": 'inputs/outputs',
    },
    "columns": {
        "station_id": "ID",
        "station_name": "Name",
        "station_lon": "Lon_web",
        "station_lat": "Lat_web",
        "node_id": "node_id",
        "reach_id": "reach_id",
        "node_lon": "x",
        "node_lat": "y",
        "gauge_date": "tm",
        "gauge_wse": "Z",
        "gauge_discharge": "Q",
    },
    "inventory": {
        "scope": "station_neighborhood",
        # Reproduce the original VC scripts: nearest node in lon/lat degree
        # space, then node_id +10 and -10 (when present), with no distance
        # exclusion.  "same_reach" remains available as the safer alternative.
        "station_matching_mode": "vc_legacy",
        # Nearest node plus at most one same-reach neighbour upstream and
        # downstream.  Keeping this at 1 caps every station at three nodes.
        "node_neighbors_each_side": 1,
        "max_station_node_distance_km": None,
        "chunksize": 250000,
    },
    "download": {
        "start_time": "2023-03-01T00:00:00Z",
        "end_time": "2026-09-03T23:59:59Z",
        "max_workers": 8,
        "timeout_seconds": 90,
        "retries": 4,
        "retry_backoff_seconds": 2.0,
        "overwrite": False,
        "include_sos_v2": True,
        "api_key_environment_variable": "HYDROCRON_API_KEY",
    },
    "quality": {
        "accepted_node_q": [0, 1],
        "accepted_reach_q": [0, 1],
        "accepted_discharge_q": [0, 1],
        "exclude_hampel_outliers": True,
        "hampel_window": 3,
        "hampel_sigma": 3.0,
    },
    "fusion": {
        # Keep official reach variables (including SWOT discharge products)
        # tied to the same overpass.  Only slope may use the wider fallback
        # below because slope is a slowly varying reach attribute and is
        # subsequently screened/repaired by the hydraulic model.
        "reach_node_time_tolerance_minutes": 30,
        # None reproduces 5.1.FindSlope.py: nearest reach slope with no
        # maximum time separation.  Source time/delta remain explicit.
        "reach_node_slope_fallback_tolerance_days": None,
        # Reproduce 6.1.R-N-S.py: exact calendar-day gauge match.
        "gauge_match_tolerance_days": 0,
        "gauge_daily_aggregation": "median",
        # The raw per-node Hampel screen runs in quality.py.  This second
        # screen catches overpass-wide WSE/width spikes after node aggregation.
        "station_overpass_hampel": True,
    },
    "roughness": {
        "max_distance_km": 15.0,
        "allow_global_mean_fallback": False,
    },
    "datum_shift": {
        "minimum_pairs": 5,
        "mad_sigma": 3.5,
        "maximum_residual_mad_m": 5.0,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration, filling omitted keys from safe defaults."""
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as stream:
            user = json.load(stream)
        _deep_update(config, user)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["inventory"]["scope"] not in {"station_neighborhood", "all"}:
        raise ValueError("inventory.scope must be 'station_neighborhood' or 'all'")
    if config["download"]["max_workers"] < 1:
        raise ValueError("download.max_workers must be positive")
    neighbor_count = int(config["inventory"]["node_neighbors_each_side"])
    if neighbor_count not in {0, 1}:
        raise ValueError(
            "inventory.node_neighbors_each_side must be 0 or 1 "
            "(nearest node plus at most two neighbours)"
        )
    station_matching_mode = config["inventory"].get(
        "station_matching_mode", "same_reach"
    )
    if station_matching_mode not in {"same_reach", "vc_legacy"}:
        raise ValueError(
            "inventory.station_matching_mode must be 'same_reach' or 'vc_legacy'"
        )
    max_distance = config["inventory"].get("max_station_node_distance_km")
    if max_distance is not None and float(max_distance) <= 0:
        raise ValueError(
            "inventory.max_station_node_distance_km must be positive or null"
        )
    if config["fusion"]["reach_node_time_tolerance_minutes"] <= 0:
        raise ValueError("fusion.reach_node_time_tolerance_minutes must be positive")
    slope_fallback_days = config["fusion"].get(
        "reach_node_slope_fallback_tolerance_days", 0
    )
    if slope_fallback_days is not None and float(slope_fallback_days) < 0:
        raise ValueError(
            "fusion.reach_node_slope_fallback_tolerance_days cannot be negative"
        )
    gauge_tolerance_days = int(config["fusion"].get("gauge_match_tolerance_days", 0))
    if gauge_tolerance_days < 0:
        raise ValueError("fusion.gauge_match_tolerance_days cannot be negative")
    start = config["download"]["start_time"]
    end = config["download"]["end_time"]
    if start >= end:
        raise ValueError("download.start_time must precede download.end_time")


def output_path(config: dict[str, Any], *parts: str) -> Path:
    return Path(config["paths"]["output_root"]).joinpath(*parts)
