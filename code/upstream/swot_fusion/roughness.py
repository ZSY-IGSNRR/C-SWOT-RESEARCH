from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .common import atomic_write_csv, chord_to_km, lonlat_to_unit_xyz, normalize_id_series
from .config import output_path


def assign_roughness(config: dict) -> pd.DataFrame:
    """Assign nearest river roughness geodesically and expose all fallbacks."""
    columns = config["columns"]
    stations = pd.read_csv(config["paths"]["station_csv"], low_memory=False)
    stations[columns["station_id"]] = normalize_id_series(stations[columns["station_id"]])
    lon = pd.to_numeric(stations[columns["station_lon"]], errors="coerce")
    lat = pd.to_numeric(stations[columns["station_lat"]], errors="coerce")

    with xr.open_dataset(config["paths"]["roughness_nc"]) as dataset:
        roughness_lon = np.asarray(dataset["lon"].values, dtype=float).ravel()
        roughness_lat = np.asarray(dataset["lat"].values, dtype=float).ravel()
        roughness_n = np.asarray(dataset["N_River"].values, dtype=float).ravel()
    valid_source = (
        np.isfinite(roughness_lon) & np.isfinite(roughness_lat)
        & np.isfinite(roughness_n) & (roughness_n > 0)
    )
    roughness_lon = roughness_lon[valid_source]
    roughness_lat = roughness_lat[valid_source]
    roughness_n = roughness_n[valid_source]
    tree = cKDTree(lonlat_to_unit_xyz(roughness_lon, roughness_lat))

    valid_station = lon.notna() & lat.notna()
    distance = np.full(len(stations), np.nan)
    nearest_n = np.full(len(stations), np.nan)
    nearest_lon = np.full(len(stations), np.nan)
    nearest_lat = np.full(len(stations), np.nan)
    chord, index = tree.query(lonlat_to_unit_xyz(lon[valid_station], lat[valid_station]), k=1)
    rows = np.flatnonzero(valid_station.to_numpy())
    distance[rows] = chord_to_km(chord)
    nearest_n[rows] = roughness_n[index]
    nearest_lon[rows] = roughness_lon[index]
    nearest_lat[rows] = roughness_lat[index]

    max_distance = float(config["roughness"]["max_distance_km"])
    within = np.isfinite(distance) & (distance <= max_distance)
    assigned = np.where(within, nearest_n, np.nan)
    source = np.where(within, "nearest_within_threshold", "unmatched_distance")
    if config["roughness"]["allow_global_mean_fallback"]:
        fallback = float(np.nanmean(roughness_n))
        missing = ~np.isfinite(assigned)
        assigned[missing] = fallback
        source[missing] = "explicit_global_mean_fallback"
    stations["N_River"] = assigned
    stations["N_source"] = source
    stations["N_distance_km"] = distance
    stations["N_nearest_lon"] = nearest_lon
    stations["N_nearest_lat"] = nearest_lat
    output = output_path(config, "fused", "station_metadata_with_roughness.csv")
    atomic_write_csv(stations, output)
    summary = pd.DataFrame([{
        "stations": len(stations),
        "assigned": int(np.isfinite(assigned).sum()),
        "unassigned": int((~np.isfinite(assigned)).sum()),
        "max_distance_km": max_distance,
        "global_mean_fallback_enabled": config["roughness"]["allow_global_mean_fallback"],
    }])
    atomic_write_csv(summary, output_path(config, "manifests", "roughness_assignment.csv"))
    return stations

