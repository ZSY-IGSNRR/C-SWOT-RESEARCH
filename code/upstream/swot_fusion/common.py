from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def normalize_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_id_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_id).astype("string")


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=target.stem + "_", suffix=".tmp", dir=target.parent)
    os.close(handle)
    try:
        frame.to_csv(temp_name, index=False, encoding="utf-8-sig")
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=target.stem + "_", suffix=".tmp", dir=target.parent)
    os.close(handle)
    try:
        with open(temp_name, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def lonlat_to_unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.radians(np.asarray(lon, dtype=float))
    lat_r = np.radians(np.asarray(lat, dtype=float))
    cos_lat = np.cos(lat_r)
    return np.column_stack((cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)))


def chord_to_km(chord: np.ndarray | float) -> np.ndarray:
    chord_array = np.asarray(chord, dtype=float)
    angle = 2.0 * np.arcsin(np.minimum(1.0, chord_array / 2.0))
    return angle * EARTH_RADIUS_KM


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def read_id_file(path: str | Path, column: str) -> list[str]:
    frame = pd.read_csv(path, dtype={column: "string"}, usecols=[column])
    values = normalize_id_series(frame[column])
    return sorted(value for value in values.dropna().unique().tolist() if value)

