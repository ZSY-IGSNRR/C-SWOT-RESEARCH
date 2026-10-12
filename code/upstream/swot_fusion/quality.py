from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_write_csv, read_id_file
from .config import output_path


SENTINEL_LIMIT = -1.0e10
MEASUREMENT_PREFIXES = (
    "wse", "width", "area_", "d_x_area", "slope", "dschg_", "sos_",
    "swot_discharge", "p_wse", "p_width", "p_maf", "geoid_",
)


def hampel_flags(series: pd.Series, window: int, sigma: float) -> pd.Series:
    """Return Hampel outlier flags while preserving the input index."""
    values = pd.to_numeric(series, errors="coerce")
    median = values.rolling(window=2 * window + 1, center=True, min_periods=3).median()
    abs_dev = (values - median).abs()
    mad = abs_dev.rolling(window=2 * window + 1, center=True, min_periods=3).median()
    robust_scale = 1.4826 * mad
    return (abs_dev > sigma * robust_scale) & robust_scale.gt(0)


def _clean_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    measurement_columns = [
        column for column in frame.columns
        if column.startswith(MEASUREMENT_PREFIXES) and not column.endswith("_units")
    ]
    if not measurement_columns:
        return frame.copy()
    numeric = frame[measurement_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.mask(numeric <= SENTINEL_LIMIT)
    # Concatenation creates a compact DataFrame rather than inserting hundreds
    # of columns one at a time (which triggers pandas fragmentation warnings).
    cleaned = pd.concat([frame.drop(columns=measurement_columns), numeric], axis=1)
    return cleaned.loc[:, frame.columns].copy()


def quality_control_frame(frame: pd.DataFrame, profile: str, config: dict) -> tuple[pd.DataFrame, dict]:
    quality = config["quality"]
    frame = _clean_measurements(frame)
    frame["time_utc"] = pd.to_datetime(frame.get("time_str"), errors="coerce", utc=True)
    frame = frame.sort_values("time_utc").reset_index(drop=True)

    overall_column = "node_q" if profile == "node_d" else "reach_q"
    accepted = quality["accepted_node_q"] if profile == "node_d" else quality["accepted_reach_q"]
    if overall_column in frame:
        q = pd.to_numeric(frame[overall_column], errors="coerce")
        frame["qc_overall_quality_ok"] = q.isin(accepted)
    else:
        # SoS-only responses do not carry RiverSP reach_q.
        frame["qc_overall_quality_ok"] = True

    frame["qc_time_ok"] = frame["time_utc"].notna()
    if profile == "node_d":
        frame["qc_hydraulic_valid"] = (
            pd.to_numeric(frame.get("wse"), errors="coerce").notna()
            & pd.to_numeric(frame.get("width"), errors="coerce").ge(0)
        )
    elif profile == "reach_d":
        frame["qc_hydraulic_valid"] = (
            pd.to_numeric(frame.get("wse"), errors="coerce").notna()
            & pd.to_numeric(frame.get("width"), errors="coerce").ge(0)
        )
        frame["qc_slope_valid"] = pd.to_numeric(frame.get("slope"), errors="coerce").gt(0)
    else:
        sos_columns = [column for column in frame if column.startswith("sos_") or column == "swot_discharge_reanalysis"]
        frame["qc_hydraulic_valid"] = frame[sos_columns].notna().any(axis=1) if sos_columns else False

    discharge_value_columns = [
        column for column in frame.columns
        if column.startswith("dschg_")
        and not column.endswith(("_u", "sf", "_q", "_q_b", "gq_b", "_units"))
    ]
    for value_column in discharge_value_columns:
        quality_column = f"{value_column}_q"
        if quality_column in frame:
            frame[f"qc_{value_column}_usable"] = (
                frame[value_column].notna()
                & pd.to_numeric(frame[quality_column], errors="coerce").isin(quality["accepted_discharge_q"])
            )

    hampel_columns = [column for column in ["wse", "width", "slope"] if column in frame]
    hampel_any = pd.Series(False, index=frame.index)
    for column in hampel_columns:
        flags = pd.Series(False, index=frame.index)
        # Different SWOT passes can have persistent cross-track offsets. Hampel
        # filtering across passes would mislabel that structure as temporal
        # outliers, so filter each pass series independently when possible.
        groups = frame.groupby("pass_id", dropna=False).groups if "pass_id" in frame else {"all": frame.index}
        for indices in groups.values():
            flags.loc[indices] = hampel_flags(
                frame.loc[indices, column],
                int(quality["hampel_window"]),
                float(quality["hampel_sigma"]),
            )
        frame[f"qc_hampel_{column}"] = flags
        hampel_any |= flags.fillna(False)
    frame["qc_hampel_any"] = hampel_any
    frame["qc_keep"] = frame["qc_overall_quality_ok"] & frame["qc_time_ok"]
    if quality["exclude_hampel_outliers"]:
        frame["qc_keep"] &= ~frame["qc_hampel_any"]

    audit = {
        "profile": profile,
        "rows_raw": len(frame),
        "rows_kept": int(frame["qc_keep"].sum()),
        "rows_quality_rejected": int((~frame["qc_overall_quality_ok"]).sum()),
        "rows_invalid_time": int((~frame["qc_time_ok"]).sum()),
        "rows_hampel_flagged": int(frame["qc_hampel_any"].sum()),
        "rows_hydraulic_valid": int(frame["qc_hydraulic_valid"].sum()),
    }
    if "qc_slope_valid" in frame:
        audit["rows_positive_slope"] = int(frame["qc_slope_valid"].sum())
    return frame.loc[frame["qc_keep"]].copy(), audit


def quality_control_profile(config: dict, profile: str) -> pd.DataFrame:
    raw_dir = output_path(config, "raw", profile)
    qc_dir = output_path(config, "qc", profile)
    qc_dir.mkdir(parents=True, exist_ok=True)
    id_file, id_column = (
        ("node_ids.csv", "node_id")
        if profile == "node_d"
        else ("reach_ids.csv", "reach_id")
    )
    feature_ids = read_id_file(
        output_path(config, "inventory", id_file), id_column
    )
    active_names = {f"{feature_id}.csv" for feature_id in feature_ids}
    # QC is a derived layer. Remove products from older inventories so that
    # downstream attachment cannot accidentally process stale node/reach IDs.
    for stale_path in qc_dir.glob("*.csv"):
        if stale_path.name not in active_names:
            stale_path.unlink()
    records: list[dict] = []
    for feature_id in feature_ids:
        path = raw_dir / f"{feature_id}.csv"
        if not path.exists():
            records.append({
                "feature_id": feature_id,
                "profile": profile,
                "status": "missing_raw_file",
            })
            continue
        try:
            raw = pd.read_csv(path, low_memory=False)
            clean, audit = quality_control_frame(raw, profile, config)
            atomic_write_csv(clean, qc_dir / path.name)
            records.append({"feature_id": path.stem, "status": "ok", **audit})
        except Exception as exc:
            records.append({
                "feature_id": path.stem, "profile": profile, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
    audit_frame = pd.DataFrame(records)
    atomic_write_csv(audit_frame, output_path(config, "manifests", f"quality_{profile}.csv"))
    return audit_frame


def run_quality_control(config: dict) -> dict[str, pd.DataFrame]:
    profiles = ["node_d", "reach_d"]
    if config["download"]["include_sos_v2"]:
        profiles.append("reach_sos_v2")
    return {profile: quality_control_profile(config, profile) for profile in profiles}
