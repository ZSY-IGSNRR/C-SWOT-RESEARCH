from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_write_csv
from .config import output_path


def _file_summary(path: Path, layer: str) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    time_column = next((name for name in ["time_utc", "time_str", "swot_date"] if name in frame), None)
    times = pd.to_datetime(frame[time_column], errors="coerce", utc=True) if time_column else pd.Series(dtype="datetime64[ns, UTC]")
    riversp_discharge_columns = [
        column for column in frame
        if column.startswith(("dschg_", "reach_dschg_"))
        and not column.endswith(("_u", "sf", "_q", "_q_b", "gq_b", "_units"))
    ]
    sos_discharge_columns = [
        column for column in frame
        if (
            (column.startswith("sos_") and not column.startswith("sos_v2_"))
            or column.startswith("sos_v2_sos_")
            or column in {"swot_discharge_reanalysis", "sos_v2_swot_discharge_reanalysis"}
        )
        and not column.endswith("_units")
        and column not in {"sos_v2_time_utc", "sos_v2_match_delta_seconds"}
    ]
    discharge_columns = list(dict.fromkeys(riversp_discharge_columns + sos_discharge_columns))
    discharge_values = frame[discharge_columns].apply(pd.to_numeric, errors="coerce") if discharge_columns else pd.DataFrame(index=frame.index)
    if discharge_columns:
        discharge_values = discharge_values.mask(discharge_values <= -1.0e10)
    return {
        "layer": layer,
        "file": path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "time_min": times.min() if not times.empty else pd.NaT,
        "time_max": times.max() if not times.empty else pd.NaT,
        "unique_node_ids": frame["node_id"].astype(str).nunique() if "node_id" in frame else np.nan,
        "unique_reach_ids": frame["reach_id"].astype(str).nunique() if "reach_id" in frame else np.nan,
        "hydraulic_valid_rows": int(frame["qc_hydraulic_valid"].fillna(False).sum()) if "qc_hydraulic_valid" in frame else np.nan,
        "discharge_columns": len(discharge_columns),
        "rows_with_any_discharge": int(discharge_values.notna().any(axis=1).sum()) if discharge_columns else 0,
        "collection_versions": "|".join(sorted(frame["collection_version"].dropna().astype(str).unique())) if "collection_version" in frame else "",
        "sword_versions": "|".join(sorted(frame["sword_version"].dropna().astype(str).unique())) if "sword_version" in frame else "",
    }


def audit_outputs(config: dict) -> pd.DataFrame:
    layers = {
        "raw_node_d": output_path(config, "raw", "node_d"),
        "raw_reach_d": output_path(config, "raw", "reach_d"),
        "raw_reach_sos_v2": output_path(config, "raw", "reach_sos_v2"),
        "qc_node_d": output_path(config, "qc", "node_d"),
        "qc_reach_d": output_path(config, "qc", "reach_d"),
        "node_reach": output_path(config, "fused", "node_reach"),
        "station_overpasses": output_path(config, "fused", "station_overpasses"),
        "station_adjusted": output_path(config, "fused", "station_overpasses_datum_adjusted"),
    }
    records: list[dict] = []
    for layer, directory in layers.items():
        for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
            try:
                records.append(_file_summary(path, layer))
            except Exception as exc:
                records.append({
                    "layer": layer, "file": path.name, "rows": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    detail = pd.DataFrame(records)
    atomic_write_csv(detail, output_path(config, "audit", "file_level_audit.csv"))
    if not detail.empty:
        layer = detail.groupby("layer", dropna=False).agg(
            files=("file", "count"), rows=("rows", "sum"),
            rows_with_any_discharge=("rows_with_any_discharge", "sum"),
            time_min=("time_min", "min"), time_max=("time_max", "max"),
        ).reset_index()
    else:
        layer = pd.DataFrame(columns=["layer", "files", "rows"])
    atomic_write_csv(layer, output_path(config, "audit", "layer_summary.csv"))
    return layer
