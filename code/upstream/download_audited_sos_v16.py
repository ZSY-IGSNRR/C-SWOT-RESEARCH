#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download SoS Version-2.0 data using audited SWORD v16 reach IDs.

Only stations present in the hydraulic-model holdout prediction table are
requested.  Existing Version-2.0 files are reused after checking that their
reported SWORD version and reach ID agree with the requested v16 feature.
All outputs remain local.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from swot_fusion.config import load_config
from swot_fusion.fields import COLLECTIONS, reach_sos_v2_field_groups
from swot_fusion.hydrocron import DownloadResult, _download_one


SCHEMES = {"wse_stratified", "temporal_forward"}


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def reusable_source(path: Path, feature_id: str) -> bool:
    try:
        sample = pd.read_csv(
            path,
            usecols=lambda value: value in {"reach_id", "sword_version", "collection_version"},
            nrows=10,
            dtype="string",
        )
    except Exception:
        return False
    if sample.empty or "reach_id" not in sample:
        return False
    ids = set(normalize_id(sample["reach_id"]).dropna())
    versions = set(normalize_id(sample.get("sword_version", pd.Series(dtype="string"))).dropna())
    return ids == {feature_id} and (not versions or versions == {"16"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-audit", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["download"]["max_workers"] = args.workers
    config["download"]["overwrite"] = args.overwrite

    audit = pd.read_csv(args.station_audit, dtype={"station_id": "string", "v16_reach_id": "string"})
    prediction_stations = pd.read_csv(
        args.predictions,
        usecols=["station_id", "method", "validation_scheme"],
        dtype={"station_id": "string"},
        low_memory=False,
    )
    prediction_stations = set(
        normalize_id(
            prediction_stations.loc[
                prediction_stations["method"].eq("full")
                & prediction_stations["validation_scheme"].isin(SCHEMES),
                "station_id",
            ]
        ).dropna()
    )
    audit["station_id"] = normalize_id(audit["station_id"])
    audit["v16_reach_id"] = normalize_id(audit["v16_reach_id"])
    usable = audit[
        audit["station_id"].isin(prediction_stations)
        & audit["mapping_status"].isin(
            ["stable_same_id_high", "remapped_high", "remapped_medium"]
        )
        & audit["v16_reach_id"].notna()
    ].copy()
    feature_ids = sorted(set(usable["v16_reach_id"]), key=int)
    pd.DataFrame({"v16_reach_id": feature_ids}).to_csv(
        args.output_dir / "requested_v16_reach_ids.csv", index=False, encoding="utf-8-sig"
    )

    results: list[DownloadResult] = []
    to_download: list[str] = []
    for feature_id in feature_ids:
        destination = args.output_dir / f"{feature_id}.csv"
        source = args.source_dir / f"{feature_id}.csv"
        if destination.exists() and not args.overwrite and reusable_source(destination, feature_id):
            results.append(DownloadResult(
                "reach_sos_v2_audited", "Reach", feature_id,
                COLLECTIONS["reach_sos_v2"], "skipped_existing",
                output_file=str(destination),
            ))
        elif source.exists() and not args.overwrite and reusable_source(source, feature_id):
            shutil.copy2(source, destination)
            row_count = max(sum(1 for _ in destination.open("r", encoding="utf-8")) - 1, 0)
            results.append(DownloadResult(
                "reach_sos_v2_audited", "Reach", feature_id,
                COLLECTIONS["reach_sos_v2"], "reused_verified_v16",
                rows=row_count, output_file=str(destination),
            ))
        else:
            to_download.append(feature_id)

    if to_download:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _download_one,
                    config=config,
                    profile="reach_sos_v2_audited",
                    feature="Reach",
                    feature_id=feature_id,
                    collection_name=COLLECTIONS["reach_sos_v2"],
                    field_groups=reach_sos_v2_field_groups(),
                    output_dir=args.output_dir,
                ): feature_id
                for feature_id in to_download
            }
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % 50 == 0 or completed == len(to_download):
                    print(f"download progress: {completed}/{len(to_download)}", flush=True)

    manifest = pd.DataFrame([asdict(result) for result in results]).sort_values("feature_id")
    manifest.to_csv(
        args.output_dir / "download_manifest.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "prediction_station_count": len(prediction_stations),
        "mapped_station_count": int(usable["station_id"].nunique()),
        "requested_v16_reach_count": len(feature_ids),
        "download_status_counts": manifest["status"].value_counts(dropna=False).to_dict(),
        "collection": COLLECTIONS["reach_sos_v2"],
        "start_time": config["download"]["start_time"],
        "end_time": config["download"]["end_time"],
        "source_station_mapping": str(args.station_audit),
    }
    (args.output_dir / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
