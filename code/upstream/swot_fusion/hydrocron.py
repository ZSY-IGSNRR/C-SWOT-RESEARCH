from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .common import atomic_write_csv, normalize_id_series, read_id_file
from .config import output_path
from .fields import COLLECTIONS, node_field_groups, reach_d_field_groups, reach_sos_v2_field_groups


HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"


@dataclass
class DownloadResult:
    profile: str
    feature: str
    feature_id: str
    collection_name: str
    status: str
    rows: int = 0
    columns: int = 0
    parts: int = 0
    seconds: float = 0.0
    output_file: str = ""
    error: str = ""


def _unwrap_payload(body: str) -> str:
    """Normalize direct CSV, JSON-encoded CSV, or common API wrapper objects."""
    value: Any = body.lstrip("\ufeff").strip()
    for _ in range(5):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ""
            if stripped[0] not in {'"', "{", "["}:
                return stripped
            try:
                value = json.loads(stripped)
                continue
            except json.JSONDecodeError:
                return stripped
        if isinstance(value, dict):
            for key in ("results", "result", "data", "body", "csv"):
                if key in value:
                    value = value[key]
                    break
            else:
                raise ValueError(f"Unrecognized Hydrocron JSON keys: {sorted(value)}")
            continue
        if isinstance(value, list):
            if not value:
                return ""
            if all(isinstance(item, dict) for item in value):
                return pd.DataFrame(value).to_csv(index=False)
            value = "\n".join(str(item) for item in value)
            continue
        return str(value)
    raise ValueError("Hydrocron response was nested too deeply")


def parse_hydrocron_csv(body: str) -> pd.DataFrame:
    csv_text = _unwrap_payload(body)
    if not csv_text or csv_text.lower() in {"no data", "no_data", "null"}:
        return pd.DataFrame()
    frame = pd.read_csv(StringIO(csv_text), low_memory=False)
    if len(frame) == 1 and len(frame.columns) == 1:
        cell = str(frame.iloc[0, 0]).strip().lower()
        if cell in {"no data", "no_data", "null"}:
            return pd.DataFrame()
    return frame


def _iso_utc(timestamp: pd.Timestamp) -> str:
    stamp = timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_once(
    feature: str,
    feature_id: str,
    collection_name: str,
    fields: list[str],
    start_time: str,
    end_time: str,
    timeout: int,
    api_key: str | None,
) -> pd.DataFrame:
    params = {
        "feature": feature,
        "feature_id": feature_id,
        "output": "csv",
        "start_time": start_time,
        "end_time": end_time,
        "collection_name": collection_name,
        "fields": ",".join(fields),
    }
    request = Request(
        f"{HYDROCRON_URL}?{urlencode(params)}",
        headers={
            # With output=csv Hydrocron expects the default/application-json
            # wrapper; advertising text/csv at the same time is a 400 error.
            "Accept": "application/json",
            "User-Agent": "swot-data-download-fusion/1.0",
            **({"x-hydrocron-key": api_key} if api_key else {}),
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return parse_hydrocron_csv(response.read().decode(charset, errors="replace"))


def _request_with_retry_and_split(
    *,
    feature: str,
    feature_id: str,
    collection_name: str,
    fields: list[str],
    start_time: str,
    end_time: str,
    timeout: int,
    api_key: str | None,
    retries: int,
    backoff: float,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _request_once(
                feature, feature_id, collection_name, fields,
                start_time, end_time, timeout, api_key,
            )
        except HTTPError as exc:
            last_error = exc
            if exc.code == 400:
                # Hydrocron represents a valid feature with no observations in
                # the requested collection/time range as HTTP 400 rather than
                # an empty CSV.  Treat only that explicit response as no-data;
                # other 400 responses remain real request errors.
                detail = exc.read().decode("utf-8", errors="replace").lower()
                if (
                    "results with the specified feature id" in detail
                    and "were not found" in detail
                ):
                    return pd.DataFrame()
            if exc.code == 413:
                start = pd.Timestamp(start_time)
                end = pd.Timestamp(end_time)
                if end - start <= pd.Timedelta(days=2):
                    raise
                midpoint = start + (end - start) / 2
                left = _request_with_retry_and_split(
                    feature=feature, feature_id=feature_id,
                    collection_name=collection_name, fields=fields,
                    start_time=_iso_utc(start), end_time=_iso_utc(midpoint),
                    timeout=timeout, api_key=api_key, retries=retries, backoff=backoff,
                )
                right = _request_with_retry_and_split(
                    feature=feature, feature_id=feature_id,
                    collection_name=collection_name, fields=fields,
                    start_time=_iso_utc(midpoint + pd.Timedelta(seconds=1)), end_time=_iso_utc(end),
                    timeout=timeout, api_key=api_key, retries=retries, backoff=backoff,
                )
                return pd.concat([left, right], ignore_index=True).drop_duplicates()
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(backoff * (2**attempt) + random.uniform(0, 0.5))
    assert last_error is not None
    raise last_error


def merge_field_parts(parts: list[pd.DataFrame], feature: str) -> pd.DataFrame:
    nonempty = [part.copy() for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame()
    id_columns = ["node_id", "reach_id"] if feature == "Node" else ["reach_id"]
    for frame in nonempty:
        for column in id_columns:
            if column in frame:
                frame[column] = normalize_id_series(frame[column])
    preferred_keys = id_columns + ["time_str", "cycle_id", "pass_id"]
    def collapse_duplicate_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        frame = frame.drop_duplicates().copy()
        if not frame.duplicated(keys, keep=False).any():
            return frame
        aggregation: dict[str, str] = {}
        for column in frame.columns:
            if column in keys:
                continue
            aggregation[column] = (
                "median" if pd.api.types.is_numeric_dtype(frame[column]) else "first"
            )
        return frame.groupby(keys, as_index=False, dropna=False).agg(aggregation)

    initial_keys = [key for key in preferred_keys if key in nonempty[0].columns]
    merged = collapse_duplicate_keys(nonempty[0], initial_keys)
    for part_number, part in enumerate(nonempty[1:], start=2):
        keys = [key for key in preferred_keys if key in merged.columns and key in part.columns]
        if not keys:
            raise ValueError(f"Cannot merge Hydrocron field part {part_number}; no shared observation keys")
        merged = collapse_duplicate_keys(merged, keys)
        part = collapse_duplicate_keys(part, keys)
        duplicate_columns = [column for column in part.columns if column in merged.columns and column not in keys]
        part = part.drop(columns=duplicate_columns)
        merged = merged.merge(part, how="outer", on=keys, validate="one_to_one")
    sort_columns = [column for column in [*id_columns, "time_str"] if column in merged]
    return merged.sort_values(sort_columns).reset_index(drop=True) if sort_columns else merged


def _download_one(
    *,
    config: dict,
    profile: str,
    feature: str,
    feature_id: str,
    collection_name: str,
    field_groups: list[list[str]],
    output_dir: Path,
) -> DownloadResult:
    started = time.perf_counter()
    output_file = output_dir / f"{feature_id}.csv"
    if output_file.exists() and not config["download"]["overwrite"]:
        return DownloadResult(
            profile, feature, feature_id, collection_name, "skipped_existing",
            output_file=str(output_file), seconds=time.perf_counter() - started,
        )
    api_env = config["download"]["api_key_environment_variable"]
    api_key = os.environ.get(api_env) if api_env else None
    try:
        parts = [
            _request_with_retry_and_split(
                feature=feature,
                feature_id=feature_id,
                collection_name=collection_name,
                fields=fields,
                start_time=config["download"]["start_time"],
                end_time=config["download"]["end_time"],
                timeout=int(config["download"]["timeout_seconds"]),
                api_key=api_key,
                retries=int(config["download"]["retries"]),
                backoff=float(config["download"]["retry_backoff_seconds"]),
            )
            for fields in field_groups
        ]
        frame = merge_field_parts(parts, feature)
        status = "no_data" if frame.empty else "downloaded"
        if not frame.empty:
            atomic_write_csv(frame, output_file)
        return DownloadResult(
            profile, feature, feature_id, collection_name, status,
            rows=len(frame), columns=len(frame.columns), parts=len(field_groups),
            seconds=time.perf_counter() - started,
            output_file=str(output_file) if not frame.empty else "",
        )
    except Exception as exc:  # manifest keeps failures; batch continues
        return DownloadResult(
            profile, feature, feature_id, collection_name, "error",
            parts=len(field_groups), seconds=time.perf_counter() - started,
            output_file=str(output_file), error=f"{type(exc).__name__}: {exc}",
        )


def _download_profile(
    config: dict,
    profile: str,
    feature: str,
    ids: list[str],
    field_groups: list[list[str]],
) -> pd.DataFrame:
    collection = COLLECTIONS[profile]
    output_dir = output_path(config, "raw", profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=int(config["download"]["max_workers"])) as executor:
        futures = {
            executor.submit(
                _download_one,
                config=config, profile=profile, feature=feature,
                feature_id=feature_id, collection_name=collection,
                field_groups=field_groups, output_dir=output_dir,
            ): feature_id
            for feature_id in ids
        }
        for future in as_completed(futures):
            results.append(future.result())
    frame = pd.DataFrame([asdict(result) for result in results]).sort_values("feature_id")
    manifest = output_path(config, "manifests", f"download_{profile}.csv")
    atomic_write_csv(frame, manifest)
    return frame


def download_swot(config: dict, limit: int | None = None) -> dict[str, pd.DataFrame]:
    inventory = output_path(config, "inventory")
    node_ids = read_id_file(inventory / "node_ids.csv", "node_id")
    reach_ids = read_id_file(inventory / "reach_ids.csv", "reach_id")
    if limit is not None:
        node_ids = node_ids[:limit]
        reach_ids = reach_ids[:limit]
    outputs = {
        "node_d": _download_profile(config, "node_d", "Node", node_ids, node_field_groups()),
        "reach_d": _download_profile(config, "reach_d", "Reach", reach_ids, reach_d_field_groups()),
    }
    if config["download"]["include_sos_v2"]:
        outputs["reach_sos_v2"] = _download_profile(
            config, "reach_sos_v2", "Reach", reach_ids, reach_sos_v2_field_groups()
        )
    return outputs
