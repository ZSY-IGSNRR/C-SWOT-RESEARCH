#!/usr/bin/env python3
"""Download only the Asia member of SWORD v16 and export compact indexes.

The full Zenodo archive is about 1.57 GB.  This script reuses the tested HTTP
Range ZIP reader from ``download_sword_v17b_asia.py`` so that only the
compressed Asia NetCDF member is transferred.  The original NetCDF is retained
because the v16-v17b audit needs reach centerline coordinates.
"""

from __future__ import annotations

import argparse
import binascii
import json
from pathlib import Path
import struct
import time
import zlib

import netCDF4
import numpy as np
import pandas as pd

from download_sword_v17b_asia import fetch_range, zip_entries


DEFAULT_URL = (
    "https://zenodo.org/api/records/10013982/files/"
    "SWORD_v16_netcdf.zip/content"
)
MEMBER_NAME = "netcdf/as_sword_v16.nc"


def extract_member_chunked(
    url: str,
    entry: dict,
    destination: Path,
    *,
    compressed_chunk_bytes: int = 16 * 1024 * 1024,
) -> None:
    """Extract one deflated ZIP member using bounded HTTP Range requests.

    Zenodo occasionally closes a single very large Range response early.  A
    persistent zlib stream can safely consume consecutive smaller byte ranges,
    while every range is length checked and retried independently.
    """
    local = fetch_range(url, entry["local_offset"], entry["local_offset"] + 29)
    values = struct.unpack("<4s5H3L2H", local)
    if values[0] != b"PK\x03\x04":
        raise ValueError("Invalid local-file signature")
    name_length, extra_length = values[9], values[10]
    data_start = entry["local_offset"] + 30 + name_length + extra_length
    data_end = data_start + entry["compressed_size"] - 1
    if entry["method"] not in {0, 8}:
        raise NotImplementedError(
            f"ZIP compression method {entry['method']} is not supported"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if entry["method"] == 8 else None
    crc = 0
    written = 0
    with temporary.open("wb") as output:
        start = data_start
        while start <= data_end:
            end = min(data_end, start + compressed_chunk_bytes - 1)
            last_error: Exception | None = None
            for attempt in range(5):
                try:
                    compressed = fetch_range(url, start, end)
                    last_error = None
                    break
                except Exception as exc:  # network retry with bounded backoff
                    last_error = exc
                    time.sleep(2 ** attempt)
            if last_error is not None:
                raise last_error
            plain = decompressor.decompress(compressed) if decompressor else compressed
            if plain:
                output.write(plain)
                crc = binascii.crc32(plain, crc)
                written += len(plain)
            start = end + 1
        if decompressor:
            plain = decompressor.flush()
            if plain:
                output.write(plain)
                crc = binascii.crc32(plain, crc)
                written += len(plain)

    crc &= 0xFFFFFFFF
    if written != entry["uncompressed_size"] or crc != entry["crc32"]:
        raise IOError(
            "Extracted member verification failed: "
            f"size {written}/{entry['uncompressed_size']}, "
            f"CRC {crc:08x}/{entry['crc32']:08x}"
        )
    temporary.replace(destination)


def _numeric(variable, dtype=None) -> np.ndarray:
    values = np.ma.asarray(variable[:], dtype=dtype)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan if dtype is float else -999)
    return np.asarray(values, dtype=dtype)


def _decode_text(variable) -> list[str]:
    values = variable[:]
    try:
        values = netCDF4.chartostring(values)
    except (AttributeError, ValueError):
        pass
    array = np.asarray(values)
    if array.ndim == 1:
        return [str(value).strip() for value in array]
    return ["".join(map(str, row)).strip() for row in array]


def export_indexes(netcdf_path: Path, output_dir: Path) -> dict[str, int]:
    with netCDF4.Dataset(netcdf_path, "r") as root:
        reaches = root.groups["reaches"]
        reach_count = len(reaches.dimensions["num_reaches"])

        def optional_reach(name: str) -> np.ndarray:
            if name not in reaches.variables:
                return np.full(reach_count, np.nan)
            return _numeric(reaches.variables[name], float)

        reach_frame = pd.DataFrame({
            "reach_id": _numeric(reaches.variables["reach_id"], np.int64),
            "x": _numeric(reaches.variables["x"], float),
            "y": _numeric(reaches.variables["y"], float),
            "reach_length": _numeric(reaches.variables["reach_length"], float),
            "n_nodes": _numeric(reaches.variables["n_nodes"], float),
            "network": optional_reach("network"),
            "stream_order": optional_reach("stream_order"),
            "trib_flag": _numeric(reaches.variables["trib_flag"], float),
            "river_name": _decode_text(reaches.variables["river_name"]),
        })
        nodes = root.groups["nodes"]
        node_frame = pd.DataFrame({
            "node_id": _numeric(nodes.variables["node_id"], np.int64),
            "reach_id": _numeric(nodes.variables["reach_id"], np.int64),
            "x": _numeric(nodes.variables["x"], float),
            "y": _numeric(nodes.variables["y"], float),
        })

    for frame in (reach_frame, node_frame):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame.dropna(subset=["x", "y"], inplace=True)
        frame.drop(frame[~frame["x"].between(-180, 180)].index, inplace=True)
        frame.drop(frame[~frame["y"].between(-90, 90)].index, inplace=True)

    reach_path = output_dir / "as_sword_reaches_v16.csv"
    node_path = output_dir / "as_sword_nodes_v16.csv"
    reach_frame.drop_duplicates("reach_id").to_csv(
        reach_path, index=False, encoding="utf-8-sig"
    )
    node_frame.drop_duplicates("node_id").to_csv(
        node_path, index=False, encoding="utf-8-sig"
    )
    return {"reach_count": len(reach_frame), "node_count": len(node_frame)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output-dir",
        default='inputs/sword_v16_asia',
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    netcdf_path = output_dir / "as_sword_v16.nc"

    archive_size, entries = zip_entries(args.url)
    matches = [
        entry for entry in entries
        if entry["name"].replace("\\", "/").lower() == MEMBER_NAME
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {MEMBER_NAME}; found {len(matches)}"
        )
    entry = matches[0]
    if not netcdf_path.exists():
        extract_member_chunked(args.url, entry, netcdf_path)

    counts = export_indexes(netcdf_path, output_dir)
    manifest = {
        "source_url": args.url,
        "zenodo_doi": "10.5281/zenodo.10013982",
        "archive_size_bytes": archive_size,
        "member": entry,
        "netcdf_path": str(netcdf_path),
        "reach_index_csv": str(output_dir / "as_sword_reaches_v16.csv"),
        "node_index_csv": str(output_dir / "as_sword_nodes_v16.csv"),
        **counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
