#!/usr/bin/env python3
"""Range-download only the Asia NetCDF member from the official SWORD v17b ZIP.

The Zenodo archive is about 1.7 GiB and contains all continents.  This script
reads the ZIP central directory with HTTP Range requests, downloads only the
``as_sword_v17b.nc`` compressed member, verifies its CRC32, and exports the
four-column node index required by the fusion inventory.
"""

from __future__ import annotations

import argparse
import binascii
import json
from pathlib import Path
import struct
import urllib.request
import zlib

import netCDF4
import numpy as np
import pandas as pd


DEFAULT_URL = (
    "https://zenodo.org/records/15299138/files/"
    "SWORD_v17b_netcdf.zip?download=1"
)


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "SWOT-local-fusion/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        if len(data) != end - start + 1:
            raise IOError(
                f"Range length mismatch for {start}-{end}: got {len(data)} bytes"
            )
        return data


def zip_entries(url: str) -> tuple[int, list[dict]]:
    size = content_length(url)
    tail_start = max(0, size - 131_072)
    tail = fetch_range(url, tail_start, size - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise ValueError("ZIP end-of-central-directory record was not found")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_at)
    entry_count = eocd[4]
    central_size = eocd[5]
    central_offset = eocd[6]
    if entry_count == 0xFFFF or central_size == 0xFFFFFFFF:
        raise NotImplementedError("ZIP64 central directory is not supported")
    if central_offset >= tail_start and central_offset + central_size <= size:
        start = central_offset - tail_start
        central = tail[start : start + central_size]
    else:
        central = fetch_range(
            url, central_offset, central_offset + central_size - 1
        )

    entries: list[dict] = []
    cursor = 0
    for _ in range(entry_count):
        values = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        if values[0] != b"PK\x01\x02":
            raise ValueError(f"Invalid central-directory signature at {cursor}")
        flags, method = values[3], values[4]
        crc32, compressed_size, uncompressed_size = values[7:10]
        name_length, extra_length, comment_length = values[10:13]
        local_offset = values[16]
        name_start = cursor + 46
        name_bytes = central[name_start : name_start + name_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        entries.append({
            "name": name_bytes.decode(encoding),
            "flags": flags,
            "method": method,
            "crc32": crc32,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        })
        cursor = name_start + name_length + extra_length + comment_length
    return size, entries


def extract_member(url: str, entry: dict, destination: Path) -> None:
    local = fetch_range(
        url, entry["local_offset"], entry["local_offset"] + 29
    )
    values = struct.unpack("<4s5H3L2H", local)
    if values[0] != b"PK\x03\x04":
        raise ValueError("Invalid local-file signature")
    name_length, extra_length = values[9], values[10]
    data_start = entry["local_offset"] + 30 + name_length + extra_length
    data_end = data_start + entry["compressed_size"] - 1
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={data_start}-{data_end}",
            "User-Agent": "SWOT-local-fusion/1.0",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    decompressor = (
        zlib.decompressobj(-zlib.MAX_WBITS)
        if entry["method"] == 8
        else None
    )
    if entry["method"] not in {0, 8}:
        raise NotImplementedError(
            f"ZIP compression method {entry['method']} is not supported"
        )
    crc = 0
    written = 0
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            plain = decompressor.decompress(chunk) if decompressor else chunk
            if plain:
                output.write(plain)
                crc = binascii.crc32(plain, crc)
                written += len(plain)
        if decompressor:
            plain = decompressor.flush()
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


def export_node_index(netcdf_path: Path, csv_path: Path) -> int:
    with netCDF4.Dataset(netcdf_path, "r") as root:
        group = root.groups.get("nodes")
        if group is None:
            raise KeyError(f"No 'nodes' group in {netcdf_path}")
        required = ["node_id", "reach_id", "x", "y"]
        missing = [name for name in required if name not in group.variables]
        if missing:
            raise KeyError(f"Missing node variables: {missing}")
        frame = pd.DataFrame({
            "node_id": np.asarray(group.variables["node_id"][:]).astype("int64"),
            "reach_id": np.asarray(group.variables["reach_id"][:]).astype("int64"),
            "x": np.asarray(group.variables["x"][:], dtype=float),
            "y": np.asarray(group.variables["y"][:], dtype=float),
        })
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[
        frame["x"].between(-180, 180) & frame["y"].between(-90, 90)
    ].drop_duplicates("node_id")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return len(frame)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output-dir",
        default='inputs/sword_v17b_asia',
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    netcdf_path = output_dir / "as_sword_v17b.nc"
    csv_path = output_dir / "as_sword_nodes_v17b.csv"
    archive_size, entries = zip_entries(args.url)
    matches = [
        entry
        for entry in entries
        if entry["name"].replace("\\", "/").lower() == "netcdf/as_sword_v17b.nc"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one Asia member; archive entries are: "
            + ", ".join(entry["name"] for entry in entries)
        )
    entry = matches[0]
    if not netcdf_path.exists():
        extract_member(args.url, entry, netcdf_path)
    row_count = export_node_index(netcdf_path, csv_path)
    manifest = {
        "source_url": args.url,
        "zenodo_record": "10.5281/zenodo.15299138",
        "archive_size_bytes": archive_size,
        "member": entry,
        "netcdf_path": str(netcdf_path),
        "node_index_csv": str(csv_path),
        "node_count": row_count,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
