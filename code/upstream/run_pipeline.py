#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from swot_fusion.audit import audit_outputs
from swot_fusion.config import load_config
from swot_fusion.fusion import attach_reach_to_nodes, fuse_stations, robust_datum_shift
from swot_fusion.hydrocron import download_swot
from swot_fusion.inventory import build_inventory
from swot_fusion.quality import run_quality_control
from swot_fusion.roughness import assign_roughness


STEP_ORDER = ["inventory", "download", "qc", "attach", "fuse", "roughness", "shift", "audit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SWOT Hydrocron data, apply QC, and fuse with gauge observations."
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config.example.json")),
        help="JSON configuration file (default: config.example.json)",
    )
    parser.add_argument(
        "--steps", default=",".join(STEP_ORDER),
        help=f"Comma-separated steps chosen from: {','.join(STEP_ORDER)}",
    )
    parser.add_argument("--limit", type=int, help="Download only the first N node and reach IDs (smoke testing)")
    parser.add_argument("--include-sos-v2", action="store_true", help="Also download the separate Version 2.0 SoS discharge layer")
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing raw feature files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.include_sos_v2:
        config["download"]["include_sos_v2"] = True
    if args.overwrite:
        config["download"]["overwrite"] = True
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = sorted(set(steps) - set(STEP_ORDER))
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    actions = {
        "inventory": lambda: build_inventory(config),
        "download": lambda: download_swot(config, limit=args.limit),
        "qc": lambda: run_quality_control(config),
        "attach": lambda: attach_reach_to_nodes(config),
        "fuse": lambda: fuse_stations(config),
        "roughness": lambda: assign_roughness(config),
        "shift": lambda: robust_datum_shift(config),
        "audit": lambda: audit_outputs(config),
    }
    for step in STEP_ORDER:
        if step in steps:
            logging.info("START %s", step)
            actions[step]()
            logging.info("DONE  %s", step)
    return 0


if __name__ == "__main__":
    sys.exit(main())

