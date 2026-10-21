"""Paths shared by the fivefold workflows; numerical settings live in config."""
from pathlib import Path
import os
import json

SOURCE = Path(__file__).resolve().parents[1]
REPOSITORY = SOURCE.parents[1]
CALIBRATION = SOURCE.parent / "calibration"


def work_root():
    value = os.environ.get("CSWOT_FIVEFOLD_ROOT")
    if not value:
        raise ValueError("Use code/run.py fivefold with an explicit --root outside the repository.")
    root = Path(value).expanduser().resolve()
    if root == REPOSITORY or REPOSITORY in root.parents:
        raise ValueError("The fivefold work root must be outside the repository.")
    return root


def external_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def configuration_path():
    explicit = external_path("CSWOT_FIVEFOLD_CONFIG")
    path = explicit or work_root() / "frozen_v15/configuration/calibration_v15.json"
    if not path.is_file():
        raise ValueError("Provide --config for workflows that load the scientific model.")
    return path


def load_configuration():
    return json.loads(configuration_path().read_text(encoding="utf-8-sig"))
