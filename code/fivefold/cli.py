"""Portable entry points for standard WSE-stratified fivefold CV.

All inputs and generated results stay outside this public source repository.
Help and input validation use only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
WORKFLOW = SOURCE / "workflow"
REPOSITORY = SOURCE.parents[1]
REFERENCE = SOURCE / "frozen_v15/configuration/calibration_v15.json"
METHODS = ["full", "beta_fixed_1", "width_train_only", "no_initial_width_guard",
           "no_width_growth_guard", "monotonic_only", "no_hampel", "no_monotonic",
           "fixed_reference_n", "n_only_fixed_geometry", "empirical_rating_curve",
           "observed_fixed", "corrected_fixed"]
COMMANDS = {
    "run": ("run_fivefold.py", "Fit the unchanged V15 model on five disjoint validation folds"),
    "summarize": ("summarize_fivefold.py", "Summarize saved OOF predictions; legacy comparison is opt-in"),
    "compare-products": ("compare_products_fivefold.py", "Compare saved OOF predictions with explicitly supplied SoS records"),
    "error-diagnostics": ("error_diagnostics_fivefold.py", "Diagnose saved residuals and parameter estimates without refitting"),
    "station-analysis": ("station_level_analysis.py", "Summarize per-gauge OOF skill on equal gauge weights"),
    "drivers": ("station_metric_drivers.py", "Explore between-gauge associations without causal attribution"),
    "plot": ("plot_fivefold.py", "Draw diagnostic plots from local aggregate score tables"),
    "audit-assignments": ("audit_all_assignments.py", "Audit input eligibility and fold assignments without fitting"),
    "check-equivalence": ("check_computation_equivalence.py", "Audit numerical preprocessing/fit reuse against the original code"),
    "check-summary": ("check_summary_logic.py", "Run the original small in-memory metric-definition checks"),
    "audit-results": ("final_independent_audit.py", "Audit the complete saved manuscript cohort using standard-library CSV tools"),
    "build-readout": ("build_readout.py", "Build the manuscript-cohort note, including its explicit legacy comparison"),
    "build-station-readout": ("build_station_readout.py", "Build the saved-cohort gauge note and local readout assets"),
    "build-station-inline": ("build_station_inline.py", "Build bounded aggregate chart data in the external work root"),
    "inspect": ("inspect_primary.py", "Independently recalculate primary metrics from saved predictions"),
    "finalize": ("finalize_outputs.py", "Check the complete manuscript-cohort result/readout chain"),
    "freeze-sources": ("freeze_sources.py", "Hash explicit inputs and verify the reused scientific core"),
    "review-documents": ("review_cv_migration.py", "Inspect explicitly supplied DOCX files; documents remain external"),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        q = sub.add_parser(name, help=help_text, description=help_text)
        q.add_argument("--root", type=Path, required=True,
                       help="External calculation/result work directory; never a directory inside this repository")
        if name in {"run", "audit-assignments", "check-equivalence", "freeze-sources"}:
            q.add_argument("--config", type=Path, required=True,
                           help="Explicit JSON settings; relative input paths resolve against the current working directory")
        if name == "run":
            q.add_argument("--workers", type=int, default=20)
            q.add_argument("--methods", nargs="+", choices=METHODS)
            q.add_argument("--pilot-station")
            q.add_argument("--pilot-fold", type=int, choices=range(5))
            q.add_argument("--aggregate-only", action="store_true")
        elif name == "summarize":
            q.add_argument("--predictions", type=Path)
            q.add_argument("--allow-incomplete", action="store_true",
                           help="Smoke/subset runs only; outputs are labelled incomplete")
            q.add_argument("--legacy-predictions", type=Path)
            q.add_argument("--legacy-local-root", type=Path,
                           help="Optional old local summary directory containing QA.json")
            q.add_argument("--legacy-controls-root", type=Path,
                           help="Optional old fixed-control project; used only with an explicit legacy comparison")
        elif name == "compare-products":
            q.add_argument("--predictions", type=Path)
            q.add_argument("--station-audit", type=Path, required=True)
            q.add_argument("--sos-raw-dir", type=Path, required=True)
            q.add_argument("--legacy-native-dir", type=Path,
                           help="Optional already-computed broad native-product summaries")
        elif name == "audit-assignments":
            q.add_argument("--legacy-qc-summary", type=Path)
            q.add_argument("--legacy-qc-flags", type=Path)
        elif name == "plot":
            q.add_argument("--tables", type=Path)
            q.add_argument("--output", type=Path)
            q.add_argument("--dpi", type=int, default=450)
        elif name == "review-documents":
            q.add_argument("--documents", type=Path, nargs="+", required=True)
    return p


def external_root(path):
    result = path.expanduser().resolve()
    if result == REPOSITORY or REPOSITORY in result.parents:
        raise ValueError("Choose --root outside the repository so private inputs/results cannot be created in Git.")
    if result.exists() and not result.is_dir():
        raise ValueError("--root must be a directory: " + str(result))
    return result


def require(files=(), directories=()):
    missing = [str(p) for p in files if not Path(p).is_file()]
    missing += [str(p) for p in directories if not Path(p).is_dir()]
    if missing:
        raise ValueError("Missing external inputs (not distributed in this repository):\n  " + "\n  ".join(missing))


def config_values(path, root, workers=None):
    require([path])
    supplied = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(supplied, dict):
        raise ValueError("The configuration must be a JSON object.")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    unknown = set(supplied) - set(reference)
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
    relocatable = {"data_dir", "metadata_file", "output_dir", "basin_shapefile", "workers"}
    changed = [key for key, value in supplied.items() if key not in relocatable and value != reference[key]]
    if changed:
        raise ValueError("This manuscript workflow retains the pinned V15 scientific settings. Changed keys: " + ", ".join(changed))
    config = dict(reference, **supplied)
    for key in ["data_dir", "metadata_file", "basin_shapefile"]:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Configuration requires a nonempty path: " + key)
        config[key] = str(Path(value).expanduser().resolve())
    config["output_dir"] = str(root)
    if workers is not None:
        if workers < 1:
            raise ValueError("--workers must be at least 1.")
        config["workers"] = workers
    require([config["metadata_file"], config["basin_shapefile"]], [config["data_dir"]])
    if not any(Path(config["data_dir"]).glob("*.csv")):
        raise ValueError("The fused station input directory contains no CSV files.")
    return config


def install_config(root, config):
    target = root / "frozen_v15/configuration/calibration_v15.json"
    if target.is_file():
        old = json.loads(target.read_text(encoding="utf-8-sig"))
        old["output_dir"] = str(root)
        different = [key for key in config if key != "workers" and old.get(key) != config[key]]
        if different:
            raise ValueError("Existing work root has a different configuration; choose a new --root. Keys: " + ", ".join(different))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return target


def prepare(a):
    root = external_root(a.root)
    env = os.environ.copy()
    # Ignore inherited adapters from another run; every invocation is explicit.
    for key in list(env):
        if key.startswith("CSWOT_"):
            env.pop(key)
    env.update({"CSWOT_FIVEFOLD_ROOT": str(root), "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(root / ".matplotlib")})
    command = a.command
    config = config_values(a.config, root, getattr(a, "workers", None)) if hasattr(a, "config") else None
    paths = []
    directories = []
    args = []
    predictions = root / "private/validation_predictions_RESTRICTED.csv"
    params = root / "private/parameter_estimates_RESTRICTED.csv"
    if command == "run":
        args += ["--workers", str(a.workers)]
        if a.methods:
            args += ["--methods", *a.methods]
        if a.pilot_station:
            args += ["--pilot-station", a.pilot_station]
        if a.pilot_fold is not None:
            args += ["--pilot-fold", str(a.pilot_fold)]
        if a.aggregate_only:
            directories += [root / "private/checkpoints"]
            args += ["--aggregate-only"]
    elif command == "summarize":
        source = (a.predictions or predictions).resolve()
        paths += [source]
        args += ["--predictions", str(source), "--output", str(root / "tables")]
        if a.allow_incomplete:
            args += ["--allow-incomplete"]
        if a.legacy_controls_root and not (a.legacy_predictions or a.legacy_local_root):
            raise ValueError("--legacy-controls-root requires --legacy-predictions or --legacy-local-root.")
        if a.legacy_predictions:
            paths += [a.legacy_predictions]
            args += ["--old-predictions", str(a.legacy_predictions.resolve())]
        if a.legacy_local_root:
            paths += [a.legacy_local_root / "QA.json"]
            env["CSWOT_LEGACY_LOCAL"] = str(a.legacy_local_root.resolve())
        if a.legacy_controls_root:
            paths += [a.legacy_controls_root / "private/new_controls_validation_predictions_RESTRICTED.csv",
                      a.legacy_controls_root / "tables/fixed_parameter_configurations.csv"]
            env["CSWOT_LEGACY_CONTROLS"] = str(a.legacy_controls_root.resolve())
        if not (a.legacy_predictions or a.legacy_local_root):
            args += ["--skip-old"]
    elif command == "compare-products":
        source = (a.predictions or predictions).resolve()
        paths += [source, a.station_audit]
        directories += [a.sos_raw_dir]
        args += ["--root", str(root), "--predictions", str(source), "--station-audit", str(a.station_audit.resolve()),
                 "--sos-raw-dir", str(a.sos_raw_dir.resolve())]
        if a.legacy_native_dir:
            directories += [a.legacy_native_dir]
            args += ["--old-native-dir", str(a.legacy_native_dir.resolve())]
        else:
            args += ["--skip-native-copy"]
    elif command == "error-diagnostics":
        paths += [predictions, params]
        args += ["--root", str(root)]
    elif command == "station-analysis":
        paths += [root / "tables/station_unique_observation_metrics.csv"]
    elif command == "drivers":
        paths += [predictions, params, root / "checks/all_station_assignment_eligibility.csv",
                  root / "tables/parameter_boundary_by_gauge.csv"]
    elif command == "plot":
        tables = (a.tables or root / "tables").resolve()
        output = (a.output or root / "figures").resolve()
        external_root(output)
        paths += [tables / "cross_gauge_summary.csv", tables / "cross_gauge_metrics_by_split.csv"]
        if a.dpi < 1:
            raise ValueError("--dpi must be positive.")
        args += ["--tables", str(tables), "--output", str(output), "--dpi", str(a.dpi)]
    elif command == "audit-assignments":
        if bool(a.legacy_qc_summary) != bool(a.legacy_qc_flags):
            raise ValueError("Provide both --legacy-qc-summary and --legacy-qc-flags, or omit both.")
        if a.legacy_qc_summary:
            paths += [a.legacy_qc_summary, a.legacy_qc_flags]
            env["CSWOT_LEGACY_QC_SUMMARY"] = str(a.legacy_qc_summary.resolve())
            env["CSWOT_LEGACY_QC_FLAGS"] = str(a.legacy_qc_flags.resolve())
    elif command == "check-equivalence":
        # This original audit uses the recorded manuscript-cohort station.
        paths += [Path(config["data_dir"]) / "401050.csv"]
    elif command == "audit-results":
        paths += [root / "private/expected_fold_assignments_RESTRICTED.csv",
                  root / "private/split_assignments_RESTRICTED.csv", predictions, params,
                  root / "tables/parameter_estimates.csv", root / "tables/station_eligibility.csv",
                  root / "checks/source_manifest.json",
                  root / "checks/run_status.json", root / "frozen_v15/configuration/calibration_v15.json"]
    elif command == "build-readout":
        paths += [root / ("tables/" + name + ".csv") for name in
                  ["cross_gauge_summary", "cross_gauge_metrics_by_split", "unique_observation_and_stacked_metrics",
                   "station_unique_observation_summary", "extreme_residual_summary", "old_new_intersection_summary"]]
        paths += [root / "checks/final_independent_audit.json"]
    elif command == "build-station-readout":
        paths += [root / ("tables/" + name + ".csv") for name in
                  ["station_analysis_summary", "products_OOF_gauge_summary", "station_analysis_correlations"]]
    elif command == "build-station-inline":
        paths += [root / "tables/station_analysis_gauge_metrics.csv"]
    elif command == "inspect":
        paths += [predictions]
    elif command == "finalize":
        paths += [root / name for name in ["checks/run_status.json", "checks/all_assignment_audit.json",
                  "checks/computation_equivalence.json", "checks/final_independent_audit.json",
                  "checks/products_final_QA.json", "tables/summary.json", "tables/error_diagnostics_method.json",
                  "checks/primary_independent_metrics.json", "tables/cross_gauge_summary.csv", "index.html"]]
    elif command == "review-documents":
        paths += a.documents
        env["CSWOT_REVIEW_DOCUMENTS"] = json.dumps([str(p.resolve()) for p in a.documents])
    require(paths, directories)
    # No directories or files have been created before all input checks pass.
    # Check an existing configuration before creating any additional outputs.
    if config is not None:
        env["CSWOT_FIVEFOLD_CONFIG"] = str(install_config(root, config))
    for name in ["checks", "tables", "figures", "private"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root, env, args


def job(script, arguments, env):
    return subprocess.run([sys.executable, "-B", str(WORKFLOW / script), *arguments],
                          env=env, check=False).returncode


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        root, env, args = prepare(a)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    if a.command == "run":
        code = job("freeze_sources.py", [], env)
        if code:
            return code
    return job(COMMANDS[a.command][0], args, env)


if __name__ == "__main__":
    raise SystemExit(main())
