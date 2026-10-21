#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SWOT-5: global satellite QC, bounded WSE-width geometry, ten paired splits,
70% initial temporal calibration, and no broad geographic parameter transfer.
Run run_calculation.cmd; default output is D:/SWOT-5/results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import quality_geometry as quality
import run_review_revision_experiments as pipeline
import swot_beta_wse0_width0_model as core


DEFAULT_DATA_DIR = Path(
    r"D:\SWOT5.2\swot_data_download_fusion\outputs\fused\station_overpasses_datum_adjusted"
)
DEFAULT_METADATA_FILE = Path(
    r"D:\SWOT5.2\inputs\Find_N.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\SWOT-5\results"
)
DEFAULT_BASIN_SHAPEFILE = Path(
    r"D:\SWOT5.2\inputs\BasinSHP\Class2_hydrounits.shp"
)

PRIMARY_METHOD = "full"
ABLATION_METHODS = (
    "beta_fixed_1",
    "width_train_only",
    "no_initial_width_guard",
    "no_width_growth_guard",
    "monotonic_only",
    "no_hampel",
    "no_monotonic",
    "fixed_reference_n",
    "n_only_fixed_geometry",
    "empirical_rating_curve",
)
METHOD_LABELS = {
    "full": "Global Hampel + bounded calibration width curve applied to validation",
    "width_train_only": "Bounded width correction in calibration only; observed validation width",
    "no_initial_width_guard": "No initial-width neighbor guard",
    "no_width_growth_guard": "No width-growth guard",
    "monotonic_only": "Monotonic curve without initial or growth guards",
    "beta_fixed_1": "No beta optimization: beta fixed at 1",
    "no_hampel": "No Hampel",
    "no_monotonic": "No monotonic-width constraint",
    "fixed_reference_n": "Fixed reference Manning n",
    "n_only_fixed_geometry": "Optimized n only; beta=1 and fixed simple bed",
    "empirical_rating_curve": "Empirical WSE-Q rating curve",
}

BETA_RELATIVE_MULTIPLIERS = (
    0.50, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50,
)
BETA_ABSOLUTE_GRID = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00)


@dataclass
class BetaRevisionConfig(pipeline.RevisionConfig):
    """扩展现有审稿实验配置；默认进行 10 次重复 WSE 分层验证。"""

    repeat_random_grouping: bool = True
    beta_min: float = core.BETA_MIN_DEFAULT
    beta_max: float = core.BETA_MAX_DEFAULT
    beta_prior_weight: float = core.BETA_PRIOR_WEIGHT_DEFAULT
    width0_ratio_min: float = core.WIDTH0_RATIO_MIN_DEFAULT
    width0_ratio_max: float = core.WIDTH0_RATIO_MAX_DEFAULT
    run_beta_sensitivity: bool = True
    run_sample_size_analysis: bool = True
    sample_size_targets: Tuple[int, ...] = (5, 10, 20)
    width_curve_scope: str = "calibration_only_for_validation"
    spatial_folds: int = 5
    run_spatial_cv: bool = False
    temporal_train_fraction: float = 0.70
    validation_multistart_count: int = 3
    hampel_half_window: int = 5
    hampel_min_points: int = 3
    hampel_sigma: float = 3.0
    wse_scale_floor_m: float = 0.02
    width_scale_floor_m: float = 1.0
    slope_log_scale_floor: float = 0.05
    initial_neighbor_count: int = 5
    initial_neighbor_radius_m: float = 1.0
    initial_min_neighbors: int = 2
    initial_width_ratio: float = 3.0
    width_max_growth_m_per_m: float = 100.0
    width_max_relative_growth_per_m: float = 2.0


def make_core_config(
    cfg: BetaRevisionConfig,
    enable_hampel: bool,
    monotonic: bool,
) -> core.Config:
    return core.Config(
        data_dir=cfg.data_dir,
        metadata_file=cfg.metadata_file,
        output_dir=cfg.output_dir,
        repeat_random_grouping=cfg.repeat_random_grouping,
        repeat_count=cfg.repeat_count,
        enable_hampel=enable_hampel,
        run_reconstruction=False,
        run_wse_cv=False,
        run_temporal_cv=False,
        run_spatial_cv=False,
        temporal_folds=cfg.temporal_folds,
        spatial_folds=2,
        workers=1,
        use_parallel=False,
        max_stations=None,
        fast=cfg.fast,
        objective=cfg.objective,
        seed=cfg.seed,
        aggregate_nodes_by_overpass=cfg.aggregate_nodes_by_overpass,
        use_global_station_monotonic_width=monotonic,
        optimizer_maxiter=cfg.optimizer_maxiter,
        optimizer_popsize=cfg.optimizer_popsize,
        optimizer_tol=cfg.optimizer_tol,
        local_optimizer_maxiter=cfg.local_optimizer_maxiter,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
        beta_prior_weight=cfg.beta_prior_weight,
        width0_ratio_min=cfg.width0_ratio_min,
        width0_ratio_max=cfg.width0_ratio_max,
    )


def make_frame_versions(base):
    pairs, diag = make_split_frame_versions(base, base['obs_id'].tolist(), [])
    return {method:pair[0] for method,pair in pairs.items()}, diag

def make_split_frame_versions(base, train_ids, test_ids):
    raw_train = pipeline.select_rows(base, train_ids)
    raw_test = pipeline.select_rows(base, test_ids)
    pairs = {}
    for method in (PRIMARY_METHOD, *ABLATION_METHODS):
        train, test = quality.retain(raw_train,method), quality.retain(raw_test,method)
        if method == 'no_monotonic':
            for frame in (train,test):
                frame['width_model'] = frame['width'].astype(float)
                frame['width_curve_scope'] = 'observed_width'
            pairs[method] = train,test
            continue
        model = quality.curve(train,
            initial=method not in {'no_initial_width_guard','monotonic_only'},
            growth=method not in {'no_width_growth_guard','monotonic_only'})
        pairs[method] = (
            quality.apply_curve(train,model,training=True),
            quality.apply_curve(test,model,training=False,use_validation_curve=method!='width_train_only'))
    primary = pairs[PRIMARY_METHOD][0]
    diag = {'global_qc_retained_count':len(primary),
            'global_qc_excluded_count':len(raw_train)-len(primary),
            'global_monotonic_anchor_count':primary['wse_model'].nunique(),
            'width_curve_training_count':len(primary)}
    raw_curve=primary.groupby('wse_model',sort=True).width.median().to_numpy(float)
    adjustment=(primary['width_model']-primary['width']).to_numpy(float)
    diag.update({
        'raw_width_wse_decrease_count':int(np.sum(np.diff(raw_curve)<0)),
        'model_width_wse_decrease_count':int(np.sum(np.diff(primary.groupby('wse_model').width_model.median()) < -1e-9)),
        'global_width_adjustment_median_abs':float(np.median(np.abs(adjustment))) if len(adjustment) else np.nan,
        'global_width_adjustment_max_abs':float(np.max(np.abs(adjustment))) if len(adjustment) else np.nan,
    })
    return pairs,diag

def _fit_method_once(
    method: str,
    frame: pd.DataFrame,
    cfg: BetaRevisionConfig,
    seed: int,
    n_reference: float,
) -> Optional[Dict[str, Any]]:
    if method in {"full", "no_hampel", "no_monotonic", "width_train_only", "no_initial_width_guard", "no_width_growth_guard", "monotonic_only"}:
        hampel = method != "no_hampel"
        monotonic = method != "no_monotonic"
        return core.fit_parameters(
            frame,
            make_core_config(cfg, hampel, monotonic),
            seed,
            parameter_source=(
                "station_joint_n_beta_WSE0_width0"
                if method == "full"
                else f"station_joint_n_beta_WSE0_width0_{method}"
            ),
        )
    if method == "beta_fixed_1":
        return core.fit_parameters(
            frame,
            make_core_config(cfg, True, True),
            seed,
            beta_fixed_value=1.0,
            parameter_source="station_joint_n_WSE0_width0_beta_fixed_1",
        )
    if method == "fixed_reference_n":
        return core.fit_parameters(
            frame,
            make_core_config(cfg, True, True),
            seed,
            n_fixed_value=n_reference,
            parameter_source="metadata_or_global_n_joint_beta_WSE0_width0",
        )
    if method == "n_only_fixed_geometry":
        prepared, _ = core.prepare_training(frame, enable_hampel=True)
        if prepared is None:
            return None
        anchor_wse, _ = core.collapse_wse_width(
            prepared["wse_model"], prepared["width_model"]
        )
        depth0 = max(0.5, float(np.ptp(anchor_wse)))
        return core.fit_parameters(
            frame,
            make_core_config(cfg, True, True),
            seed,
            beta_fixed_value=1.0,
            depth0_fixed_value=depth0,
            width0_fixed_value=0.0,
            parameter_source="optimized_n_fixed_beta_and_simple_triangular_bed",
        )
    if method == "empirical_rating_curve":
        return pipeline.fit_rating_curve(frame, cfg, seed)
    raise ValueError(f"Unknown method: {method}")


def fit_method(method, frame, cfg, seed, n_reference):
    candidates = []
    for start in range(cfg.validation_multistart_count):
        fit = _fit_method_once(method,frame,cfg,pipeline.stable_seed(seed,'matched_optimizer_start',start,base=cfg.seed),n_reference)
        if fit is not None and np.isfinite(fit.get('objective_loss',np.nan)) and fit['objective_loss'] < 1e11:
            fit['selected_validation_start'] = start
            candidates.append(fit)
    if not candidates:return None
    fit = min(candidates,key=lambda f:f['objective_loss'])
    if fit.get('model_kind')!='rating_curve':
        fit.update(quality.curve_metadata(frame))
    fit['validation_optimizer_start_count'] = cfg.validation_multistart_count
    return fit


def parameter_row(
    fit: Dict[str, Any],
    station_id: str,
    method: str,
    validation_scheme: str,
    fold_id: int,
    repeat_id: int,
    train_count: int,
    validation_count: int,
) -> Dict[str, Any]:
    excluded = {"anchor_wse", "anchor_width", "train_metrics", "qc"}
    row = {key: value for key, value in fit.items() if key not in excluded}
    row.update({
        "station_id": station_id,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "validation_scheme": validation_scheme,
        "fold_id": fold_id,
        "repeat_id": repeat_id,
        "train_count_before_qc": int(train_count),
        "validation_count": int(validation_count),
        "optimization_unit": "station",
        "width_rule": (
            "observed_nonnegative"
            if method == "no_monotonic"
            else "not_used"
            if method == "empirical_rating_curve"
            else (
                "calibration_only_station_isotonic"
                if fit.get("width_curve_scope") == "calibration_only"
                else "saved_calibration_width_curve"
            )
        ),
        **fit.get("qc", {}),
    })
    row.update({
        f"train_{key}": value
        for key, value in fit.get("train_metrics", {}).items()
    })
    return row


def fit_primary_multistart(
    frame: pd.DataFrame,
    station_id: str,
    cfg: BetaRevisionConfig,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    starts = cfg.multistart_count if cfg.run_multistart else 1
    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    core_cfg = make_core_config(cfg, enable_hampel=True, monotonic=True)
    for start_id in range(max(1, starts)):
        fit = core.fit_parameters(
            frame,
            core_cfg,
            pipeline.stable_seed(
                station_id, "beta_bed_multistart", start_id, base=cfg.seed
            ),
        )
        if fit is None:
            rows.append({
                "station_id": station_id,
                "start_id": start_id,
                "fit_success": False,
            })
            continue
        candidates.append((start_id, fit))
        rows.append({
            "station_id": station_id,
            "start_id": start_id,
            "fit_success": True,
            "objective_loss": fit["objective_loss"],
            "n": fit["n"],
            "beta": fit["beta"],
            "WSE0": fit["WSE0"],
            "depth0": fit["depth0"],
            "width0": fit["width0"],
            "width0_to_min_width_ratio": fit[
                "width0_to_min_width_ratio"
            ],
            "A0": fit["A0"],
            "optimizer_success": fit["optimizer_success"],
            "optimizer_nfev": fit["optimizer_nfev"],
            "optimizer_nit": fit["optimizer_nit"],
            "fit_accepted": fit.get("fit_accepted", False),
            "global_optimizer_success": fit.get(
                "global_optimizer_success", False
            ),
            "local_optimizer_success": fit.get(
                "local_optimizer_success", False
            ),
        })
    if not candidates:
        return None, rows
    best_start, best = min(
        candidates, key=lambda pair: float(pair[1]["objective_loss"])
    )
    for row in rows:
        row["selected_best"] = bool(
            row.get("fit_success") and row["start_id"] == best_start
        )
    return best, rows


def multistart_summary(multistart: pd.DataFrame) -> pd.DataFrame:
    if multistart.empty:
        return pd.DataFrame()
    valid = multistart[
        multistart["fit_success"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    rows: List[Dict[str, Any]] = []
    for station_id, group in valid.groupby("station_id"):
        row: Dict[str, Any] = {
            "station_id": station_id,
            "successful_start_count": int(len(group)),
            "all_optimizer_success": bool(
                group["optimizer_success"].fillna(False).all()
            ),
            "all_fit_accepted": bool(
                group["fit_accepted"].fillna(False).all()
            ),
        }
        for column in (
            "objective_loss",
            "n",
            "beta",
            "WSE0",
            "depth0",
            "width0",
            "width0_to_min_width_ratio",
            "A0",
        ):
            values = pd.to_numeric(
                group.get(column), errors="coerce"
            ).dropna()
            row[f"{column}_min"] = (
                float(values.min()) if len(values) else np.nan
            )
            row[f"{column}_max"] = (
                float(values.max()) if len(values) else np.nan
            )
            row[f"{column}_median"] = (
                float(values.median()) if len(values) else np.nan
            )
            denom = abs(float(values.mean())) if len(values) else np.nan
            row[f"{column}_cv"] = (
                float(values.std(ddof=0) / denom)
                if len(values) > 1 and denom > 0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


_ORIGINAL_PERFORMANCE_DRIVER_TABLES = pipeline.performance_driver_tables


def performance_driver_tables(
    station_metrics: pd.DataFrame,
    statistics: pd.DataFrame,
    parameters: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: BetaRevisionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    drivers, correlations = _ORIGINAL_PERFORMANCE_DRIVER_TABLES(
        station_metrics, statistics, parameters, predictions, cfg
    )
    params = parameters[
        (parameters["method"] == PRIMARY_METHOD)
        & (parameters["validation_scheme"] == "station_reconstruction")
    ].drop_duplicates("station_id")
    extra_columns = [
        column
        for column in (
            "station_id",
            "beta",
            "width0_to_min_width_ratio",
            "beta_at_lower_bound",
            "beta_at_upper_bound",
            "width0_at_lower_bound",
            "width0_at_upper_bound",
        )
        if column in params and column not in drivers
    ]
    if "station_id" not in extra_columns and extra_columns:
        extra_columns.insert(0, "station_id")
    if len(extra_columns) > 1:
        drivers = drivers.merge(
            params[extra_columns], on="station_id", how="left"
        )

    new_rows = []
    for target_metric in (
        "R2", "NSE", "RMSE", "MAE", "peak_error_percent"
    ):
        if target_metric not in drivers:
            continue
        for feature in (
            "beta", "width0", "width0_to_min_width_ratio", "A0"
        ):
            if feature not in drivers:
                continue
            x = pd.to_numeric(drivers[feature], errors="coerce")
            y = pd.to_numeric(drivers[target_metric], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if (
                mask.sum() < 10
                or x[mask].nunique() < 2
                or y[mask].nunique() < 2
            ):
                continue
            rho, p_value = spearmanr(x[mask], y[mask])
            new_rows.append({
                "target_metric": target_metric,
                "feature": feature,
                "station_count": int(mask.sum()),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            })
    if new_rows:
        correlations = pd.concat(
            [correlations, pd.DataFrame(new_rows)],
            ignore_index=True,
        ).drop_duplicates(
            ["target_metric", "feature"], keep="last"
        )
    return drivers, correlations


_ORIGINAL_LOAD_STATION = pipeline.load_station_with_audit

def load_station_with_global_qc(path, aggregate_nodes):
    base,audit,raw,overpass = _ORIGINAL_LOAD_STATION(path,aggregate_nodes)
    base = quality.annotate(base)
    qc_cols = [c for c in base if c.startswith('global_')]
    overpass = overpass.merge(base[['obs_id']+qc_cols],on='obs_id',how='left',validate='one_to_one')
    overpass['primary_training_qc_pass'] = overpass['global_qc_pass'].fillna(False)
    for old,new in [('hampel_wse_candidate','global_hampel_wse'),('hampel_width_candidate','global_hampel_width'),('hampel_slope_candidate','global_hampel_log_slope')]:
        overpass[old]=overpass[new].fillna(False)
    audit['global_qc_retained_count'] = int(base.global_qc_pass.sum())
    audit['global_qc_excluded_count'] = int((~base.global_qc_pass).sum())
    return base,audit,raw,overpass


_ORIGINAL_PROCESS_STATION = pipeline.process_station


def process_station(
    path_text: str,
    meta: Dict[str, Any],
    global_reference_n: float,
    cfg: BetaRevisionConfig,
) -> Dict[str, Any]:
    """
    Windows 多进程安全包装。

    子进程导入本模块后会安装相同覆盖，再调用成熟的站点实验编排。
    """
    _install_overrides()
    quality.configure(cfg)
    return _ORIGINAL_PROCESS_STATION(
        path_text, meta, global_reference_n, cfg
    )


def _install_overrides() -> None:
    """把成熟流水线的可替换水力接口指向本版本。"""
    pipeline.load_station_with_audit = load_station_with_global_qc
    pipeline.core = core
    pipeline.PRIMARY_METHOD = PRIMARY_METHOD
    pipeline.ABLATION_METHODS = ABLATION_METHODS
    pipeline.METHOD_LABELS = METHOD_LABELS
    pipeline.make_core_config = make_core_config
    pipeline.make_frame_versions = make_frame_versions
    pipeline.make_split_frame_versions = make_split_frame_versions
    pipeline.fit_method = fit_method
    pipeline.parameter_row = parameter_row
    pipeline.fit_primary_multistart = fit_primary_multistart
    pipeline.multistart_summary = multistart_summary
    pipeline.performance_driver_tables = performance_driver_tables
    pipeline.process_station = process_station


def parse_args() -> BetaRevisionConfig:
    parser = argparse.ArgumentParser(
        description="SWOT beta + WSE0/width0 station experiments"
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SWOT_DATA_DIR", str(DEFAULT_DATA_DIR)),
    )
    parser.add_argument(
        "--metadata-file",
        default=os.environ.get(
            "SWOT_METADATA_FILE", str(DEFAULT_METADATA_FILE)
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SWOT_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument(
        "--repeat-random-grouping",
        dest="repeat_random_grouping",
        action="store_true",
        help="启用重复 WSE 分层随机分组（默认启用）。",
    )
    parser.add_argument(
        "--no-repeat-random-grouping",
        dest="repeat_random_grouping",
        action="store_false",
    )
    parser.set_defaults(repeat_random_grouping=True)
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=10,
        help="WSE 分层随机分组重复次数（默认 10）；关闭时固定为 1。",
    )
    parser.add_argument("--ablation-repeat-count", type=int, default=10)
    parser.add_argument('--temporal-train-fraction', type=float, default=0.70)
    parser.add_argument('--validation-multistart-count', type=int, default=3)
    for name, default in quality.asdict(quality.Rules()).items():
        parser.add_argument('--'+name.replace('_','-'),type=type(default),default=default)
    parser.add_argument("--spatial-folds", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--temporal-folds", type=int, default=3)
    parser.add_argument("--multistart-count", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument(
        "--no-ablation-reconstruction", action="store_true"
    )
    parser.add_argument("--no-n-sensitivity", action="store_true")
    parser.add_argument("--no-beta-sensitivity", action="store_true")
    parser.add_argument(
        "--no-sample-size-analysis",
        action="store_true",
        help="关闭固定验证集的训练数据量学习曲线。",
    )
    parser.add_argument(
        "--sample-size-targets",
        default="5,10,20",
        help="学习曲线的训练样本数，逗号分隔；默认 5,10,20。",
    )
    parser.add_argument("--no-multistart", action="store_true")
    parser.add_argument(
        "--beta-min", type=float, default=core.BETA_MIN_DEFAULT
    )
    parser.add_argument(
        "--beta-max", type=float, default=core.BETA_MAX_DEFAULT
    )
    parser.add_argument(
        "--beta-prior-weight",
        type=float,
        default=core.BETA_PRIOR_WEIGHT_DEFAULT,
    )
    parser.add_argument("--width0-ratio-min", type=float, default=0.0)
    parser.add_argument("--width0-ratio-max", type=float, default=1.0)
    parser.add_argument("--optimizer-maxiter", type=int, default=60)
    parser.add_argument("--optimizer-popsize", type=int, default=8)
    parser.add_argument("--optimizer-tol", type=float, default=1e-3)
    parser.add_argument(
        "--local-optimizer-maxiter", type=int, default=400
    )
    parser.add_argument(
        "--basin-shapefile",
        default=os.environ.get(
            "SWOT_BASIN_SHAPEFILE", str(DEFAULT_BASIN_SHAPEFILE)
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("old_composite", "normalized_mse", "log_mse"),
        default="old_composite",
    )
    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    if not args.output_dir.isascii():
        parser.error('Output path must contain ASCII characters only.')
    if not 0 < args.temporal_train_fraction < 1:
        parser.error('--temporal-train-fraction must be between 0 and 1.')
    for name in quality.asdict(quality.Rules()):
        if getattr(args,name) <= 0:parser.error(name+' must be positive.')
    if args.initial_width_ratio <= 1:parser.error('--initial-width-ratio must exceed 1.')
    if args.hampel_min_points > 2*args.hampel_half_window+1:parser.error('Hampel minimum exceeds window size.')
    if args.initial_min_neighbors > args.initial_neighbor_count:parser.error('Insufficient configured initial neighbors.')

    if not (0 < args.beta_min < args.beta_max):
        parser.error("需要满足 0 < --beta-min < --beta-max")
    if args.beta_prior_weight < 0:
        parser.error("--beta-prior-weight 不能为负")
    if not (
        0 <= args.width0_ratio_min
        < args.width0_ratio_max
        <= 1
    ):
        parser.error(
            "需要满足 0 <= width0-ratio-min < "
            "width0-ratio-max <= 1"
        )
    try:
        sample_size_targets = tuple(sorted({
            int(value.strip())
            for value in str(args.sample_size_targets).split(",")
            if value.strip()
        }))
    except ValueError:
        parser.error("--sample-size-targets 必须是逗号分隔的整数")
    if any(value < core.MIN_TRAIN_RECORDS for value in sample_size_targets):
        parser.error(
            f"样本量目标不能小于最低率定记录数 {core.MIN_TRAIN_RECORDS}"
        )
    effective_repeat_count = (
        max(1, int(args.repeat_count))
        if args.repeat_random_grouping
        else 1
    )
    return BetaRevisionConfig(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        repeat_count=effective_repeat_count,
        ablation_repeat_count=max(
            1,
            min(int(args.ablation_repeat_count), effective_repeat_count),
        ),
        temporal_folds=max(1, int(args.temporal_folds)),
        multistart_count=max(1, int(args.multistart_count)),
        bootstrap_iterations=max(200, int(args.bootstrap_iterations)),
        workers=max(1, int(args.workers)),
        use_parallel=not args.serial,
        max_stations=args.max_stations,
        fast=bool(args.fast),
        objective=args.objective,
        run_temporal=not args.no_temporal,
        run_ablation=not args.no_ablation,
        run_ablation_reconstruction=(
            not args.no_ablation_reconstruction
        ),
        run_n_sensitivity=not args.no_n_sensitivity,
        run_multistart=not args.no_multistart,
        optimizer_maxiter=max(1, int(args.optimizer_maxiter)),
        optimizer_popsize=max(5, int(args.optimizer_popsize)),
        optimizer_tol=max(float(args.optimizer_tol), 1e-10),
        local_optimizer_maxiter=max(
            50, int(args.local_optimizer_maxiter)
        ),
        basin_shapefile=str(args.basin_shapefile),
        run_spatial_cv=False,
        temporal_train_fraction=args.temporal_train_fraction,
        validation_multistart_count=max(1,args.validation_multistart_count),
        **{name:getattr(args,name) for name in quality.asdict(quality.Rules())},
        spatial_folds=max(2, int(args.spatial_folds)),
        repeat_random_grouping=bool(
            args.repeat_random_grouping
        ),
        beta_min=float(args.beta_min),
        beta_max=float(args.beta_max),
        beta_prior_weight=float(args.beta_prior_weight),
        width0_ratio_min=float(args.width0_ratio_min),
        width0_ratio_max=float(args.width0_ratio_max),
        run_beta_sensitivity=not args.no_beta_sensitivity,
        run_sample_size_analysis=not args.no_sample_size_analysis,
        sample_size_targets=sample_size_targets,
    )


def _beta_sensitivity_rows(
    predictions: pd.DataFrame,
    cfg: BetaRevisionConfig,
) -> pd.DataFrame:
    data = predictions[
        (predictions["method"] == PRIMARY_METHOD)
        & (
            predictions["validation_scheme"]
            == "station_reconstruction"
        )
    ].copy()
    if data.empty or "beta" not in data:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for station_id, group in data.groupby("station_id"):
        beta_values = pd.to_numeric(
            group["beta"], errors="coerce"
        ).dropna()
        if beta_values.empty:
            continue
        beta_opt = float(beta_values.iloc[0])
        obs = group["Q_observed"].to_numpy(dtype=float)
        for multiplier in BETA_RELATIVE_MULTIPLIERS:
            requested = beta_opt * float(multiplier)
            beta_test = float(
                np.clip(requested, cfg.beta_min, cfg.beta_max)
            )
            sim = core.q_for_beta(group, beta_test)
            rows.append({
                "station_id": station_id,
                "grid_type": "relative_to_optimized",
                "beta_optimized": beta_opt,
                "beta_requested": requested,
                "beta_test": beta_test,
                "beta_multiplier_requested": float(multiplier),
                "beta_multiplier_actual": beta_test / beta_opt,
                "beta_was_clipped": bool(
                    not np.isclose(
                        beta_test / beta_opt,
                        float(multiplier),
                        rtol=1e-10,
                        atol=1e-12,
                    )
                ),
                **core.calculate_metrics(obs, sim),
            })
        for requested in BETA_ABSOLUTE_GRID:
            if requested < cfg.beta_min or requested > cfg.beta_max:
                continue
            sim = core.q_for_beta(group, requested)
            rows.append({
                "station_id": station_id,
                "grid_type": "absolute_grid",
                "beta_optimized": beta_opt,
                "beta_requested": requested,
                "beta_test": requested,
                "beta_multiplier_requested": np.nan,
                "beta_multiplier_actual": requested / beta_opt,
                "beta_was_clipped": False,
                **core.calculate_metrics(obs, sim),
            })
    return pd.DataFrame(rows)


def _beta_sensitivity_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    metrics = pipeline.METRICS
    absolute = data[data["grid_type"] == "absolute_grid"]
    for beta_test, group in absolute.groupby("beta_test"):
        row: Dict[str, Any] = {
            "grid_type": "absolute_grid",
            "beta_test": beta_test,
            "station_count": int(group["station_id"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(
                group.get(metric), errors="coerce"
            )
            suffix = metric
            if metric == "PBIAS":
                values = values.abs()
                suffix = "absPBIAS"
            values = values.dropna()
            row[f"station_median_{suffix}"] = (
                float(values.median()) if len(values) else np.nan
            )
            row[f"station_q25_{suffix}"] = (
                float(values.quantile(0.25)) if len(values) else np.nan
            )
            row[f"station_q75_{suffix}"] = (
                float(values.quantile(0.75)) if len(values) else np.nan
            )
        rows.append(row)

    relative = data[data["grid_type"] == "relative_to_optimized"]
    for multiplier, group in relative.groupby(
        "beta_multiplier_requested"
    ):
        actual = pd.to_numeric(
            group["beta_multiplier_actual"], errors="coerce"
        )
        clipped = group["beta_was_clipped"].astype(
            str
        ).str.lower().isin(["true", "1"])
        row = {
            "grid_type": "relative_multiplier_summary",
            "beta_test": np.nan,
            "beta_multiplier_requested": multiplier,
            "actual_multiplier_median": float(actual.median()),
            "actual_multiplier_q25": float(actual.quantile(0.25)),
            "actual_multiplier_q75": float(actual.quantile(0.75)),
            "clipped_station_fraction": float(clipped.mean()),
            "station_count": int(group["station_id"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(
                group.get(metric), errors="coerce"
            )
            suffix = metric
            if metric == "PBIAS":
                values = values.abs()
                suffix = "absPBIAS"
            values = values.dropna()
            row[f"station_median_{suffix}"] = (
                float(values.median()) if len(values) else np.nan
            )
            row[f"station_q25_{suffix}"] = (
                float(values.quantile(0.25)) if len(values) else np.nan
            )
            row[f"station_q75_{suffix}"] = (
                float(values.quantile(0.75)) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _wse_balanced_nested_order(
    frame: pd.DataFrame,
    seed: int,
    strata_count: int = 5,
) -> List[int]:
    """
    构造 WSE 分层、确定性且可嵌套的抽样顺序。

    每一层内部先按固定随机种子打乱，再轮流从各层取一条。任意前 k
    条都是前 k-1 条的超集，同时尽量覆盖完整 WSE 范围。
    """
    if frame.empty:
        return []
    work = frame.reset_index(drop=True).copy()
    ranks = pd.to_numeric(
        work["wse_model"], errors="coerce"
    ).rank(method="first")
    bins = min(max(1, int(strata_count)), len(work))
    try:
        strata = pd.qcut(
            ranks, q=bins, labels=False, duplicates="drop"
        )
    except ValueError:
        strata = pd.Series(np.zeros(len(work), dtype=int))
    rng = np.random.default_rng(int(seed))
    queues: List[List[int]] = []
    for stratum in sorted(pd.Series(strata).dropna().unique()):
        indices = np.flatnonzero(
            pd.Series(strata).to_numpy() == stratum
        )
        queues.append(rng.permutation(indices).astype(int).tolist())
    if not queues:
        return rng.permutation(len(work)).astype(int).tolist()
    order: List[int] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return order


def sample_size_station_worker(
    path_text: str,
    cfg: BetaRevisionConfig,
) -> Dict[str, Any]:
    """
    一个站点的受控学习曲线。

    验证集固定为基础 repeat 0 的 WSE 分层留出集；训练子集是同一训练
    池的 WSE 均衡嵌套前缀。因此不同样本量的差异不来自验证点改变。
    """
    _install_overrides()
    quality.configure(cfg)
    path = Path(path_text)
    station_id = core.normalize_station_id(path.stem)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        base, _, _, _ = pipeline.load_station_with_audit(
            path, cfg.aggregate_nodes_by_overpass
        )
        primary = base.copy()
        if len(primary) < core.MIN_TRAIN_RECORDS:
            return {
                "rows": rows,
                "failures": [{
                    "station_id": station_id,
                    "requested_train_count": np.nan,
                    "reason": "insufficient_station_records",
                }],
            }
        split = core.wse_stratified_split(
            primary,
            0.70,
            5,
            pipeline.stable_seed(
                station_id, "wse_split", 0, base=cfg.seed
            ),
        )
        if split is None:
            return {
                "rows": rows,
                "failures": [{
                    "station_id": station_id,
                    "requested_train_count": np.nan,
                    "reason": "base_wse_split_failed",
                }],
            }
        train_pool, test_raw = split
        train_pool = quality.retain(train_pool)
        test_raw = quality.retain(test_raw)
        order = _wse_balanced_nested_order(
            train_pool,
            pipeline.stable_seed(
                station_id, "sample_size_order", base=cfg.seed
            ),
        )
        for requested in cfg.sample_size_targets:
            if requested >= len(train_pool):
                continue
            train_raw = train_pool.iloc[
                order[:requested]
            ].copy().reset_index(drop=True)
            fold_versions, _ = make_split_frame_versions(
                base,
                train_raw["obs_id"].tolist(),
                test_raw["obs_id"].tolist(),
            )
            train, test = fold_versions[PRIMARY_METHOD]
            if test.empty:
                failures.append({"station_id":station_id,"requested_train_count":requested,"reason":"no_validation_after_global_qc"})
                continue
            fit = fit_method(
                PRIMARY_METHOD,
                train,
                cfg,
                pipeline.stable_seed(
                    station_id,
                    "sample_size_fit",
                    requested,
                    base=cfg.seed,
                ),
                n_reference=0.035,
            )
            if fit is None:
                failures.append({
                    "station_id": station_id,
                    "requested_train_count": requested,
                    "reason": "fit_failed_after_training_QC",
                })
                continue
            pred = pipeline.decorate_predictions(
                pipeline.predict_method(
                    PRIMARY_METHOD,
                    fit,
                    test,
                    "sample_size_learning_curve",
                    0,
                    0,
                    "validation",
                ),
                base,
            )
            metric = core.calculate_metrics(
                pred["Q_observed"], pred["Q_calculated"]
            )
            rows.append({
                "station_id": station_id,
                "sample_size_label": str(requested),
                "sample_size_order": int(requested),
                "requested_train_count": int(requested),
                "train_pool_count": int(len(train_pool)),
                "train_count_input": int(len(train)),
                "train_count_effective": int(
                    fit.get("qc", {}).get(
                        "train_valid_count", len(train)
                    )
                ),
                "validation_count": int(len(test)),
                "train_wse_min": float(
                    pd.to_numeric(
                        train_raw["wse_model"], errors="coerce"
                    ).min()
                ),
                "train_wse_max": float(
                    pd.to_numeric(
                        train_raw["wse_model"], errors="coerce"
                    ).max()
                ),
                "train_wse_range": float(
                    pd.to_numeric(
                        train_raw["wse_model"], errors="coerce"
                    ).max()
                    - pd.to_numeric(
                        train_raw["wse_model"], errors="coerce"
                    ).min()
                ),
                "pool_wse_range": float(
                    pd.to_numeric(
                        train_pool["wse_model"], errors="coerce"
                    ).max()
                    - pd.to_numeric(
                        train_pool["wse_model"], errors="coerce"
                    ).min()
                ),
                "n": fit["n"],
                "beta": fit["beta"],
                "WSE0": fit["WSE0"],
                "width0": fit["width0"],
                "depth0": fit["depth0"],
                "A0": fit["A0"],
                "n_at_lower_bound": fit["n_at_lower_bound"],
                "n_at_upper_bound": fit["n_at_upper_bound"],
                "beta_at_lower_bound": fit[
                    "beta_at_lower_bound"
                ],
                "beta_at_upper_bound": fit[
                    "beta_at_upper_bound"
                ],
                "width0_at_lower_bound": fit[
                    "width0_at_lower_bound"
                ],
                "width0_at_upper_bound": fit[
                    "width0_at_upper_bound"
                ],
                "objective_loss": fit["objective_loss"],
                **metric,
            })
    except Exception as exc:
        failures.append({
            "station_id": station_id,
            "requested_train_count": np.nan,
            "reason": f"{type(exc).__name__}: {exc}",
        })
    return {"rows": rows, "failures": failures}


def _all_available_sample_size_rows(
    tables_dir: Path,
    target_order: int,
) -> pd.DataFrame:
    fold_path = tables_dir / "validation_metrics_by_fold.csv"
    parameter_path = tables_dir / "parameter_estimates.csv"
    if not fold_path.exists() or not parameter_path.exists():
        return pd.DataFrame()
    fold = pd.read_csv(fold_path, low_memory=False)
    params = pd.read_csv(parameter_path, low_memory=False)
    fold = fold[
        (fold["method"] == PRIMARY_METHOD)
        & (fold["validation_scheme"] == "wse_stratified")
        & (fold["repeat_id"] == 0)
        & (fold["fold_id"] == 0)
    ].copy()
    params = params[
        (params["method"] == PRIMARY_METHOD)
        & (params["validation_scheme"] == "wse_stratified")
        & (params["repeat_id"] == 0)
        & (params["fold_id"] == 0)
    ].drop_duplicates("station_id")
    keep = [
        column
        for column in (
            "station_id",
            "train_valid_count",
            "train_wse_min",
            "train_wse_max",
            "train_wse_range",
            "n",
            "beta",
            "WSE0",
            "width0",
            "depth0",
            "A0",
            "n_at_lower_bound",
            "n_at_upper_bound",
            "beta_at_lower_bound",
            "beta_at_upper_bound",
            "width0_at_lower_bound",
            "width0_at_upper_bound",
            "objective_loss",
        )
        if column in params
    ]
    out = fold.merge(
        params[keep], on="station_id", how="left"
    )
    out["sample_size_label"] = "all_available"
    out["sample_size_order"] = int(target_order)
    out["requested_train_count"] = np.nan
    out["train_pool_count"] = pd.to_numeric(
        out["train_count"], errors="coerce"
    )
    out["train_count_input"] = pd.to_numeric(
        out["train_count"], errors="coerce"
    )
    out["train_count_effective"] = pd.to_numeric(
        out.get("train_valid_count"), errors="coerce"
    )
    out["pool_wse_range"] = pd.to_numeric(
        out.get("train_wse_range"), errors="coerce"
    )
    return out


def _sample_size_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    data = data.copy()
    data["station_id"] = data["station_id"].map(
        core.normalize_station_id
    )
    expected_labels = set(
        data["sample_size_label"].dropna().astype(str).unique()
    )
    station_label_count = data.groupby(
        "station_id"
    )["sample_size_label"].nunique()
    common_ids = set(
        station_label_count[
            station_label_count >= len(expected_labels)
        ].index.astype(str)
    )
    rows: List[Dict[str, Any]] = []
    metric_columns = (
        "R2", "NSE", "KGE", "RMSE", "MAE", "PBIAS",
        "logR2", "logNSE", "factor_error_median",
    )
    for cohort, cohort_data in (
        ("available_case", data),
        (
            "common_max_size",
            data[data["station_id"].astype(str).isin(common_ids)],
        ),
    ):
        for (
            label,
            order,
        ), group in cohort_data.groupby(
            ["sample_size_label", "sample_size_order"],
            dropna=False,
        ):
            row: Dict[str, Any] = {
                "cohort": cohort,
                "sample_size_label": label,
                "sample_size_order": order,
                "station_count": int(
                    group["station_id"].nunique()
                ),
                "median_train_count_input": float(
                    pd.to_numeric(
                        group["train_count_input"], errors="coerce"
                    ).median()
                ),
                "median_train_count_effective": float(
                    pd.to_numeric(
                        group["train_count_effective"],
                        errors="coerce",
                    ).median()
                ),
                "median_validation_count": float(
                    pd.to_numeric(
                        group["validation_count"], errors="coerce"
                    ).median()
                ),
            }
            for metric in metric_columns:
                values = pd.to_numeric(
                    group.get(metric), errors="coerce"
                )
                if metric == "PBIAS":
                    values = values.abs()
                values = values.dropna()
                suffix = "absPBIAS" if metric == "PBIAS" else metric
                row[f"station_median_{suffix}"] = (
                    float(values.median()) if len(values) else np.nan
                )
                row[f"station_q25_{suffix}"] = (
                    float(values.quantile(0.25))
                    if len(values) else np.nan
                )
                row[f"station_q75_{suffix}"] = (
                    float(values.quantile(0.75))
                    if len(values) else np.nan
                )
                row[f"station_valid_{suffix}_count"] = int(
                    len(values)
                )
            for flag in (
                "n_at_lower_bound",
                "n_at_upper_bound",
                "beta_at_lower_bound",
                "beta_at_upper_bound",
                "width0_at_lower_bound",
                "width0_at_upper_bound",
            ):
                if flag in group:
                    values = group[flag].astype(
                        str
                    ).str.lower().isin(["true", "1"])
                    row[f"{flag}_fraction"] = float(values.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _observational_sample_size_summary(
    tables_dir: Path,
) -> pd.DataFrame:
    """
    利用已经产生的 WSE/时间折，描述实际训练记录数与性能的关联。

    这是观测性结果，可能混入站点难度、WSE 覆盖和时间跨度差异；
    不能替代固定验证集的受控学习曲线。
    """
    path = tables_dir / "validation_metrics_by_fold.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, low_memory=False)
    data = data[
        (data["method"] == PRIMARY_METHOD)
        & data["validation_scheme"].isin(
            ["wse_stratified", "temporal_forward"]
        )
    ].copy()
    train_count = pd.to_numeric(
        data["train_count"], errors="coerce"
    )
    data["train_count_bin"] = pd.cut(
        train_count,
        bins=[4, 7, 11, 19, 29, np.inf],
        labels=["5-7", "8-11", "12-19", "20-29", "30+"],
    ).astype(str)
    rows: List[Dict[str, Any]] = []
    for (
        scheme,
        size_bin,
    ), group in data.groupby(
        ["validation_scheme", "train_count_bin"],
        dropna=False,
    ):
        row: Dict[str, Any] = {
            "validation_scheme": scheme,
            "train_count_bin": size_bin,
            "fold_count": int(len(group)),
            "station_count": int(group["station_id"].nunique()),
            "median_train_count": float(
                pd.to_numeric(
                    group["train_count"], errors="coerce"
                ).median()
            ),
        }
        for metric in ("R2", "NSE", "KGE", "RMSE", "MAE", "PBIAS"):
            values = pd.to_numeric(
                group.get(metric), errors="coerce"
            )
            if metric == "PBIAS":
                values = values.abs()
            values = values.dropna()
            suffix = "absPBIAS" if metric == "PBIAS" else metric
            row[f"median_{suffix}"] = (
                float(values.median()) if len(values) else np.nan
            )
            row[f"q25_{suffix}"] = (
                float(values.quantile(0.25))
                if len(values) else np.nan
            )
            row[f"q75_{suffix}"] = (
                float(values.quantile(0.75))
                if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_sample_size_analysis(
    cfg: BetaRevisionConfig,
    tables_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not cfg.run_sample_size_analysis:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    files = sorted(Path(cfg.data_dir).glob("*.csv"))
    if cfg.max_stations is not None:
        files = files[:max(0, cfg.max_stations)]
    results: List[Dict[str, Any]] = []
    print(
        "Sample-size learning curve: fixed validation set; "
        f"targets={cfg.sample_size_targets}",
        flush=True,
    )
    if cfg.use_parallel and cfg.workers > 1:
        with ProcessPoolExecutor(
            max_workers=cfg.workers
        ) as pool:
            futures = {
                pool.submit(
                    sample_size_station_worker, str(path), cfg
                ): path
                for path in files
            }
            for completed, future in enumerate(
                as_completed(futures), start=1
            ):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({
                        "rows": [],
                        "failures": [{
                            "station_id": core.normalize_station_id(
                                futures[future].stem
                            ),
                            "requested_train_count": np.nan,
                            "reason": (
                                f"process_pool: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }],
                    })
                if (
                    completed % 20 == 0
                    or completed == len(futures)
                ):
                    print(
                        "Sample-size progress: "
                        f"{completed}/{len(futures)}",
                        flush=True,
                    )
    else:
        for index, path in enumerate(files, start=1):
            results.append(
                sample_size_station_worker(str(path), cfg)
            )
            if index % 10 == 0 or index == len(files):
                print(
                    f"Sample-size progress: {index}/{len(files)}",
                    flush=True,
                )

    controlled = pd.DataFrame([
        row for result in results for row in result["rows"]
    ])
    failures = pd.DataFrame([
        row for result in results for row in result["failures"]
    ])
    all_rows = _all_available_sample_size_rows(
        tables_dir,
        target_order=(
            max(cfg.sample_size_targets, default=20) + 1
        ),
    )
    controlled = pd.concat(
        [controlled, all_rows], ignore_index=True, sort=False
    )
    summary = _sample_size_summary(controlled)
    observational = _observational_sample_size_summary(
        tables_dir
    )
    return controlled, summary, observational, failures


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _postprocess_beta_outputs(cfg: BetaRevisionConfig) -> None:
    output_dir = Path(cfg.output_dir)
    tables_dir = output_dir / "tables"
    prediction_path = tables_dir / "validation_predictions.csv"
    predictions = (
        pd.read_csv(prediction_path, low_memory=False)
        if prediction_path.exists()
        else pd.DataFrame()
    )
    beta_rows = (
        _beta_sensitivity_rows(predictions, cfg)
        if cfg.run_beta_sensitivity
        else pd.DataFrame()
    )
    beta_summary = _beta_sensitivity_summary(beta_rows)
    _save_table(
        beta_rows, tables_dir / "beta_sensitivity_by_station.csv"
    )
    _save_table(
        beta_summary, tables_dir / "beta_sensitivity_summary.csv"
    )
    (
        sample_size_rows,
        sample_size_summary,
        sample_size_observational,
        sample_size_failures,
    ) = _run_sample_size_analysis(cfg, tables_dir)
    _save_table(
        sample_size_rows,
        tables_dir / "sample_size_learning_curve_by_station.csv",
    )
    _save_table(
        sample_size_summary,
        tables_dir / "sample_size_learning_curve_summary.csv",
    )
    _save_table(
        sample_size_observational,
        tables_dir / "sample_size_observational_summary.csv",
    )
    _save_table(
        sample_size_failures,
        tables_dir / "sample_size_learning_curve_failures.csv",
    )

    manifest_path = tables_dir / "table_manifest.csv"
    manifest = (
        pd.read_csv(manifest_path)
        if manifest_path.exists()
        else pd.DataFrame(
            columns=["file", "row_count", "column_count"]
        )
    )
    extra = pd.DataFrame([
        {
            "file": "beta_sensitivity_by_station.csv",
            "row_count": int(len(beta_rows)),
            "column_count": int(len(beta_rows.columns)),
        },
        {
            "file": "beta_sensitivity_summary.csv",
            "row_count": int(len(beta_summary)),
            "column_count": int(len(beta_summary.columns)),
        },
        {
            "file": "sample_size_learning_curve_by_station.csv",
            "row_count": int(len(sample_size_rows)),
            "column_count": int(len(sample_size_rows.columns)),
        },
        {
            "file": "sample_size_learning_curve_summary.csv",
            "row_count": int(len(sample_size_summary)),
            "column_count": int(len(sample_size_summary.columns)),
        },
        {
            "file": "sample_size_observational_summary.csv",
            "row_count": int(len(sample_size_observational)),
            "column_count": int(len(sample_size_observational.columns)),
        },
        {
            "file": "sample_size_learning_curve_failures.csv",
            "row_count": int(len(sample_size_failures)),
            "column_count": int(len(sample_size_failures.columns)),
        },
    ])
    if not manifest.empty:
        manifest = manifest[
            ~manifest["file"].isin(extra["file"])
        ]
    _save_table(
        pd.concat([manifest, extra], ignore_index=True),
        manifest_path,
    )

    with open(
        output_dir / "software_environment.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "calculation_entry": str(Path(__file__).resolve()),
            "hydraulic_core": str(
                (SCRIPT_DIR / "swot_beta_wse0_width0_model.py").resolve()
            ),
            "reused_experiment_pipeline": str(
                (
                    SCRIPT_DIR
                    / "run_review_revision_experiments.py"
                ).resolve()
            ),
            "spatial_cross_validation": "removed_by_current_protocol",
        }, handle, ensure_ascii=False, indent=2)

    with open(
        output_dir / "method_and_optimizer_definition.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({
            "primary_method": (
                "station-specific joint optimization of n, beta, "
                "WSE0/depth0 and width0"
            ),
            "bed_geometry": (
                "A0 = 0.5 * (width0 + width_at_min_training_WSE) "
                "* (min_training_WSE - WSE0)"
            ),
            "area_above_minimum_training_wse": (
                "A = A0 + beta * trapezoidal integral of the station "
                "monotonic width-WSE curve"
            ),
            "width_curve_validation_scope": (
                "station reconstruction uses all station covariates; "
                "WSE, temporal, ablation, and sample-size validation "
                "fit the monotonic curve only on that fold's calibration "
                "WSE-width observations. Validation width never fits or "
                "updates the curve. Spatial station-block transfer uses "
                "the held-out station's satellite WSE-width observations "
                "but no held-out station discharge for curve fitting."
            ),
            "width_curve_out_of_bounds": (
                "endpoint clipping with explicit width_curve_position flag"
            ),
            "area_below_minimum_training_wse": (
                "direct trapezoid from (WSE0,width0) to the current "
                "observation; beta is not applied"
            ),
            "objective": cfg.objective,
            "optimizer": (
                "differential_evolution global search followed by "
                "bounded Powell"
            ),
            "optimizer_stopping_rule": {
                "tol": cfg.optimizer_tol,
                "maxiter_primary": (
                    10 if cfg.fast else cfg.optimizer_maxiter
                ),
                "popsize_primary": (
                    5 if cfg.fast else cfg.optimizer_popsize
                ),
                "local_maxiter": (
                    80 if cfg.fast else cfg.local_optimizer_maxiter
                ),
                "workers_inside_optimizer": 1,
                "selection": (
                    "lowest finite objective loss across deterministic "
                    "starts"
                ),
            },
            "parameter_bounds": {
                "n": [core.N_MIN, core.N_MAX],
                "n_typical_diagnostic_range": [
                    core.N_TYPICAL_MIN, core.N_TYPICAL_MAX
                ],
                "beta": [cfg.beta_min, cfg.beta_max],
                "beta_prior_weight": cfg.beta_prior_weight,
                "depth0_lower": core.DEPTH_MIN,
                "depth0_upper_per_fit": (
                    "min(5000 m, max(50 m, "
                    "20 * training WSE range))"
                ),
                "width0_fraction_of_minimum_width": [
                    cfg.width0_ratio_min,
                    cfg.width0_ratio_max,
                ],
                "A0_bounds": (
                    "written explicitly for every fit in "
                    "parameter_estimates.csv"
                ),
            },
            "hampel": {
                "window": core.HAMPEL_WINDOW,
                "n_sigmas": core.HAMPEL_N_SIGMAS,
            },
            "minimum_training_records": core.MIN_TRAIN_RECORDS,
            "wse_validation": {
                "strata": 5,
                "training_fraction": 0.70,
                "repeated_random_grouping_enabled": (
                    cfg.repeat_random_grouping
                ),
                "effective_repeat_count": cfg.repeat_count,
                "ablation_repeat_count": cfg.ablation_repeat_count,
            },
            "temporal_validation": {
                "folds": cfg.temporal_folds,
                "expanding_window": True,
            },
            "sample_size_analysis": {
                "enabled": cfg.run_sample_size_analysis,
                "training_targets": list(cfg.sample_size_targets),
                "validation_set": (
                    "fixed repeat-0 WSE-stratified holdout"
                ),
                "training_subsets": (
                    "nested WSE-balanced prefixes of the same "
                    "training pool"
                ),
                "all_available_reference": (
                    "reused from the primary repeat-0 WSE split"
                ),
                "summary_cohorts": [
                    "available_case",
                    "common_max_size",
                ],
            },
            "spatial_cross_validation": cfg.run_spatial_cv,
            "spatial_folds": cfg.spatial_folds,
            "spatial_validation_protocol": "Large-region transfer removed; local transfer radius remains untested.",
            "interpretation_warning": (
                "n, beta, WSE0 and width0 are effective station "
                "parameters; A0 and multi-start stability must also be "
                "reported because the bed pair may not be unique"
            ),
        }, handle, ensure_ascii=False, indent=2)


def write_revised_protocol(cfg):
    out=Path(cfg.output_dir)
    path=out/'method_and_optimizer_definition.json'
    info=json.loads(path.read_text(encoding='utf-8'))
    info['hampel'] = {**quality.asdict(quality.RULES),'scope':'full-station time series before split, identical flags on training and validation','scale':'1.4826 times exact local MAD, with stated numeric floors','fields':['WSE','width','log10(slope)'],'invalid_slope':'excluded in every method, including no-Hampel','endpoints':'truncated windows, minimum three finite values','gauge_discharge_used':False}
    info['width_curve_validation_scope']='All curve parameters estimated from retained calibration covariates only; frozen for validation. width_train_only uses observed validation widths. No validation width updates any curve.'
    info['width_curve_guards']={'initial':'First unique-WSE width checked against up to five following anchors within 1 m; needs >=2 neighbors and >3-fold local median by default. Replace only the inconsistent first width.','growth':'After isotonic fitting, cap each positive increment by min(absolute_gradient, relative_gradient * previous_width) * delta_WSE. Defaults 100 m/m and 2 per m.','config':quality.asdict(quality.RULES),'interpretation':'Experimental configurable limits; no universal geomorphic threshold is claimed.'}
    info['temporal_validation']={'folds':cfg.temporal_folds,'expanding_window':True,'initial_training_fraction':cfg.temporal_train_fraction,'scope':'retrospective evaluation following full-record satellite-only QC; not a strict real-time forecast'}
    info['ablation_methods']=list(ABLATION_METHODS)
    info['paired_comparison_support']='Exact shared station/observation/repeat test rows for each pair; native-support performance and QC coverage are retained separately.'
    info['validation_optimizer_starts']=cfg.validation_multistart_count
    info['spatial_cross_validation']=False
    info.pop('spatial_folds',None)
    info['spatial_validation_protocol']='Large-region median transfer removed. Local transfer distance remains an untested future research question.'
    path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')


def main() -> int:
    cfg = parse_args()
    if (Path(cfg.output_dir)/'run_configuration.json').exists():
        raise FileExistsError('Existing run found. Select a new ASCII --output-dir to preserve prior results.')
    _install_overrides()
    quality.configure(cfg)
    pipeline.parse_args = lambda: cfg
    print(
        "Repeated WSE grouping: "
        f"{'ON' if cfg.repeat_random_grouping else 'OFF'}; "
        f"effective repeats={cfg.repeat_count}",
        flush=True,
    )
    status = pipeline.main()
    if status == 0:
        _postprocess_beta_outputs(cfg)
        write_revised_protocol(cfg)
        import revision_audits
        revision_audits.export(cfg.output_dir)

        print(
            "Next: run plot_beta_wse0_width0_results.py "
            "--results-dir <output_dir>",
            flush=True,
        )
    return status


_install_overrides()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
