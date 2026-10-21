#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRR 审稿意见补充实验：SWOT 流量复现与稳健性分析。

本脚本复用 ``swot_n_wse0_width0_nonnegative_cv.py`` 中已经验证过的水力
计算、Hampel、单调回归和指标函数，但将“算法版本”和“验证方式”分开：

主方法
    full = Hampel + 全站单调 width + 逐站优化 n/WSE0/最低 WSE 有效宽度；
           width0 固定为 0

验证方式
    station_reconstruction  全样本逐站率定与复现
    wse_stratified          WSE 分层随机留出，默认重复 10 次
    temporal_forward        严格时间向前扩展窗口验证

消融/基线（默认只对第 1 次 WSE 划分和全样本复现运行，控制耗时）
    no_hampel               关闭 Hampel
    no_monotonic            关闭单调 width
    fixed_reference_n       固定元数据/全局中位 Manning n，仅优化 WSE0/最低 WSE 有效宽度
    n_only_fixed_geometry   使用明确的简单起算几何，仅优化 n
    empirical_rating_curve  经验 WSE-Q 幂函数基线

附加审稿分析
    - n 相对扰动与绝对网格敏感性；
    - 全样本率定多随机种子稳定性；
    - 原始行、有效节点、独立过境和可率定站点的逐级 QC 审计；
    - 流量等级、WSE 插值/外推、时间间隔和洪峰误差；
    - 站点性能驱动因素和透明的 Spearman 相关分析；
    - 站点级 bootstrap 置信区间和配对消融检验。

空间交叉验证在本版本中明确关闭，不会运行。

运行：
    python run_review_revision_experiments.py

快速检查：
    python run_review_revision_experiments.py --max-stations 10 --repeat-count 2 --fast --serial
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import swot_n_wse0_width0_nonnegative_cv as core


DEFAULT_DATA_DIR = Path(r"D:\SWOT5.1 - 大改重投\103.水文站卫星数据融合0506_adjusted")
DEFAULT_METADATA_FILE = Path(r"D:\SWOT5.1 - 大改重投\4_Station_with_N\Find_N.csv")
DEFAULT_OUTPUT_DIR = Path(
    r"D:\SWOT5.1 - 大改重投\20260723_swot_review_revision_experiments_revised_geometry"
)
DEFAULT_BASIN_SHAPEFILE = Path(r"D:\SWOT5.1 - 大改重投\BasinSHP\Class2_hydrounits.shp")

PRIMARY_METHOD = "full"
ABLATION_METHODS = (
    "no_hampel",
    "no_monotonic",
    "fixed_reference_n",
    "n_only_fixed_geometry",
    "empirical_rating_curve",
)
METHOD_LABELS = {
    "full": "Full: Hampel + monotonic width + optimized n/WSE0/minimum-WSE width",
    "no_hampel": "No Hampel",
    "no_monotonic": "No monotonic-width constraint",
    "fixed_reference_n": "Fixed reference Manning n",
    "n_only_fixed_geometry": "Optimized n only; fixed simple geometry",
    "empirical_rating_curve": "Empirical WSE-Q rating curve",
}

RELATIVE_N_MULTIPLIERS = (0.50, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50)
ABSOLUTE_N_GRID = (
    0.010, 0.015, 0.020, 0.025, 0.030, 0.035,
    0.040, 0.050, 0.075, 0.100, 0.125, 0.150,
)
METRICS = (
    "R2", "NSE", "KGE", "logR2", "logNSE", "RMSE", "MAE", "Bias", "PBIAS",
    "median_signed_log_ratio", "median_abs_log_ratio", "factor_error_median",
)
HIGHER_IS_BETTER = {"R2", "NSE", "KGE", "logR2", "logNSE"}


@dataclass
class RevisionConfig:
    data_dir: str
    metadata_file: str
    output_dir: str
    repeat_count: int = 10
    ablation_repeat_count: int = 10
    temporal_folds: int = 3
    multistart_count: int = 3
    bootstrap_iterations: int = 2000
    workers: int = max(1, min(8, (os.cpu_count() or 2) - 1))
    use_parallel: bool = True
    max_stations: Optional[int] = None
    fast: bool = False
    seed: int = 20260723
    objective: str = "old_composite"
    aggregate_nodes_by_overpass: bool = True
    min_width_factor_low: float = core.MIN_WIDTH_FACTOR_LOW_DEFAULT
    min_width_factor_high: float = core.MIN_WIDTH_FACTOR_HIGH_DEFAULT
    optimizer_maxiter: int = 60
    optimizer_popsize: int = 8
    optimizer_tol: float = 1e-3
    local_optimizer_maxiter: int = 400
    basin_shapefile: str = str(DEFAULT_BASIN_SHAPEFILE)
    run_temporal: bool = True
    run_ablation: bool = True
    run_ablation_reconstruction: bool = True
    run_n_sensitivity: bool = True
    run_multistart: bool = True
    # Spatial station-block transfer is handled by spatial_transfer_cv.py
    # from the beta entry point after the within-station pipeline finishes.
    run_spatial_cv: bool = False


def stable_seed(*parts: Any, base: int) -> int:
    return core.stable_seed(*parts, base=base)


def finite_number(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def geographic_region(lon: Any, lat: Any) -> str:
    """仅作广域地理分层，不冒充流域分类。"""
    x, y = finite_number(lon), finite_number(lat)
    if not np.isfinite(x) or not np.isfinite(y):
        return "Unknown"
    east_west = "East" if x >= 105.0 else "West"
    north_south = "North" if y >= 35.0 else "South"
    return north_south + east_west


def attach_basin_attributes(
    metadata: Dict[str, Dict[str, Any]],
    shapefile_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    将站点离线匹配到本地二级水资源区。

    使用 ``covers`` 而不是 ``within``，使位于流域边界上的站点也能匹配。
    不进行未经标记的最近邻强制归类；未覆盖点保留 Unknown，便于审计。
    """
    if not shapefile_path.exists():
        for row in metadata.values():
            row.update({
                "basin_code": "",
                "basin_name": "Unknown",
                "basin_match_method": "shapefile_missing",
            })
        return metadata
    try:
        import shapefile  # pyshp
        from shapely.geometry import Point, shape as shapely_shape
    except Exception as exc:
        raise RuntimeError(
            "真实流域匹配需要 pyshp 与 shapely；当前环境无法导入。"
        ) from exc

    reader = shapefile.Reader(str(shapefile_path), encoding="gbk")
    fields = [field[0] for field in reader.fields[1:]]
    basin_geometries: List[Tuple[Any, Dict[str, Any]]] = []
    for shape_record in reader.iterShapeRecords():
        record = dict(zip(fields, list(shape_record.record)))
        basin_geometries.append((
            shapely_shape(shape_record.shape.__geo_interface__),
            record,
        ))

    for row in metadata.values():
        lon = finite_number(row.get("Lon_web"))
        lat = finite_number(row.get("Lat_web"))
        matches: List[Tuple[Any, Dict[str, Any]]] = []
        if np.isfinite(lon) and np.isfinite(lat):
            point = Point(lon, lat)
            matches = [
                (geometry, record)
                for geometry, record in basin_geometries
                if geometry.covers(point)
            ]
        if matches:
            # 重叠时选择面积最小的水文单元，避免被更粗的外包多边形覆盖。
            _, selected = min(matches, key=lambda item: float(item[0].area))
            row.update({
                "basin_code": str(selected.get("WRRCD", "")),
                "basin_name": str(selected.get("WRRNM", "Unknown")),
                "basin_match_method": "polygon_covers",
            })
        else:
            row.update({
                "basin_code": "",
                "basin_name": "Unknown",
                "basin_match_method": "unmatched",
            })
    return metadata


def parse_reason(time_ok: bool, wse_ok: bool, width_finite: bool, width_positive: bool,
                 q_finite: bool, q_positive: bool) -> str:
    reasons: List[str] = []
    if not time_ok:
        reasons.append("invalid_time")
    if not wse_ok:
        reasons.append("invalid_wse")
    if not width_finite:
        reasons.append("invalid_width")
    elif not width_positive:
        reasons.append("nonpositive_width")
    if not q_finite:
        reasons.append("invalid_Q")
    elif not q_positive:
        reasons.append("nonpositive_Q")
    return "retained_basic" if not reasons else ";".join(reasons)


def load_station_with_audit(path: Path, aggregate_nodes: bool) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """读取一个站点，同时保留原始行和独立过境两种粒度的 QC 轨迹。"""
    station_id = core.normalize_station_id(path.stem)
    raw = core.read_csv_safely(path)

    def first_column(candidates: Sequence[str]) -> Optional[str]:
        return next((column for column in candidates if column in raw.columns), None)

    # Support both the original station tables and the new Version-D fusion
    # schema.  gauge_Q is deliberately preferred/required for the fused data:
    # SWOT discharge products must never be substituted for gauge observations.
    wse_column = first_column(("wse_datum_adjusted", "wse_adj", "wse"))
    width_column = first_column(("width", "reach_width"))
    slope_column = first_column(("reach_slope", "slope", "reach_slope2"))
    q_column = first_column(("gauge_Q", "Q"))
    time_column = first_column(("time_utc", "swot_date", "time_str", "date", "time"))
    selected_columns = {
        "WSE": wse_column,
        "width": width_column,
        "slope": slope_column,
        "Q": q_column,
        "time": time_column,
    }
    missing = [name for name, column in selected_columns.items() if column is None]
    if missing:
        raise ValueError(
            "Missing required variables: "
            f"{missing}; available columns={list(raw.columns)}"
        )

    raw_audit = pd.DataFrame({
        "station_id": station_id,
        "source_file": str(path),
        "raw_row_id": np.arange(len(raw), dtype=int),
        "time_raw": raw[time_column].astype(str),
        "time_dt": pd.to_datetime(raw[time_column], errors="coerce", utc=True),
        "WSE": pd.to_numeric(raw[wse_column], errors="coerce"),
        "width": pd.to_numeric(raw[width_column], errors="coerce"),
        "slope": pd.to_numeric(raw[slope_column], errors="coerce"),
        "Q": pd.to_numeric(raw[q_column], errors="coerce"),
    })
    t_ok = raw_audit["time_dt"].notna().to_numpy()
    wse_ok = np.isfinite(raw_audit["WSE"].to_numpy(dtype=float))
    width_finite = np.isfinite(raw_audit["width"].to_numpy(dtype=float))
    width_positive = raw_audit["width"].to_numpy(dtype=float) > 0
    q_finite = np.isfinite(raw_audit["Q"].to_numpy(dtype=float))
    q_positive = raw_audit["Q"].to_numpy(dtype=float) > 0
    retained = t_ok & wse_ok & width_finite & width_positive & q_finite & q_positive
    raw_audit["basic_qc_pass"] = retained
    raw_audit["basic_qc_reason"] = [
        parse_reason(a, b, c, d, e, f)
        for a, b, c, d, e, f in zip(t_ok, wse_ok, width_finite, width_positive, q_finite, q_positive)
    ]
    raw_audit["slope_requires_repair"] = ~(
        np.isfinite(raw_audit["slope"].to_numpy(dtype=float)) &
        (raw_audit["slope"].to_numpy(dtype=float) > 0)
    )

    valid = raw_audit.loc[retained, ["station_id", "source_file", "time_dt", "WSE", "width", "slope", "Q"]].copy()
    valid = valid.rename(columns={"WSE": "wse_model"})
    group_size = valid.groupby("time_dt")["time_dt"].transform("size") if len(valid) else pd.Series(dtype=float)
    valid["overpass_group_size"] = group_size.to_numpy(dtype=int) if len(valid) else []
    size_map = valid.groupby("time_dt").size().to_dict() if len(valid) else {}
    raw_audit["overpass_group_size"] = raw_audit["time_dt"].map(size_map).fillna(0).astype(int)
    raw_audit["merged_with_same_time_nodes"] = raw_audit["basic_qc_pass"] & (raw_audit["overpass_group_size"] > 1)

    if aggregate_nodes:
        frame = core.aggregate_overpasses(valid)
    else:
        frame = valid.drop(columns=["overpass_group_size"], errors="ignore").sort_values("time_dt").reset_index(drop=True)
    frame = frame.sort_values("time_dt").reset_index(drop=True)
    frame["obs_id"] = [f"{station_id}|{t.isoformat()}|{i}" for i, t in enumerate(frame["time_dt"])]
    frame["time_gap_days"] = frame["time_dt"].diff().dt.total_seconds().div(86400.0)
    ranks = frame["Q"].rank(method="average", pct=True)
    frame["flow_regime"] = pd.cut(
        ranks, bins=[0.0, 0.50, 0.80, 0.95, 1.0000001],
        labels=["Q00-50", "Q50-80", "Q80-95", "Q95-100"], include_lowest=True,
    ).astype(str)

    overpass = frame[[
        "station_id", "obs_id", "time_dt", "wse_model", "width", "slope", "Q",
        "time_gap_days", "flow_regime",
    ]].copy()
    overpass["overpass_group_size"] = overpass["time_dt"].map(size_map).fillna(1).astype(int)
    overpass["hampel_wse_candidate"] = core.hampel_mask(overpass["wse_model"])
    overpass["hampel_width_candidate"] = core.hampel_mask(overpass["width"])
    positive_slope = pd.to_numeric(overpass["slope"], errors="coerce").where(lambda x: x > 0)
    overpass["hampel_slope_candidate"] = core.hampel_mask(np.log10(positive_slope))
    overpass["slope_requires_repair"] = ~(
        np.isfinite(overpass["slope"].to_numpy(dtype=float)) &
        (overpass["slope"].to_numpy(dtype=float) > 0)
    )
    overpass["primary_training_qc_pass"] = ~(
        overpass["hampel_wse_candidate"] | overpass["hampel_width_candidate"]
    )

    audit = {
        "station_id": station_id,
        "source_file": str(path),
        "raw_row_count": int(len(raw_audit)),
        "parsed_time_count": int(t_ok.sum()),
        "valid_row_count_before_overpass_aggregation": int(retained.sum()),
        "overpass_count": int(len(frame)),
        "wse_source_column": wse_column,
        "width_source_column": width_column,
        "slope_source_column": slope_column,
        "q_source_column": q_column,
        "time_source_column": time_column,
        "raw_invalid_time_count": int((~t_ok).sum()),
        "raw_invalid_wse_count": int((~wse_ok).sum()),
        "raw_invalid_or_nonpositive_width_count": int((~(width_finite & width_positive)).sum()),
        "raw_invalid_or_nonpositive_Q_count": int((~(q_finite & q_positive)).sum()),
        "raw_slope_repair_candidate_count": int(raw_audit["slope_requires_repair"].sum()),
        "same_time_node_rows_count": int((raw_audit["overpass_group_size"] > 1).sum()),
    }
    return frame, audit, raw_audit, overpass


def make_core_config(cfg: RevisionConfig, enable_hampel: bool, monotonic: bool) -> core.Config:
    return core.Config(
        data_dir=cfg.data_dir,
        metadata_file=cfg.metadata_file,
        output_dir=cfg.output_dir,
        repeat_random_grouping=True,
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
        min_width_factor_low=cfg.min_width_factor_low,
        min_width_factor_high=cfg.min_width_factor_high,
        optimizer_maxiter=cfg.optimizer_maxiter,
        optimizer_popsize=cfg.optimizer_popsize,
        optimizer_tol=cfg.optimizer_tol,
        local_optimizer_maxiter=cfg.local_optimizer_maxiter,
    )


def make_frame_versions(base: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    primary, primary_diag = core.apply_global_station_monotonic_width(base, enable_hampel=True)
    no_hampel, _ = core.apply_global_station_monotonic_width(base, enable_hampel=False)
    no_monotonic = base.copy()
    no_monotonic["width_model"] = np.maximum(
        pd.to_numeric(no_monotonic["width"], errors="coerce").to_numpy(dtype=float), core.WIDTH_FLOOR
    )
    versions = {
        "full": primary,
        "no_hampel": no_hampel,
        "no_monotonic": no_monotonic,
        "fixed_reference_n": primary.copy(),
        "n_only_fixed_geometry": primary.copy(),
        "empirical_rating_curve": primary.copy(),
    }
    return versions, primary_diag


def make_split_frame_versions(
    base: pd.DataFrame,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
) -> Tuple[
    Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    Dict[str, Any],
]:
    """默认兼容实现；具体实验版本可覆盖为仅用率定折拟合几何。"""
    versions, diagnostics = make_frame_versions(base)
    pairs = {
        method: (
            select_rows(frame, train_ids),
            select_rows(frame, test_ids),
        )
        for method, frame in versions.items()
    }
    return pairs, diagnostics


def objective_from_predictions(obs: np.ndarray, sim: np.ndarray, objective: str) -> float:
    if not np.all(np.isfinite(sim)):
        return 1e12
    if objective == "old_composite":
        return -core.old_composite_score(obs, sim)
    if objective == "normalized_mse":
        scale = max(float(np.std(obs)), 0.1 * float(np.mean(obs)), core.EPS)
        return float(np.mean(((sim - obs) / scale) ** 2))
    if objective == "log_mse":
        return float(np.mean((np.log10(np.maximum(sim, core.EPS)) - np.log10(obs)) ** 2))
    raise ValueError(f"Unknown objective: {objective}")


def hydraulic_fit_result(train: pd.DataFrame, qc: Dict[str, Any], n_value: float,
                         depth0: float, min_width_effective: float, objective_loss: float,
                         optimizer: Mapping[str, Any], parameter_source: str) -> Dict[str, Any]:
    anchor_wse, anchor_width_observed = core.collapse_wse_width(
        train["wse_model"], train["width_model"]
    )
    anchor_width = core.apply_effective_min_width(
        anchor_width_observed, min_width_effective
    )
    min_width_observed = float(anchor_width_observed[0])
    min_width_effective = float(anchor_width[0])
    qtrain, _ = core.training_predictions(
        train, n_value, depth0, min_width_effective
    )
    wse_span = max(float(np.ptp(anchor_wse)), 0.1)
    wse0 = float(anchor_wse[0] - depth0)
    return {
        "model_kind": "hydraulic",
        "parameter_source": parameter_source,
        "n": float(n_value),
        "WSE0": wse0,
        "width0": core.WIDTH0_FIXED,
        "depth0": float(depth0),
        "A0": float(0.5 * min_width_effective * depth0),
        "train_wse_min": float(anchor_wse[0]),
        "train_wse_max": float(anchor_wse[-1]),
        "train_wse_range": float(np.ptp(anchor_wse)),
        "train_width_at_min_wse": min_width_effective,
        "min_width_observed": min_width_observed,
        "min_width_effective": min_width_effective,
        "min_width_factor": min_width_effective / max(min_width_observed, core.WIDTH_FLOOR),
        "depth_to_wse_range_ratio": float(depth0 / wse_span),
        "width0_to_min_width_ratio": 0.0,
        "objective_loss": float(objective_loss),
        "optimizer_success": bool(optimizer.get("success", True)),
        "fit_accepted": bool(optimizer.get("fit_accepted", True)),
        "global_optimizer_success": bool(optimizer.get("global_success", True)),
        "local_optimizer_success": bool(optimizer.get("local_success", True)),
        "optimizer_selected_stage": str(optimizer.get("selected_stage", "")),
        "optimizer_message": str(optimizer.get("message", "")),
        "optimizer_nfev": int(optimizer.get("nfev", 0)),
        "optimizer_nit": int(optimizer.get("nit", 0)),
        "n_in_typical_range": bool(core.N_TYPICAL_MIN <= n_value <= core.N_TYPICAL_MAX),
        "n_above_typical_range": bool(n_value > core.N_TYPICAL_MAX),
        "n_at_lower_bound": bool(n_value <= core.N_MIN * 1.01),
        "n_at_upper_bound": bool(n_value >= core.N_MAX * 0.99),
        "depth_at_lower_bound": bool(depth0 <= core.DEPTH_MIN * 1.01),
        "depth_at_upper_bound": False,
        "width0_fixed_zero": True,
        "width0_at_lower_bound": True,
        "width0_at_upper_bound": False,
        "anchor_wse": anchor_wse,
        "anchor_width": anchor_width,
        "train_slope_fallback": float(qc["slope_fallback"]),
        "train_metrics": core.calculate_metrics(train["Q"], qtrain),
        "qc": qc,
    }


def fit_fixed_reference_n(train_input: pd.DataFrame, cfg: RevisionConfig, seed: int,
                          n_reference: float) -> Optional[Dict[str, Any]]:
    train, qc = core.prepare_training(train_input, enable_hampel=True)
    if train is None:
        return None
    anchor_wse, anchor_width = core.collapse_wse_width(train["wse_model"], train["width_model"])
    span = max(float(np.ptp(anchor_wse)), 0.1)
    depth_upper = min(core.DEPTH_ABSOLUTE_MAX, max(50.0, 20.0 * span))
    min_width_observed, min_width_lower, min_width_upper = core.effective_min_width_bounds(
        anchor_width,
        cfg.min_width_factor_low,
        cfg.min_width_factor_high,
    )
    bounds = [(math.log(core.DEPTH_MIN), math.log(depth_upper)),
              (math.log(min_width_lower), math.log(min_width_upper))]
    obs = train["Q"].to_numpy(dtype=float)

    def loss(z: np.ndarray) -> float:
        depth0, min_width_effective = np.exp(z)
        sim, _ = core.training_predictions(
            train, n_reference, depth0, min_width_effective
        )
        return objective_from_predictions(obs, sim, cfg.objective)

    result = core.hybrid_global_local_optimize(
        loss,
        bounds=bounds,
        args=(),
        seed=seed,
        fast=cfg.fast,
        maxiter=cfg.optimizer_maxiter,
        popsize=cfg.optimizer_popsize,
        tol=cfg.optimizer_tol,
        local_maxiter=cfg.local_optimizer_maxiter,
    )
    if not np.isfinite(result["fun"]):
        return None
    depth0, min_width_effective = np.exp(result["x"])
    fit = hydraulic_fit_result(
        train, qc, n_reference, depth0, min_width_effective, float(result["fun"]),
        result,
        "metadata_N_River_or_global_median",
    )
    fit.update({
        "min_width_search_lower": min_width_lower,
        "min_width_search_upper": min_width_upper,
        "min_width_at_lower_bound": bool(
            min_width_effective <= min_width_lower * 1.01
        ),
        "min_width_at_upper_bound": bool(
            min_width_effective >= min_width_upper * 0.99
        ),
    })
    return fit


def fit_n_only_fixed_geometry(train_input: pd.DataFrame, cfg: RevisionConfig) -> Optional[Dict[str, Any]]:
    train, qc = core.prepare_training(train_input, enable_hampel=True)
    if train is None:
        return None
    anchor_wse, anchor_width = core.collapse_wse_width(train["wse_model"], train["width_model"])
    # 明确而可复现的简单几何基线：最低观测以下深度=max(0.5 m, 观测 WSE 范围)，
    # 河床起点宽度=0，最低 WSE 有效宽度=对应单调观测宽度。
    depth0 = max(0.5, float(np.ptp(anchor_wse)))
    min_width_effective = max(float(anchor_width[0]), core.WIDTH_FLOOR)
    obs = train["Q"].to_numpy(dtype=float)

    def scalar_loss(log_n: float) -> float:
        sim, _ = core.training_predictions(
            train, math.exp(log_n), depth0, min_width_effective
        )
        return objective_from_predictions(obs, sim, cfg.objective)

    result = minimize_scalar(
        scalar_loss, bounds=(math.log(core.N_MIN), math.log(core.N_MAX)),
        method="bounded", options={"xatol": 1e-7, "maxiter": 300},
    )
    if not np.isfinite(result.fun):
        return None
    n_value = math.exp(float(result.x))
    return hydraulic_fit_result(
        train, qc, n_value, depth0, min_width_effective, float(result.fun),
        {"success": result.success, "message": result.message, "nfev": result.nfev, "nit": result.nfev},
        "optimized_n_with_fixed_simple_geometry",
    )


def fit_rating_curve(train_input: pd.DataFrame, cfg: RevisionConfig, seed: int) -> Optional[Dict[str, Any]]:
    # 使用与主方法相同的训练 QC 样本，确保算法比较不因样本不同而偏移。
    train, qc = core.prepare_training(train_input, enable_hampel=True)
    if train is None:
        return None
    wse = train["wse_model"].to_numpy(dtype=float)
    obs = train["Q"].to_numpy(dtype=float)
    wmin, wmax = float(np.min(wse)), float(np.max(wse))
    span = max(wmax - wmin, 0.1)
    bounds = [
        (-20.0, 20.0),                          # log(a)
        (0.20, 5.00),                           # b
        (math.log(core.DEPTH_MIN), math.log(max(50.0, 20.0 * span))),  # log(depth)
    ]

    def loss(z: np.ndarray) -> float:
        log_a, exponent, log_depth = z
        h0 = wmin - math.exp(log_depth)
        sim = math.exp(log_a) * np.maximum(wse - h0, core.EPS) ** exponent
        return objective_from_predictions(obs, sim, cfg.objective)

    result = core.hybrid_global_local_optimize(
        loss,
        bounds=bounds,
        args=(),
        seed=seed,
        fast=cfg.fast,
        maxiter=cfg.optimizer_maxiter,
        popsize=cfg.optimizer_popsize,
        tol=cfg.optimizer_tol,
        local_maxiter=cfg.local_optimizer_maxiter,
    )
    if not np.isfinite(result["fun"]):
        return None
    log_a, exponent, log_depth = result["x"]
    depth = math.exp(log_depth)
    h0 = wmin - depth
    sim = math.exp(log_a) * np.maximum(wse - h0, core.EPS) ** exponent
    return {
        "model_kind": "rating_curve",
        "parameter_source": "station_training_WSE_Q_power_law",
        "rating_a": float(math.exp(log_a)),
        "rating_b": float(exponent),
        "rating_h0": float(h0),
        "n": np.nan, "WSE0": float(h0), "width0": np.nan,
        "depth0": float(depth), "A0": np.nan,
        "train_wse_min": wmin, "train_wse_max": wmax, "train_wse_range": wmax - wmin,
        "objective_loss": float(result["fun"]),
        "optimizer_success": bool(result["success"]),
        "fit_accepted": bool(result["fit_accepted"]),
        "global_optimizer_success": bool(result["global_success"]),
        "local_optimizer_success": bool(result["local_success"]),
        "optimizer_selected_stage": result.get("selected_stage", ""),
        "optimizer_message": str(result["message"]),
        "optimizer_nfev": int(result["nfev"]), "optimizer_nit": int(result["nit"]),
        "anchor_wse": np.array([], dtype=float), "anchor_width": np.array([], dtype=float),
        "train_slope_fallback": float(qc["slope_fallback"]),
        "train_metrics": core.calculate_metrics(obs, sim), "qc": qc,
    }


def fit_method(method: str, frame: pd.DataFrame, cfg: RevisionConfig, seed: int,
               n_reference: float) -> Optional[Dict[str, Any]]:
    if method in {"full", "no_hampel", "no_monotonic"}:
        hampel = method != "no_hampel"
        monotonic = method != "no_monotonic"
        return core.fit_parameters(frame, make_core_config(cfg, hampel, monotonic), seed)
    if method == "fixed_reference_n":
        return fit_fixed_reference_n(frame, cfg, seed, n_reference)
    if method == "n_only_fixed_geometry":
        return fit_n_only_fixed_geometry(frame, cfg)
    if method == "empirical_rating_curve":
        return fit_rating_curve(frame, cfg, seed)
    raise ValueError(f"Unknown method: {method}")


def predict_rating_curve(fit: Mapping[str, Any], frame: pd.DataFrame, validation_scheme: str,
                         fold_id: int, repeat_id: int, sample_role: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    wmin, wmax = float(fit["train_wse_min"]), float(fit["train_wse_max"])
    for _, row in frame.iterrows():
        wse = float(row["wse_model"])
        qcalc = float(fit["rating_a"] * max(wse - fit["rating_h0"], core.EPS) ** fit["rating_b"])
        if wse < wmin:
            position = "below_train_min_wse"
        elif wse > wmax:
            position = "above_train_max_wse"
        else:
            position = "within_train_wse_range"
        rows.append({
            "station_id": row["station_id"], "obs_id": row["obs_id"], "time": row["time_dt"],
            "scheme": validation_scheme, "fold_id": fold_id, "repeat_id": repeat_id,
            "sample_role": sample_role, "WSE": wse,
            "width_raw": float(row["width"]), "width_used": np.nan,
            "width_source": "not_used_by_rating_curve", "width_adjustment": np.nan,
            "width_adjustment_percent": np.nan, "slope_raw": finite_number(row.get("slope")),
            "slope_used": np.nan, "slope_was_repaired": False, "area": np.nan,
            "area_position": position, "Q_observed": float(row["Q"]), "Q_calculated": qcalc,
            "n": np.nan, "WSE0": fit["rating_h0"], "width0": np.nan,
            "depth0": fit["depth0"], "A0": np.nan,
        })
    return pd.DataFrame(rows)


def predict_method(method: str, fit: Dict[str, Any], frame: pd.DataFrame,
                   validation_scheme: str, fold_id: int, repeat_id: int,
                   sample_role: str) -> pd.DataFrame:
    if fit.get("model_kind") == "rating_curve" or method == "empirical_rating_curve":
        pred = predict_rating_curve(fit, frame, validation_scheme, fold_id, repeat_id, sample_role)
    else:
        pred = core.predict_frame(
            fit, frame, validation_scheme, fold_id, repeat_id, sample_role,
            use_global_station_monotonic_width=(method != "no_monotonic"),
        )
    if not pred.empty:
        columns=['obs_id']+[c for c in frame if c.startswith('global_')]
        pred=pred.merge(frame[columns],on='obs_id',how='left',validate='one_to_one')
    pred["method"] = method
    pred["method_label"] = METHOD_LABELS[method]
    pred["validation_scheme"] = validation_scheme
    pred["model_kind"] = fit.get("model_kind", "hydraulic")
    return pred


def parameter_row(fit: Dict[str, Any], station_id: str, method: str,
                  validation_scheme: str, fold_id: int, repeat_id: int,
                  train_count: int, validation_count: int) -> Dict[str, Any]:
    excluded = {"anchor_wse", "anchor_width", "train_metrics", "qc"}
    row = {k: v for k, v in fit.items() if k not in excluded}
    if fit.get("model_kind", "hydraulic") == "hydraulic" and len(fit.get("anchor_wse", [])):
        anchor_wse = np.asarray(fit["anchor_wse"], dtype=float)
        anchor_width = np.asarray(fit["anchor_width"], dtype=float)
        span = max(float(np.ptp(anchor_wse)), 0.1)
        depth_upper = min(core.DEPTH_ABSOLUTE_MAX, max(50.0, 20.0 * span))
        min_width_observed = float(fit.get("min_width_observed", anchor_width[0]))
        min_width_lower = float(fit.get(
            "min_width_search_lower",
            max(
                core.WIDTH_FLOOR,
                min_width_observed * core.MIN_WIDTH_FACTOR_LOW_DEFAULT,
            ),
        ))
        min_width_upper = float(fit.get(
            "min_width_search_upper",
            max(
                min_width_lower,
                min_width_observed * core.MIN_WIDTH_FACTOR_HIGH_DEFAULT,
            ),
        ))
        row.update({
            "n_search_lower": core.N_MIN, "n_search_upper": core.N_MAX,
            "depth0_search_lower": core.DEPTH_MIN, "depth0_search_upper": depth_upper,
            "width0_fixed": core.WIDTH0_FIXED,
            "min_width_search_lower": min_width_lower,
            "min_width_search_upper": min_width_upper,
            "A0_search_lower": float(0.5 * min_width_lower * core.DEPTH_MIN),
            "A0_search_upper": float(0.5 * min_width_upper * depth_upper),
        })
    row.update({
        "station_id": station_id, "method": method, "method_label": METHOD_LABELS[method],
        "validation_scheme": validation_scheme, "fold_id": fold_id, "repeat_id": repeat_id,
        "train_count_before_qc": int(train_count), "validation_count": int(validation_count),
        "optimization_unit": "station", "width_rule": (
            "observed_nonnegative" if method == "no_monotonic" else
            "not_used" if method == "empirical_rating_curve" else "global_station_isotonic"
        ),
        **fit.get("qc", {}),
    })
    row.update({f"train_{k}": v for k, v in fit.get("train_metrics", {}).items()})
    return row


def metric_record(pred: pd.DataFrame, station_id: str, method: str,
                  validation_scheme: str, fold_id: int, repeat_id: int,
                  train_count: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "station_id": station_id, "method": method, "validation_scheme": validation_scheme,
        "fold_id": fold_id, "repeat_id": repeat_id, "train_count": int(train_count),
        "validation_count": int(len(pred)),
    }
    row.update(core.calculate_metrics(pred["Q_observed"], pred["Q_calculated"]))
    if len(pred):
        row["above_train_wse_count"] = int((pred["area_position"] == "above_train_max_wse").sum())
        row["below_train_wse_count"] = int((pred["area_position"] == "below_train_min_wse").sum())
    return row


def select_rows(frame: pd.DataFrame, obs_ids: Sequence[str]) -> pd.DataFrame:
    order = {obs_id: i for i, obs_id in enumerate(obs_ids)}
    out = frame[frame["obs_id"].isin(order)].copy()
    out["__order"] = out["obs_id"].map(order)
    return out.sort_values("__order").drop(columns="__order")


def decorate_predictions(pred: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred
    lookup = reference[["obs_id", "time_gap_days", "flow_regime"]].drop_duplicates("obs_id")
    pred = pred.merge(lookup, on="obs_id", how="left")
    obs = pred["Q_observed"].to_numpy(dtype=float)
    sim = pred["Q_calculated"].to_numpy(dtype=float)
    pred["error"] = sim - obs
    pred["absolute_error"] = np.abs(sim - obs)
    pred["relative_error_percent"] = np.where(obs > 0, 100.0 * (sim - obs) / obs, np.nan)
    pred["absolute_relative_error_percent"] = np.abs(pred["relative_error_percent"])
    pred["squared_error"] = (sim - obs) ** 2
    pred["wse_position_class"] = np.where(
        pred["area_position"].eq("within_train_wse_range"), "interpolation", "extrapolation"
    )
    pred["time_gap_class"] = pd.cut(
        pd.to_numeric(pred["time_gap_days"], errors="coerce"),
        bins=[-np.inf, 21, 42, 84, np.inf],
        labels=["<=21 d", "22-42 d", "43-84 d", ">84 d"],
    ).astype(str)
    return pred


def fit_primary_multistart(frame: pd.DataFrame, station_id: str, cfg: RevisionConfig) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    starts = cfg.multistart_count if cfg.run_multistart else 1
    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    core_cfg = make_core_config(cfg, enable_hampel=True, monotonic=True)
    for start_id in range(max(1, starts)):
        fit = core.fit_parameters(
            frame, core_cfg, stable_seed(station_id, "multistart", start_id, base=cfg.seed)
        )
        if fit is None:
            rows.append({"station_id": station_id, "start_id": start_id, "fit_success": False})
            continue
        candidates.append((start_id, fit))
        rows.append({
            "station_id": station_id, "start_id": start_id, "fit_success": True,
            "objective_loss": fit["objective_loss"], "n": fit["n"], "WSE0": fit["WSE0"],
            "depth0": fit["depth0"], "width0": fit["width0"], "A0": fit["A0"],
            "min_width_observed": fit["min_width_observed"],
            "min_width_effective": fit["min_width_effective"],
            "min_width_factor": fit["min_width_factor"],
            "optimizer_success": fit["optimizer_success"], "optimizer_nfev": fit["optimizer_nfev"],
            "optimizer_nit": fit["optimizer_nit"],
            "fit_accepted": fit.get("fit_accepted", False),
            "global_optimizer_success": fit.get("global_optimizer_success", False),
            "local_optimizer_success": fit.get("local_optimizer_success", False),
        })
    if not candidates:
        return None, rows
    best_start, best = min(candidates, key=lambda pair: float(pair[1]["objective_loss"]))
    for row in rows:
        row["selected_best"] = bool(row.get("fit_success") and row["start_id"] == best_start)
    return best, rows


def n_sensitivity_rows(pred: pd.DataFrame, fit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if pred.empty or not np.isfinite(fit.get("n", np.nan)):
        return []
    n_opt = float(fit["n"])
    obs = pred["Q_observed"].to_numpy(dtype=float)
    baseline = pred["Q_calculated"].to_numpy(dtype=float)
    station_id = str(pred["station_id"].iloc[0])
    rows: List[Dict[str, Any]] = []
    for multiplier in RELATIVE_N_MULTIPLIERS:
        requested_multiplier = float(multiplier)
        n_requested = n_opt * requested_multiplier
        n_test = float(np.clip(n_requested, core.N_MIN, core.N_MAX))
        actual_multiplier = n_test / n_opt
        sim = baseline * n_opt / n_test
        metrics = core.calculate_metrics(obs, sim)
        rows.append({
            "station_id": station_id, "grid_type": "relative_to_optimized",
            "n_optimized": n_opt, "n_requested": n_requested, "n_test": n_test,
            "n_multiplier_requested": requested_multiplier,
            "n_multiplier_actual": actual_multiplier,
            # 兼容旧表头，但新汇总不再按该字段分组。
            "n_multiplier": actual_multiplier,
            "n_was_clipped": bool(not np.isclose(
                actual_multiplier, requested_multiplier, rtol=1e-10, atol=1e-12
            )),
            **metrics,
        })
    for n_test in ABSOLUTE_N_GRID:
        sim = baseline * n_opt / n_test
        metrics = core.calculate_metrics(obs, sim)
        rows.append({
            "station_id": station_id, "grid_type": "absolute_grid",
            "n_optimized": n_opt, "n_requested": n_test, "n_test": n_test,
            "n_multiplier_requested": np.nan,
            "n_multiplier_actual": n_test / n_opt,
            "n_multiplier": n_test / n_opt,
            "n_was_clipped": False,
            **metrics,
        })
    return rows


def record_split_assignment(result, train, test, scheme, repeat_id, fold_id):
    cols=['station_id','obs_id','time_dt','global_qc_pass','global_qc_reason']
    for role,frame in [('training',train),('validation',test)]:
        part=frame[[c for c in cols if c in frame]].copy()
        part['role']=role;part['validation_scheme']=scheme;part['repeat_id']=repeat_id;part['fold_id']=fold_id
        result['split_audit'].append(part)


def process_station(path_text: str, meta: Dict[str, Any], global_reference_n: float,
                    cfg: RevisionConfig) -> Dict[str, Any]:
    path = Path(path_text)
    station_id = core.normalize_station_id(path.stem)
    result: Dict[str, Any] = {
        "statistics": None, "parameters": [], "predictions": [], "fold_metrics": [],
        "failures": [], "raw_audit": None, "overpass_audit": None,
        "multistart": [], "n_sensitivity": [], "split_audit": [],
    }
    try:
        base, audit, raw_audit, overpass_audit = load_station_with_audit(
            path, cfg.aggregate_nodes_by_overpass
        )
        versions, monotonic_diag = make_frame_versions(base)
        primary = versions[PRIMARY_METHOD]
        audit.update(monotonic_diag)
        result["statistics"] = core.station_statistics(primary, audit, meta, enable_hampel=True)
        result["statistics"].update({
            "geographic_region": geographic_region(meta.get("Lon_web"), meta.get("Lat_web")),
            "basin_code": meta.get("basin_code", ""),
            "basin_name": meta.get("basin_name", "Unknown"),
            "basin_match_method": meta.get("basin_match_method", "unavailable"),
            "median_time_gap_days": float(primary["time_gap_days"].median()) if len(primary) else np.nan,
            "max_time_gap_days": float(primary["time_gap_days"].max()) if len(primary) else np.nan,
            "q95_sample_count": int((primary["flow_regime"] == "Q95-100").sum()) if len(primary) else 0,
            "eligible_for_calibration": bool(len(primary) >= core.MIN_TRAIN_RECORDS),
        })
        for audit_frame in (raw_audit, overpass_audit):
            audit_frame["station_type"] = meta.get("type", "")
            audit_frame["longitude"] = finite_number(meta.get("Lon_web"))
            audit_frame["latitude"] = finite_number(meta.get("Lat_web"))
            audit_frame["geographic_region"] = geographic_region(meta.get("Lon_web"), meta.get("Lat_web"))
            audit_frame["basin_code"] = meta.get("basin_code", "")
            audit_frame["basin_name"] = meta.get("basin_name", "Unknown")
            audit_frame["basin_match_method"] = meta.get("basin_match_method", "unavailable")
        result["raw_audit"] = raw_audit
        result["overpass_audit"] = overpass_audit

        if len(primary) < core.MIN_TRAIN_RECORDS:
            result["failures"].append({
                "station_id": station_id, "method": PRIMARY_METHOD,
                "validation_scheme": "all", "reason": "insufficient_records",
            })
            return result

        n_meta = finite_number(meta.get("N_River"))
        n_reference = n_meta if np.isfinite(n_meta) and n_meta > 0 else global_reference_n
        n_reference = float(np.clip(n_reference, core.N_MIN, core.N_MAX))

        full_fit, multistart_rows = fit_primary_multistart(primary, station_id, cfg)
        result["multistart"].extend(multistart_rows)
        if full_fit is None:
            result["failures"].append({
                "station_id": station_id, "method": PRIMARY_METHOD,
                "validation_scheme": "station_reconstruction", "reason": "fit_failed",
            })
        else:
            pred = predict_method(
                PRIMARY_METHOD, full_fit, primary, "station_reconstruction", 0, 0, "reconstruction"
            )
            pred = decorate_predictions(pred, primary)
            result["predictions"].append(pred)
            result["parameters"].append(parameter_row(
                full_fit, station_id, PRIMARY_METHOD, "station_reconstruction", 0, 0,
                len(primary), len(primary),
            ))
            result["fold_metrics"].append(metric_record(
                pred, station_id, PRIMARY_METHOD, "station_reconstruction", 0, 0, len(primary)
            ))
            if cfg.run_n_sensitivity:
                result["n_sensitivity"].extend(n_sensitivity_rows(pred, full_fit))

        # 消融版本的全样本复现：只用于拟合能力与参数/QC差异，不当作独立预测证据。
        if cfg.run_ablation and cfg.run_ablation_reconstruction:
            for method in ABLATION_METHODS:
                frame = versions[method]
                fit = fit_method(
                    method, frame, cfg,
                    stable_seed(station_id, method, "reconstruction", base=cfg.seed), n_reference,
                )
                if fit is None:
                    result["failures"].append({
                        "station_id": station_id, "method": method,
                        "validation_scheme": "station_reconstruction", "reason": "fit_failed",
                    })
                    continue
                pred = decorate_predictions(predict_method(
                    method, fit, frame, "station_reconstruction", 0, 0, "reconstruction"
                ), base)
                result["predictions"].append(pred)
                result["parameters"].append(parameter_row(
                    fit, station_id, method, "station_reconstruction", 0, 0, len(frame), len(frame)
                ))
                result["fold_metrics"].append(metric_record(
                    pred, station_id, method, "station_reconstruction", 0, 0, len(frame)
                ))

        # All methods share all ten assignments; paired metrics use identical retained test observations.
        splits: Dict[int, Tuple[List[str], List[str]]] = {}
        split_versions: Dict[
            int, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
        ] = {}
        for repeat_id in range(cfg.repeat_count):
            split = core.wse_stratified_split(
                base, 0.70, 5,
                stable_seed(station_id, "wse_split", repeat_id, base=cfg.seed),
            )
            if split is None:
                result["failures"].append({
                    "station_id": station_id, "method": PRIMARY_METHOD,
                    "validation_scheme": "wse_stratified", "repeat_id": repeat_id,
                    "reason": "split_failed",
                })
                continue
            raw_train0, raw_test0 = split
            record_split_assignment(result,raw_train0,raw_test0,'wse_stratified',repeat_id,0)
            train_ids = raw_train0["obs_id"].tolist()
            test_ids = raw_test0["obs_id"].tolist()
            splits[repeat_id] = (train_ids, test_ids)
            fold_versions, _ = make_split_frame_versions(
                base, train_ids, test_ids
            )
            split_versions[repeat_id] = fold_versions
            train0, test0 = fold_versions[PRIMARY_METHOD]
            if test0.empty:
                result["failures"].append({"station_id":station_id,"method":PRIMARY_METHOD,"validation_scheme":"wse_stratified","repeat_id":repeat_id,"reason":"no_validation_after_global_qc"})
                continue
            fit = fit_method(
                PRIMARY_METHOD, train0, cfg,
                stable_seed(station_id, PRIMARY_METHOD, "wse_fit", repeat_id, base=cfg.seed),
                n_reference,
            )
            if fit is None:
                result["failures"].append({
                    "station_id": station_id, "method": PRIMARY_METHOD,
                    "validation_scheme": "wse_stratified", "repeat_id": repeat_id,
                    "reason": "fit_failed",
                })
                continue
            pred = decorate_predictions(predict_method(
                PRIMARY_METHOD, fit, test0, "wse_stratified", 0, repeat_id, "validation"
            ), base)
            result["predictions"].append(pred)
            result["parameters"].append(parameter_row(
                fit, station_id, PRIMARY_METHOD, "wse_stratified", 0, repeat_id,
                len(train0), len(test0),
            ))
            result["fold_metrics"].append(metric_record(
                pred, station_id, PRIMARY_METHOD, "wse_stratified", 0, repeat_id, len(train0)
            ))

        if cfg.run_ablation:
            for repeat_id in range(min(cfg.ablation_repeat_count, cfg.repeat_count)):
                if repeat_id not in splits:
                    continue
                fold_versions = split_versions[repeat_id]
                for method in ABLATION_METHODS:
                    train, test = fold_versions[method]
                    if test.empty:
                        result["failures"].append({"station_id":station_id,"method":method,"validation_scheme":"wse_stratified","repeat_id":repeat_id,"reason":"no_validation_after_global_qc"})
                        continue
                    fit = fit_method(
                        method, train, cfg,
                        stable_seed(station_id, PRIMARY_METHOD, "wse_fit", repeat_id, base=cfg.seed),
                        n_reference,
                    )
                    if fit is None:
                        result["failures"].append({
                            "station_id": station_id, "method": method,
                            "validation_scheme": "wse_stratified", "repeat_id": repeat_id,
                            "reason": "fit_failed",
                        })
                        continue
                    pred = decorate_predictions(predict_method(
                        method, fit, test, "wse_stratified", 0, repeat_id, "validation"
                    ), base)
                    result["predictions"].append(pred)
                    result["parameters"].append(parameter_row(
                        fit, station_id, method, "wse_stratified", 0, repeat_id,
                        len(train), len(test),
                    ))
                    result["fold_metrics"].append(metric_record(
                        pred, station_id, method, "wse_stratified", 0, repeat_id, len(train)
                    ))

        if cfg.run_temporal:
            temporal_splits = core.temporal_forward_splits(base, cfg.temporal_folds, initial_fraction=cfg.temporal_train_fraction)
            if not temporal_splits:
                result["failures"].append({
                    "station_id": station_id, "method": PRIMARY_METHOD,
                    "validation_scheme": "temporal_forward", "reason": "split_failed",
                })
            for fold_id, raw_train, raw_test in temporal_splits:
                record_split_assignment(result,raw_train,raw_test,"temporal_forward",0,fold_id)
                temporal_versions, _ = make_split_frame_versions(
                    base,
                    raw_train["obs_id"].tolist(),
                    raw_test["obs_id"].tolist(),
                )
                train, test = temporal_versions[PRIMARY_METHOD]
                if test.empty:
                    result["failures"].append({"station_id":station_id,"method":PRIMARY_METHOD,"validation_scheme":"temporal_forward","fold_id":fold_id,"reason":"no_validation_after_global_qc"})
                    continue
                fit = fit_method(
                    PRIMARY_METHOD, train, cfg,
                    stable_seed(station_id, PRIMARY_METHOD, "temporal", fold_id, base=cfg.seed),
                    n_reference,
                )
                if fit is None:
                    result["failures"].append({
                        "station_id": station_id, "method": PRIMARY_METHOD,
                        "validation_scheme": "temporal_forward", "fold_id": fold_id,
                        "reason": "fit_failed",
                    })
                    continue
                pred = decorate_predictions(predict_method(
                    PRIMARY_METHOD, fit, test, "temporal_forward", fold_id, 0, "validation"
                ), base)
                result["predictions"].append(pred)
                result["parameters"].append(parameter_row(
                    fit, station_id, PRIMARY_METHOD, "temporal_forward", fold_id, 0,
                    len(train), len(test),
                ))
                result["fold_metrics"].append(metric_record(
                    pred, station_id, PRIMARY_METHOD, "temporal_forward", fold_id, 0, len(train)
                ))
    except Exception as exc:
        result["failures"].append({
            "station_id": station_id, "method": "station_worker", "validation_scheme": "all",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=10),
        })
    return result


def build_station_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    keys = ["station_id", "method", "validation_scheme", "repeat_id"]
    for key, group in predictions.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["fold_count"] = int(group["fold_id"].nunique())
        row.update(core.calculate_metrics(group["Q_observed"], group["Q_calculated"]))
        row["interpolation_count"] = int((group["wse_position_class"] == "interpolation").sum())
        row["extrapolation_count"] = int((group["wse_position_class"] == "extrapolation").sum())
        row["extreme_flow_count"] = int((group["flow_regime"] == "Q95-100").sum())
        row["peak_observed"] = float(group["Q_observed"].max())
        peak_row = group.loc[group["Q_observed"].idxmax()]
        row["peak_calculated_at_observed_peak"] = float(peak_row["Q_calculated"])
        row["peak_error_percent"] = float(peak_row["relative_error_percent"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_overall_summary(predictions: pd.DataFrame, station_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    keys = ["method", "validation_scheme", "repeat_id"]
    for key, group in predictions.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update({
            "method_label": METHOD_LABELS.get(row["method"], row["method"]),
            "station_count": int(group["station_id"].nunique()),
            "prediction_count": int(len(group)),
        })
        row.update({f"pooled_{k}": v for k, v in core.calculate_metrics(
            group["Q_observed"], group["Q_calculated"]
        ).items()})
        station_group = station_metrics[
            (station_metrics["method"] == row["method"]) &
            (station_metrics["validation_scheme"] == row["validation_scheme"]) &
            (station_metrics["repeat_id"] == row["repeat_id"])
        ]
        for metric in METRICS:
            values = pd.to_numeric(station_group.get(metric), errors="coerce").dropna()
            row[f"station_median_{metric}"] = float(values.median()) if len(values) else np.nan
            row[f"station_mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"station_valid_{metric}_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty or group_column not in predictions:
        return pd.DataFrame()
    keys = ["method", "validation_scheme", "repeat_id", group_column]
    for key, group in predictions.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["station_count"] = int(group["station_id"].nunique())
        row["prediction_count"] = int(len(group))
        row.update(core.calculate_metrics(group["Q_observed"], group["Q_calculated"]))
        station_rows: List[Dict[str, Any]] = []
        for _, station_group in group.groupby("station_id"):
            station_rows.append(core.calculate_metrics(
                station_group["Q_observed"], station_group["Q_calculated"]
            ))
        station_frame = pd.DataFrame(station_rows)
        for metric in METRICS:
            values = pd.to_numeric(station_frame.get(metric), errors="coerce").dropna()
            row[f"station_median_{metric}"] = (
                float(values.median()) if len(values) else np.nan
            )
            row[f"station_valid_{metric}_count"] = int(len(values))
        row["median_absolute_relative_error_percent"] = float(
            pd.to_numeric(group["absolute_relative_error_percent"], errors="coerce").median()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_peak_metrics(station_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "station_id", "method", "validation_scheme", "repeat_id",
        "peak_observed", "peak_calculated_at_observed_peak", "peak_error_percent",
        "extreme_flow_count",
    ]
    return station_metrics[[c for c in columns if c in station_metrics]].copy()


def build_repeat_uncertainty(station_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "validation_scheme", "station_id", "repeat_id"}
    if station_metrics.empty or not required.issubset(station_metrics.columns):
        return pd.DataFrame()
    data = station_metrics[
        (station_metrics["method"] == PRIMARY_METHOD) &
        (station_metrics["validation_scheme"] == "wse_stratified")
    ]
    if data.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for station_id, group in data.groupby("station_id"):
        row: Dict[str, Any] = {
            "station_id": station_id,
            "repeat_count": int(group["repeat_id"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_q05"] = float(values.quantile(0.05)) if len(values) else np.nan
            row[f"{metric}_q95"] = float(values.quantile(0.95)) if len(values) else np.nan
            row[f"{metric}_valid_repeat_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_intervals(station_metrics: pd.DataFrame, cfg: RevisionConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, Any]] = []
    required = {"station_id", "method", "validation_scheme", *METRICS}
    if station_metrics.empty or not required.issubset(station_metrics.columns):
        return pd.DataFrame()
    # 重复验证先在站点内取中位数，避免将 10 次同站结果当作 10 个独立站点。
    collapsed = station_metrics.groupby(
        ["station_id", "method", "validation_scheme"], as_index=False
    )[list(METRICS)].median(numeric_only=True)
    for (method, validation), group in collapsed.groupby(["method", "validation_scheme"]):
        for metric in ("R2", "NSE", "KGE", "logNSE", "RMSE", "MAE", "PBIAS"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) < 2:
                continue
            samples = rng.choice(values, size=(cfg.bootstrap_iterations, len(values)), replace=True)
            for statistic, function in (("median", np.nanmedian), ("mean", np.nanmean)):
                distribution = function(samples, axis=1)
                rows.append({
                    "method": method, "validation_scheme": validation, "metric": metric,
                    "statistic": statistic, "station_count": len(values),
                    "point_estimate": float(function(values)),
                    "ci_lower_2p5": float(np.quantile(distribution, 0.025)),
                    "ci_upper_97p5": float(np.quantile(distribution, 0.975)),
                    "bootstrap_iterations": cfg.bootstrap_iterations,
                })
    return pd.DataFrame(rows)


def paired_ablation_tests(station_metrics: pd.DataFrame, cfg: RevisionConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "station_id", "method", "validation_scheme", "repeat_id", *METRICS
    }
    if station_metrics.empty or not required.issubset(station_metrics.columns):
        return pd.DataFrame(), pd.DataFrame()
    data = station_metrics[
        (station_metrics["validation_scheme"] == "wse_stratified") &
        (station_metrics["repeat_id"] < cfg.ablation_repeat_count)
    ].copy()
    full = data[data["method"] == PRIMARY_METHOD]
    raw_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.seed + 19)
    for comparator in ABLATION_METHODS:
        comp = data[data["method"] == comparator]
        merged = full.merge(comp, on=["station_id", "repeat_id"], suffixes=("_full", "_comparator"))
        for metric in ("R2", "NSE", "KGE", "RMSE", "MAE", "PBIAS"):
            a = pd.to_numeric(merged[f"{metric}_full"], errors="coerce")
            b = pd.to_numeric(merged[f"{metric}_comparator"], errors="coerce")
            if metric == "PBIAS":
                full_value, comp_value = a.abs(), b.abs()
                advantage = comp_value - full_value
                comparison_definition = "abs(comparator) - abs(full); positive favors full"
            elif metric in HIGHER_IS_BETTER:
                full_value, comp_value = a, b
                advantage = full_value - comp_value
                comparison_definition = "full - comparator; positive favors full"
            else:
                full_value, comp_value = a, b
                advantage = comp_value - full_value
                comparison_definition = "comparator - full; positive favors full"
            valid = np.isfinite(full_value) & np.isfinite(comp_value) & np.isfinite(advantage)
            values = advantage[valid].to_numpy(dtype=float)
            station_ids = merged.loc[valid, "station_id"].astype(str).to_numpy()
            repeat_ids = merged.loc[valid, "repeat_id"].astype(int).to_numpy()
            for sid, rid, fval, cval, adv in zip(
                station_ids, repeat_ids, full_value[valid], comp_value[valid], advantage[valid]
            ):
                raw_rows.append({
                    "station_id": sid, "repeat_id": rid,
                    "comparator": comparator, "metric": metric,
                    "full_value": float(fval), "comparator_value": float(cval),
                    "full_advantage": float(adv), "definition": comparison_definition,
                })
            if len(values) == 0:
                continue
            # Repeated splits from the same station are correlated. Treat each
            # station's median paired effect as one independent observation.
            station_values = (
                pd.DataFrame({"station_id": station_ids, "advantage": values})
                .groupby("station_id", sort=True)["advantage"].median()
                .to_numpy(dtype=float)
            )
            if len(station_values) > 1 and np.any(np.abs(station_values) > 0):
                try:
                    p_value = float(wilcoxon(station_values, alternative="two-sided", zero_method="wilcox").pvalue)
                except ValueError:
                    p_value = np.nan
            else:
                p_value = np.nan
            boot = np.median(
                rng.choice(station_values, size=(cfg.bootstrap_iterations, len(station_values)), replace=True), axis=1
            )
            summaries.append({
                "comparator": comparator, "metric": metric,
                "paired_station_count": len(station_values),
                "paired_station_repeat_count": len(values),
                "station_repeat_aggregation": "median advantage within station",
                "median_full_advantage": float(np.median(station_values)),
                "mean_full_advantage": float(np.mean(station_values)),
                "full_better_fraction": float(np.mean(station_values > 0)),
                "ci_lower_2p5": float(np.quantile(boot, 0.025)),
                "ci_upper_97p5": float(np.quantile(boot, 0.975)),
                "wilcoxon_p_value": p_value, "definition": comparison_definition,
            })
    return pd.DataFrame(raw_rows), pd.DataFrame(summaries)


def multistart_summary(multistart: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if multistart.empty:
        return pd.DataFrame()
    valid = multistart[multistart["fit_success"] == True].copy()  # noqa: E712
    for station_id, group in valid.groupby("station_id"):
        row: Dict[str, Any] = {
            "station_id": station_id, "successful_start_count": len(group),
            "all_optimizer_success": bool(group["optimizer_success"].fillna(False).all()),
            "all_fit_accepted": bool(group["fit_accepted"].fillna(False).all())
            if "fit_accepted" in group else False,
        }
        for column in (
            "objective_loss", "n", "WSE0", "depth0",
            "min_width_effective", "min_width_factor", "A0",
        ):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{column}_max"] = float(values.max()) if len(values) else np.nan
            row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
            denom = abs(float(values.mean())) if len(values) else np.nan
            row[f"{column}_cv"] = float(values.std(ddof=0) / denom) if len(values) > 1 and denom > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def n_sensitivity_overall(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame()
    absolute = data[data["grid_type"] == "absolute_grid"]
    for (grid_type, value), group in absolute.groupby(["grid_type", "n_test"], dropna=False):
        row: Dict[str, Any] = {
            "grid_type": grid_type, "n_test": value,
            "station_count": int(group["station_id"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            row[f"station_median_{metric}"] = float(values.median()) if len(values) else np.nan
            row[f"station_q25_{metric}"] = float(values.quantile(0.25)) if len(values) else np.nan
            row[f"station_q75_{metric}"] = float(values.quantile(0.75)) if len(values) else np.nan
        rows.append(row)
    # 必须按“请求倍率”汇总。实际倍率可能因 n 的硬边界而被裁剪。
    relative = data[data["grid_type"] == "relative_to_optimized"]
    group_column = (
        "n_multiplier_requested"
        if "n_multiplier_requested" in relative.columns
        else "n_multiplier"
    )
    for multiplier, group in relative.groupby(group_column):
        actual = pd.to_numeric(
            group.get("n_multiplier_actual", group.get("n_multiplier")),
            errors="coerce",
        )
        clipped = (
            group["n_was_clipped"].astype(str).str.lower().isin(["true", "1"])
            if "n_was_clipped" in group
            else pd.Series(False, index=group.index)
        )
        row = {
            "grid_type": "relative_multiplier_summary", "n_test": np.nan,
            "n_multiplier_requested": multiplier,
            "n_multiplier": multiplier,
            "actual_multiplier_median": float(actual.median()) if actual.notna().any() else np.nan,
            "actual_multiplier_q25": float(actual.quantile(0.25)) if actual.notna().any() else np.nan,
            "actual_multiplier_q75": float(actual.quantile(0.75)) if actual.notna().any() else np.nan,
            "clipped_station_fraction": float(clipped.mean()),
            "station_count": int(group["station_id"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            row[f"station_median_{metric}"] = float(values.median()) if len(values) else np.nan
            row[f"station_q25_{metric}"] = float(values.quantile(0.25)) if len(values) else np.nan
            row[f"station_q75_{metric}"] = float(values.quantile(0.75)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def retention_tables(statistics: pd.DataFrame, parameters: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if statistics.empty:
        return pd.DataFrame(), pd.DataFrame()
    stats = statistics.copy()
    full_params = parameters[
        (parameters["method"] == PRIMARY_METHOD) &
        (parameters["validation_scheme"] == "station_reconstruction")
    ][["station_id", "train_after_hampel_count"]].drop_duplicates("station_id")
    stats = stats.merge(full_params, on="station_id", how="left")
    stats["calibrated_station"] = stats["train_after_hampel_count"].notna()
    stage_rows = [
        {"stage": "Raw node rows", "count": int(stats["raw_row_count"].sum())},
        {"stage": "Basic-QC valid node rows", "count": int(stats["valid_row_count_before_overpass_aggregation"].sum())},
        {"stage": "Independent station-overpasses", "count": int(stats["overpass_count"].sum())},
        {"stage": "Overpasses at calibratable stations", "count": int(stats.loc[stats["calibrated_station"], "overpass_count"].sum())},
        {"stage": "Rows used by full calibration after Hampel", "count": int(stats["train_after_hampel_count"].fillna(0).sum())},
    ]
    stage = pd.DataFrame(stage_rows)
    stage["fraction_of_raw"] = stage["count"] / max(stage["count"].iloc[0], 1)

    group_rows: List[Dict[str, Any]] = []
    for group_col in ("station_type", "geographic_region", "basin_name"):
        if group_col not in stats:
            continue
        for group_name, group in stats.groupby(group_col, dropna=False):
            raw = float(group["raw_row_count"].sum())
            valid = float(group["valid_row_count_before_overpass_aggregation"].sum())
            overpass = float(group["overpass_count"].sum())
            eligible = float(group.loc[group["calibrated_station"], "overpass_count"].sum())
            group_rows.append({
                "group_variable": group_col, "group": group_name,
                "station_file_count": int(len(group)),
                "calibrated_station_count": int(group["calibrated_station"].sum()),
                "raw_row_count": int(raw), "valid_node_count": int(valid),
                "overpass_count": int(overpass), "eligible_overpass_count": int(eligible),
                "basic_qc_retention_rate": valid / raw if raw else np.nan,
                "overpass_per_valid_node_rate": overpass / valid if valid else np.nan,
                "eligible_overpass_rate": eligible / overpass if overpass else np.nan,
            })
    return stage, pd.DataFrame(group_rows)


def qc_reason_tables(raw_audit: pd.DataFrame, overpass_audit: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reason_rows: List[Dict[str, Any]] = []
    if not raw_audit.empty:
        exploded = raw_audit.copy()
        exploded["reason"] = exploded["basic_qc_reason"].fillna("unknown").astype(str).str.split(";")
        exploded = exploded.explode("reason")
        for group_variable in (None, "station_type", "geographic_region", "basin_name"):
            if group_variable is not None and group_variable not in raw_audit:
                continue
            keys = ["reason"] if group_variable is None else [group_variable, "reason"]
            for key, group in exploded.groupby(keys, dropna=False):
                if group_variable is None:
                    group_name, reason = "All", key[0] if isinstance(key, tuple) else key
                    variable = "overall"
                else:
                    group_name, reason = key
                    variable = group_variable
                reason_rows.append({
                    "group_variable": variable, "group": group_name, "reason": reason,
                    "raw_row_count": int(len(group)),
                    "station_count": int(group["station_id"].nunique()),
                })
    qc_rows: List[Dict[str, Any]] = []
    if not overpass_audit.empty:
        flags = [
            "hampel_wse_candidate", "hampel_width_candidate", "hampel_slope_candidate",
            "slope_requires_repair", "primary_training_qc_pass",
        ]
        for group_variable in (
            None, "flow_regime", "station_type", "geographic_region", "basin_name",
        ):
            if group_variable is not None and group_variable not in overpass_audit:
                continue
            groups = [("All", overpass_audit)] if group_variable is None else overpass_audit.groupby(group_variable, dropna=False)
            for group_name, group in groups:
                for flag in flags:
                    values = group[flag].astype(str).str.lower().isin(["true", "1"])
                    qc_rows.append({
                        "group_variable": "overall" if group_variable is None else group_variable,
                        "group": group_name, "flag": flag, "overpass_count": int(len(group)),
                        "flagged_count": int(values.sum()), "flagged_fraction": float(values.mean()),
                        "station_count": int(group["station_id"].nunique()),
                    })
    return pd.DataFrame(reason_rows), pd.DataFrame(qc_rows)


def performance_driver_tables(station_metrics: pd.DataFrame, statistics: pd.DataFrame,
                              parameters: pd.DataFrame, predictions: pd.DataFrame,
                              cfg: RevisionConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = station_metrics[
        (station_metrics["method"] == PRIMARY_METHOD) &
        (station_metrics["validation_scheme"] == "wse_stratified")
    ].groupby("station_id", as_index=False)[list(METRICS) + ["peak_error_percent"]].median(numeric_only=True)
    params = parameters[
        (parameters["method"] == PRIMARY_METHOD) &
        (parameters["validation_scheme"] == "station_reconstruction")
    ].drop_duplicates("station_id")
    keep_param = [c for c in (
        "station_id", "n", "WSE0", "depth0", "width0", "A0",
        "min_width_observed", "min_width_effective", "min_width_factor",
        "n_at_lower_bound", "n_at_upper_bound", "depth_at_lower_bound",
        "depth_at_upper_bound", "min_width_at_lower_bound", "min_width_at_upper_bound",
        "objective_loss", "optimizer_nfev", "optimizer_nit", "train_after_hampel_count",
    ) if c in params]
    stats = statistics.copy()
    stats["basic_qc_retention_rate"] = stats["valid_row_count_before_overpass_aggregation"] / stats["raw_row_count"].replace(0, np.nan)
    stats["overpass_per_valid_node_rate"] = stats["overpass_count"] / stats["valid_row_count_before_overpass_aggregation"].replace(0, np.nan)
    stats["hampel_candidate_rate"] = (
        pd.to_numeric(stats.get("hampel_wse_model_candidate_count", 0), errors="coerce").fillna(0) +
        pd.to_numeric(stats.get("hampel_width_candidate_count", 0), errors="coerce").fillna(0)
    ) / stats["overpass_count"].replace(0, np.nan)
    primary_pred = predictions[
        (predictions["method"] == PRIMARY_METHOD) &
        (predictions["validation_scheme"] == "wse_stratified")
    ]
    extra = primary_pred.groupby("station_id").agg(
        extrapolation_fraction=("wse_position_class", lambda x: float(np.mean(x == "extrapolation"))),
        median_abs_relative_error=("absolute_relative_error_percent", "median"),
        mean_time_gap_days=("time_gap_days", "mean"),
    ).reset_index() if len(primary_pred) else pd.DataFrame(columns=["station_id"])
    drivers = target.merge(stats, on="station_id", how="left").merge(
        params[keep_param], on="station_id", how="left", suffixes=("", "_parameter")
    ).merge(extra, on="station_id", how="left")

    feature_candidates = [
        "overpass_count", "wse_range", "q_range", "width_mean", "slope_mean",
        "nonpositive_slope_count", "median_time_gap_days", "max_time_gap_days",
        "hampel_candidate_rate", "global_width_adjustment_median_abs",
        "global_width_adjustment_max_abs", "raw_width_wse_decrease_count",
        "basic_qc_retention_rate", "overpass_per_valid_node_rate",
        "extrapolation_fraction", "q95_sample_count", "n", "depth0",
        "min_width_effective", "min_width_factor", "A0",
    ]
    features = [c for c in feature_candidates if c in drivers]
    corr_rows: List[Dict[str, Any]] = []
    for target_metric in ("R2", "NSE", "RMSE", "MAE", "peak_error_percent"):
        if target_metric not in drivers:
            continue
        for feature in features:
            x = pd.to_numeric(drivers[feature], errors="coerce")
            y = pd.to_numeric(drivers[target_metric], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                continue
            rho, p_value = spearmanr(x[mask], y[mask])
            corr_rows.append({
                "target_metric": target_metric, "feature": feature,
                "station_count": int(mask.sum()), "spearman_rho": float(rho),
                "p_value": float(p_value),
            })
    correlations = pd.DataFrame(corr_rows)

    return drivers, correlations


def parse_args() -> RevisionConfig:
    parser = argparse.ArgumentParser(description="WRR reviewer-response SWOT experiments")
    parser.add_argument("--data-dir", default=os.environ.get("SWOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
    parser.add_argument("--metadata-file", default=os.environ.get("SWOT_METADATA_FILE", str(DEFAULT_METADATA_FILE)))
    parser.add_argument("--output-dir", default=os.environ.get("SWOT_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--ablation-repeat-count", type=int, default=10)
    parser.add_argument("--temporal-folds", type=int, default=3)
    parser.add_argument("--multistart-count", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-ablation-reconstruction", action="store_true")
    parser.add_argument("--no-n-sensitivity", action="store_true")
    parser.add_argument("--no-multistart", action="store_true")
    parser.add_argument(
        "--min-width-factor-low", type=float,
        default=core.MIN_WIDTH_FACTOR_LOW_DEFAULT,
    )
    parser.add_argument(
        "--min-width-factor-high", type=float,
        default=core.MIN_WIDTH_FACTOR_HIGH_DEFAULT,
    )
    parser.add_argument("--optimizer-maxiter", type=int, default=60)
    parser.add_argument("--optimizer-popsize", type=int, default=8)
    parser.add_argument("--optimizer-tol", type=float, default=1e-3)
    parser.add_argument("--local-optimizer-maxiter", type=int, default=400)
    parser.add_argument(
        "--basin-shapefile",
        default=os.environ.get("SWOT_BASIN_SHAPEFILE", str(DEFAULT_BASIN_SHAPEFILE)),
    )
    parser.add_argument("--objective", choices=("old_composite", "normalized_mse", "log_mse"), default="old_composite")
    args = parser.parse_args()
    if args.min_width_factor_low <= 0:
        parser.error("--min-width-factor-low must be greater than 0")
    if args.min_width_factor_high <= args.min_width_factor_low:
        parser.error("--min-width-factor-high must exceed --min-width-factor-low")
    return RevisionConfig(
        data_dir=args.data_dir, metadata_file=args.metadata_file, output_dir=args.output_dir,
        repeat_count=max(1, args.repeat_count),
        ablation_repeat_count=max(1, min(args.ablation_repeat_count, args.repeat_count)),
        temporal_folds=max(1, args.temporal_folds),
        multistart_count=max(1, args.multistart_count),
        bootstrap_iterations=max(200, args.bootstrap_iterations),
        workers=max(1, args.workers), use_parallel=not args.serial,
        max_stations=args.max_stations, fast=bool(args.fast), objective=args.objective,
        run_temporal=not args.no_temporal, run_ablation=not args.no_ablation,
        run_ablation_reconstruction=not args.no_ablation_reconstruction,
        run_n_sensitivity=not args.no_n_sensitivity,
        run_multistart=not args.no_multistart,
        min_width_factor_low=float(args.min_width_factor_low),
        min_width_factor_high=float(args.min_width_factor_high),
        optimizer_maxiter=max(1, int(args.optimizer_maxiter)),
        optimizer_popsize=max(5, int(args.optimizer_popsize)),
        optimizer_tol=max(float(args.optimizer_tol), 1e-10),
        local_optimizer_maxiter=max(50, int(args.local_optimizer_maxiter)),
        basin_shapefile=str(args.basin_shapefile),
        run_spatial_cv=False,
    )


def main() -> int:
    cfg = parse_args()
    data_dir, metadata_file, output_dir = Path(cfg.data_dir), Path(cfg.metadata_file), Path(cfg.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_configuration.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)
    with open(output_dir / "software_environment.json", "w", encoding="utf-8") as handle:
        json.dump({
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "core_script": str((SCRIPT_DIR / "swot_n_wse0_width0_nonnegative_cv.py").resolve()),
            "spatial_cross_validation": "disabled_by_design",
        }, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "method_and_optimizer_definition.json", "w", encoding="utf-8") as handle:
        json.dump({
            "primary_method": (
                "station-specific optimization of n, WSE0/depth0, and effective "
                "width at minimum observed WSE; width0 fixed at zero"
            ),
            "objective": cfg.objective,
            "optimizer": "differential_evolution global search followed by bounded Powell",
            "optimizer_stopping_rule": {
                "tol": cfg.optimizer_tol,
                "maxiter_primary": 10 if cfg.fast else cfg.optimizer_maxiter,
                "popsize_primary": 5 if cfg.fast else cfg.optimizer_popsize,
                "local_maxiter": 80 if cfg.fast else cfg.local_optimizer_maxiter,
                "workers_inside_optimizer": 1,
                "selection": "lowest finite objective loss across deterministic starts",
            },
            "parameter_bounds": {
                "n": [core.N_MIN, core.N_MAX],
                "n_typical_diagnostic_range": [core.N_TYPICAL_MIN, core.N_TYPICAL_MAX],
                "depth0_lower": core.DEPTH_MIN,
                "depth0_upper_per_fit": "min(5000 m, max(50 m, 20 * training WSE range))",
                "width0": 0.0,
                "minimum_wse_effective_width_factor": [
                    cfg.min_width_factor_low, cfg.min_width_factor_high,
                ],
                "minimum_wse_effective_width_upper_monotonic_guard": (
                    "not greater than the second monotonic width anchor"
                ),
                "A0_bounds": "written explicitly for every fit in parameter_estimates.csv",
            },
            "hampel": {"window": core.HAMPEL_WINDOW, "n_sigmas": core.HAMPEL_N_SIGMAS},
            "minimum_training_records": core.MIN_TRAIN_RECORDS,
            "wse_validation": {
                "strata": 5, "training_fraction": 0.70, "repeat_count": cfg.repeat_count,
                "ablation_repeat_count": cfg.ablation_repeat_count,
            },
            "temporal_validation": {"folds": cfg.temporal_folds, "expanding_window": True},
            "spatial_cross_validation": False,
        }, handle, ensure_ascii=False, indent=2)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    files = sorted(data_dir.glob("*.csv"))
    if cfg.max_stations is not None:
        files = files[:max(0, cfg.max_stations)]
    if not files:
        raise FileNotFoundError(f"No station CSV files found: {data_dir}")

    metadata = core.metadata_lookup(metadata_file)
    metadata = attach_basin_attributes(metadata, Path(cfg.basin_shapefile))
    n_values = [
        finite_number(row.get("N_River")) for row in metadata.values()
        if np.isfinite(finite_number(row.get("N_River"))) and finite_number(row.get("N_River")) > 0
    ]
    global_reference_n = float(np.median(n_values)) if n_values else 0.035
    global_reference_n = float(np.clip(global_reference_n, core.N_MIN, core.N_MAX))
    jobs = [
        (str(path), metadata.get(core.normalize_station_id(path.stem), {}), global_reference_n, cfg)
        for path in files
    ]
    print(f"Stations: {len(jobs)}", flush=True)
    print(f"WSE repeated validation: {cfg.repeat_count}; spatial CV: disabled; full-record global QC", flush=True)
    print(f"Ablation repeats: {cfg.ablation_repeat_count}; multistart: {cfg.multistart_count if cfg.run_multistart else 1}", flush=True)

    results: List[Dict[str, Any]] = []
    if cfg.use_parallel and cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(process_station, *job): job[0] for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                try:
                    results.append(future.result())
                except Exception as exc:
                    sid = core.normalize_station_id(Path(futures[future]).stem)
                    results.append({
                        "statistics": None, "parameters": [], "predictions": [], "fold_metrics": [],
                        "failures": [{"station_id": sid, "method": "process_pool", "validation_scheme": "all", "reason": str(exc)}],
                        "raw_audit": None, "overpass_audit": None, "multistart": [], "n_sensitivity": [],
                    })
                if completed % 20 == 0 or completed == len(futures):
                    print(f"Progress: {completed}/{len(futures)}", flush=True)
    else:
        for i, job in enumerate(jobs, start=1):
            results.append(process_station(*job))
            if i % 10 == 0 or i == len(jobs):
                print(f"Progress: {i}/{len(jobs)}", flush=True)

    statistics = pd.DataFrame([r["statistics"] for r in results if r.get("statistics") is not None])
    parameters = pd.DataFrame([x for r in results for x in r.get("parameters", [])])
    fold_metrics = pd.DataFrame([x for r in results for x in r.get("fold_metrics", [])])
    failures = pd.DataFrame([x for r in results for x in r.get("failures", [])])
    predictions = pd.concat(
        [x for r in results for x in r.get("predictions", []) if isinstance(x, pd.DataFrame) and not x.empty],
        ignore_index=True,
    ) if any(r.get("predictions") for r in results) else pd.DataFrame()
    raw_audit = pd.concat(
        [r["raw_audit"] for r in results if isinstance(r.get("raw_audit"), pd.DataFrame)],
        ignore_index=True,
    ) if any(isinstance(r.get("raw_audit"), pd.DataFrame) for r in results) else pd.DataFrame()
    overpass_audit = pd.concat(
        [r["overpass_audit"] for r in results if isinstance(r.get("overpass_audit"), pd.DataFrame)],
        ignore_index=True,
    ) if any(isinstance(r.get("overpass_audit"), pd.DataFrame) for r in results) else pd.DataFrame()
    multistart = pd.DataFrame([x for r in results for x in r.get("multistart", [])])
    n_sensitivity = pd.DataFrame([x for r in results for x in r.get("n_sensitivity", [])])

    basin_columns = [
        column for column in
        ("station_id", "basin_code", "basin_name", "basin_match_method")
        if column in statistics
    ]
    if not predictions.empty and len(basin_columns) > 1:
        predictions = predictions.merge(
            statistics[basin_columns].drop_duplicates("station_id"),
            on="station_id",
            how="left",
        )

    station_metrics = build_station_metrics(predictions)
    overall = build_overall_summary(predictions, station_metrics)
    repeat_uncertainty = build_repeat_uncertainty(station_metrics)
    flow_metrics = grouped_metrics(predictions, "flow_regime")
    wse_position_metrics = grouped_metrics(predictions, "wse_position_class")
    time_gap_metrics = grouped_metrics(predictions, "time_gap_class")
    basin_metrics = grouped_metrics(predictions, "basin_name")
    peak_metrics = build_peak_metrics(station_metrics)
    bootstrap = bootstrap_intervals(station_metrics, cfg)
    from common_support import paired_comparisons
    ablation_raw, ablation_summary, common_metrics, common_coverage = paired_comparisons(predictions,cfg,sys.modules[__name__])
    assignment_parts=[part for result in results for part in result.get('split_audit',[])]
    assignments=pd.concat(assignment_parts,ignore_index=True) if assignment_parts else pd.DataFrame()
    multistart_station = multistart_summary(multistart)
    n_sensitivity_summary = n_sensitivity_overall(n_sensitivity)
    retention_stage, retention_group = retention_tables(statistics, parameters)
    loss_reason_summary, qc_candidate_summary = qc_reason_tables(raw_audit, overpass_audit)
    drivers, driver_correlations = performance_driver_tables(
        station_metrics, statistics, parameters, predictions, cfg
    )

    table_map = {
        "split_assignments.csv": assignments,
        "ablation_common_support_metrics.csv": common_metrics,
        "ablation_common_support_coverage.csv": common_coverage,
        "station_data_statistics.csv": statistics,
        "parameter_estimates.csv": parameters,
        "validation_predictions.csv": predictions,
        "validation_metrics_by_fold.csv": fold_metrics,
        "validation_metrics_by_station.csv": station_metrics,
        "experiment_overall_summary.csv": overall,
        "wse_repeat_uncertainty_by_station.csv": repeat_uncertainty,
        "bootstrap_confidence_intervals.csv": bootstrap,
        "ablation_paired_station_differences.csv": ablation_raw,
        "ablation_paired_test_summary.csv": ablation_summary,
        "n_sensitivity_by_station.csv": n_sensitivity,
        "n_sensitivity_summary.csv": n_sensitivity_summary,
        "parameter_multistart_runs.csv": multistart,
        "parameter_multistart_summary.csv": multistart_station,
        "raw_record_qc_audit.csv": raw_audit,
        "overpass_qc_audit.csv": overpass_audit,
        "data_retention_stages.csv": retention_stage,
        "data_retention_by_group.csv": retention_group,
        "data_loss_reason_summary.csv": loss_reason_summary,
        "qc_candidate_summary.csv": qc_candidate_summary,
        "flow_regime_metrics.csv": flow_metrics,
        "wse_position_metrics.csv": wse_position_metrics,
        "time_gap_metrics.csv": time_gap_metrics,
        "basin_performance_metrics.csv": basin_metrics,
        "peak_flow_metrics_by_station.csv": peak_metrics,
        "performance_drivers_by_station.csv": drivers,
        "performance_driver_spearman.csv": driver_correlations,
        "failed_experiments.csv": failures,
    }
    manifest_rows = []
    for filename, frame in table_map.items():
        save_table(frame, tables_dir / filename)
        manifest_rows.append({
            "file": filename, "row_count": int(len(frame)), "column_count": int(len(frame.columns)),
        })
    save_table(pd.DataFrame(manifest_rows), tables_dir / "table_manifest.csv")

    print(f"Completed: {output_dir}", flush=True)
    print(f"Tables: {tables_dir}", flush=True)
    print("Next: run plot_review_revision_experiments.py --results-dir <output_dir>", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
