#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWOT 流量复现：n + WSE0 + 最低 WSE 有效宽度，非负约束逻辑。

核心水力关系
------------
1. 不使用 beta，不做 WSE 水力纠偏。
2. 每个站点先用该站全部 SWOT WSE-width（不使用 Q）建立一条全站统一的单调
   width 曲线；率定、复现和所有验证均固定使用这条曲线。
3. 对单调 WSE-width 折线做梯形积分；width 和断面积做非负保护。
4. 河床起点宽度固定为 0；WSE0 与训练期最低 WSE 处的有效宽度共同定义
   起算三角形：

       A0 = 0.5 * width_min_effective * (WSE_min - WSE0)

   最低 WSE 的观测宽度可能存在误差，因此 width_min_effective 在观测单调
   宽度的一定倍率范围内逐站搜索；默认范围为 0.5--1.5 倍。

5. Manning 宽河道近似：

       Q = (1/n) * A^(5/3) * width^(-2/3) * slope^(1/2)

6. 同时率定 n、WSE0 和 width_min_effective，使当前站点训练期流量综合误差
   最小。width0 始终固定为 0，不再参与率定。

验证设计
--------
- station_reconstruction：全样本率定后的站点内复现；
- wse_stratified：站点内 WSE 分层随机留出；
- temporal_forward：严格按时间顺序的扩展窗口交叉验证；
- spatial_block：空间块留站验证。目标站不使用 Q 率定，而是从训练站迁移
  n、相对起算深度和最低 WSE 有效宽度倍率；目标站自身的 WSE/width/slope 属于 SWOT
  当期可观测量，允许用于构造几何折线和计算 Q。

重要说明
--------
- 所有参数都逐站独立优化；不存在跨站点的全局 Q 目标函数。总体指标只在优化完成
  后汇总，不反向影响任一站点参数。
- 单调 width 是站点级全局预处理：同一站点所有交叉验证折使用完全相同的曲线。
  它可以使用完整卫星 WSE/width，但不使用任何 Q；这对应“已知卫星条件下复现”，
  不应被表述为严格禁止验证期协变量参与预处理的预测实验。
- Hampel 默认启用，但只用在训练数据或不含 Q 的卫星几何锚点上。验证记录不会
  因验证 Q 被删除，也不会用验证 Q 调参。
- 非正 slope 不再导致运行失败：用相应训练集的正 slope 中位数修复；若训练集中
  完全没有正值，使用全局安全值 1e-5。
- R2 为与旧代码一致的 Pearson 相关系数平方；NSE 单独输出。
- n、WSE0、最低 WSE 有效宽度仍可能存在等效性，参数本身不能直接解释为
  唯一真实河道断面。

直接运行
--------
    python swot_n_wse0_width0_nonnegative_cv.py

启用 50 次重复 WSE 分层随机分组
---------------------------------
    python swot_n_wse0_width0_nonnegative_cv.py --repeat-random-grouping \
        --repeat-count 50

先用 10 个站点快速检查
----------------------
    python swot_n_wse0_width0_nonnegative_cv.py --max-stations 10 --fast --serial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Windows 上外层已经按站点并行；限制数值库和 KMeans 的内部线程，避免嵌套并行、
# MKL 小样本内存警告以及不同机器上不一致的 CPU 探测。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

try:
    from sklearn.cluster import KMeans
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover - 仅在 sklearn 缺失时使用经度排序后备方案
    KMeans = None
    IsotonicRegression = None


# =============================================================================
# 用户最常修改的开关
# =============================================================================

DEFAULT_DATA_DIR = Path(r"D:\SWOT5.1 - 大改重投\103.水文站卫星数据融合0506_adjusted")
DEFAULT_METADATA_FILE = Path(r"D:\SWOT5.1 - 大改重投\4_Station_with_N\Find_N.csv")
DEFAULT_OUTPUT_DIR = Path(
    r"D:\SWOT5.1 - 大改重投\20260723_swot_n_wse0_minwidth_nonnegative_cv"
)

# 用户要求保留的重复随机分组开关。False 时仍运行 1 次基础 WSE 分层验证。
RUN_REPEATED_RANDOM_GROUPING = False
REPEATED_RANDOM_GROUP_COUNT = 50

ENABLE_HAMPEL = True
RUN_STATION_RECONSTRUCTION = True
RUN_WSE_STRATIFIED_CV = True
RUN_TEMPORAL_FORWARD_CV = True
RUN_SPATIAL_BLOCK_CV = True

# 用户指定：每个站点使用一条基于该站全部卫星观测的全局单调 width 曲线。
USE_GLOBAL_STATION_MONOTONIC_WIDTH = True

DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
RANDOM_SEED = 20260723


# =============================================================================
# 数值和实验设置
# =============================================================================

EPS = 1e-12
WIDTH_FLOOR = 1e-6
AREA_FLOOR = 1e-8
MIN_EFFECTIVE_DEPTH = 0.01
SLOPE_GLOBAL_FALLBACK = 1e-5

MIN_TRAIN_RECORDS = 5
MIN_METRIC_RECORDS = 5
MIN_TEMPORAL_TEST_RECORDS = 2

HAMPEL_WINDOW = 5
HAMPEL_N_SIGMAS = 3.0

WSE_STRATA_COUNT = 5
WSE_TRAIN_FRACTION = 0.70
TEMPORAL_FOLDS = 3
SPATIAL_FOLDS = 5

# n 的硬上限遵循本研究的物理判断；常见范围仅用于诊断，不作为硬约束或先验。
N_MIN, N_MAX = 0.010, 0.150
N_TYPICAL_MIN, N_TYPICAL_MAX = 0.025, 0.050
DEPTH_MIN, DEPTH_ABSOLUTE_MAX = 0.01, 5000.0
WIDTH0_FIXED = 0.0
MIN_WIDTH_FACTOR_LOW_DEFAULT = 0.50
MIN_WIDTH_FACTOR_HIGH_DEFAULT = 1.50

# 旧版“高 R2”综合目标的权重。没有 beta，也没有 n/depth/最低宽度先验惩罚。
GLOBAL_LOG_NSE_WEIGHT = 0.22
GLOBAL_LOG_R2_WEIGHT = 0.16
GLOBAL_KGE_WEIGHT = 0.12
LOW_LOG_NSE_WEIGHT = 0.20
LOW_LOG_R2_WEIGHT = 0.08
GLOBAL_PBIAS_PENALTY_WEIGHT = 0.18
LOW_PBIAS_PENALTY_WEIGHT = 0.22
LOW_LOG_BIAS_PENALTY_WEIGHT = 0.18
MEDIAN_RATIO_PENALTY_WEIGHT = 0.08
METRIC_CLIP_MIN_FOR_OBJECTIVE = -5.0


@dataclass
class Config:
    data_dir: str
    metadata_file: str
    output_dir: str
    repeat_random_grouping: bool = RUN_REPEATED_RANDOM_GROUPING
    repeat_count: int = REPEATED_RANDOM_GROUP_COUNT
    enable_hampel: bool = ENABLE_HAMPEL
    run_reconstruction: bool = RUN_STATION_RECONSTRUCTION
    run_wse_cv: bool = RUN_WSE_STRATIFIED_CV
    run_temporal_cv: bool = RUN_TEMPORAL_FORWARD_CV
    run_spatial_cv: bool = RUN_SPATIAL_BLOCK_CV
    wse_strata: int = WSE_STRATA_COUNT
    wse_train_fraction: float = WSE_TRAIN_FRACTION
    temporal_folds: int = TEMPORAL_FOLDS
    spatial_folds: int = SPATIAL_FOLDS
    workers: int = DEFAULT_WORKERS
    use_parallel: bool = True
    max_stations: Optional[int] = None
    fast: bool = False
    objective: str = "old_composite"
    seed: int = RANDOM_SEED
    aggregate_nodes_by_overpass: bool = True
    use_global_station_monotonic_width: bool = USE_GLOBAL_STATION_MONOTONIC_WIDTH
    min_width_factor_low: float = MIN_WIDTH_FACTOR_LOW_DEFAULT
    min_width_factor_high: float = MIN_WIDTH_FACTOR_HIGH_DEFAULT
    optimizer_maxiter: int = 60
    optimizer_popsize: int = 8
    optimizer_tol: float = 1e-3
    local_optimizer_maxiter: int = 400


# =============================================================================
# 基础工具
# =============================================================================

def stable_seed(*parts: Any, base: int = RANDOM_SEED) -> int:
    text = "|".join(str(x) for x in (base,) + parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def read_csv_safely(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path)


def normalize_station_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def finite_number(value: Any, default: float = np.nan) -> float:
    """Return a finite float or a stable missing-value fallback."""
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def finite_xy(obs: Sequence[float], sim: Sequence[float], positive: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(obs, dtype=float)
    y = np.asarray(sim, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if positive:
        mask &= (x > 0) & (y > 0)
    return x[mask], y[mask]


def hampel_mask(values: Sequence[float], window: int = HAMPEL_WINDOW,
                n_sigmas: float = HAMPEL_N_SIGMAS) -> np.ndarray:
    """与旧脚本一致的居中 Hampel：MAD=0 或序列过短时不删除。"""
    s = pd.Series(values, dtype=float).reset_index(drop=True)
    mask = np.zeros(len(s), dtype=bool)
    if len(s) < 2 * window + 1:
        return mask
    for i in range(window, len(s) - window):
        sample = s.iloc[i - window:i + window + 1].dropna()
        if len(sample) < 3 or not np.isfinite(s.iloc[i]):
            continue
        median = float(sample.median())
        mad_value = float(np.nanmedian(np.abs(sample.to_numpy(dtype=float) - median)))
        if not np.isfinite(median) or not np.isfinite(mad_value) or mad_value == 0:
            continue
        if abs(float(s.iloc[i]) - median) > n_sigmas * mad_value:
            mask[i] = True
    return mask


def repair_slopes(values: Sequence[float], fallback: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw = np.asarray(values, dtype=float)
    good = np.isfinite(raw) & (raw > 0)
    if fallback is None:
        fallback_value = float(np.nanmedian(raw[good])) if np.any(good) else SLOPE_GLOBAL_FALLBACK
        used_global = not np.any(good)
    else:
        fallback_value = float(fallback) if np.isfinite(fallback) and fallback > 0 else SLOPE_GLOBAL_FALLBACK
        used_global = not (np.isfinite(fallback) and fallback > 0)
    safe = np.where(good, raw, fallback_value)
    safe = np.maximum(safe, EPS)
    return safe, {
        "slope_positive_count": int(good.sum()),
        "slope_nonpositive_count": int((np.isfinite(raw) & (raw <= 0)).sum()),
        "slope_nonfinite_count": int((~np.isfinite(raw)).sum()),
        "slope_repaired_count": int((~good).sum()),
        "slope_fallback": float(fallback_value),
        "slope_used_global_fallback": bool(used_global),
    }


def collapse_wse_width(wse: Sequence[float], width: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"wse": wse, "width": width})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame["width"] = np.maximum(frame["width"].astype(float), WIDTH_FLOOR)
    frame = frame.groupby("wse", as_index=False)["width"].median().sort_values("wse")
    return frame["wse"].to_numpy(dtype=float), frame["width"].to_numpy(dtype=float)


def monotonic_width_curve(wse: Sequence[float], width: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """生成 WSE 单调不减的 width 曲线；不使用 Q。"""
    x, y = collapse_wse_width(wse, width)
    if len(x) < 2:
        return x, y
    if IsotonicRegression is not None:
        try:
            fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(x, y)
            return x, np.maximum(np.asarray(fitted, dtype=float), WIDTH_FLOOR)
        except Exception:
            pass
    return x, np.maximum.accumulate(np.maximum(y, WIDTH_FLOOR))


def width_from_monotonic_curve(curve_wse: Sequence[float], curve_width: Sequence[float],
                               test_wse: float, observed_width: float) -> Tuple[float, str]:
    x = np.asarray(curve_wse, dtype=float)
    y = np.asarray(curve_width, dtype=float)
    if len(x) == 0 or not np.isfinite(test_wse):
        return max(float(observed_width), WIDTH_FLOOR), "observed_nonnegative_fallback"
    value = float(np.interp(float(test_wse), x, y, left=y[0], right=y[-1]))
    return max(value, WIDTH_FLOOR), "global_station_monotonic_width_wse"


def fit_station_monotonic_width_curve(
    calibration_frame: pd.DataFrame,
    enable_hampel: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """仅从给定率定样本建立单调 WSE-width 曲线。"""
    work = calibration_frame.copy().sort_values("time_dt").reset_index(drop=True)
    if work.empty:
        raise ValueError("无法从空率定样本建立单调 WSE-width 曲线")

    wse_bad = (
        hampel_mask(work["wse_model"])
        if enable_hampel else np.zeros(len(work), dtype=bool)
    )
    width_bad = (
        hampel_mask(work["width"])
        if enable_hampel else np.zeros(len(work), dtype=bool)
    )
    anchors = work.loc[~(wse_bad | width_bad), ["wse_model", "width"]].copy()
    if len(anchors) < 2:
        anchors = work[["wse_model", "width"]].copy()
    curve_wse, curve_width = monotonic_width_curve(
        anchors["wse_model"], anchors["width"]
    )
    if len(curve_wse) == 0:
        raise ValueError("无法建立单调 WSE-width 曲线")

    model = {
        "curve_wse": np.asarray(curve_wse, dtype=float),
        "curve_width": np.asarray(curve_width, dtype=float),
        "calibration_count": int(len(work)),
        "anchor_count": int(len(curve_wse)),
        "enable_hampel": bool(enable_hampel),
    }
    diagnostics = {
        "global_monotonic_anchor_count": int(len(curve_wse)),
        "global_monotonic_hampel_wse_excluded_count": int(wse_bad.sum()),
        "global_monotonic_hampel_width_excluded_count": int(width_bad.sum()),
        "global_monotonic_wse_min": float(curve_wse[0]),
        "global_monotonic_wse_max": float(curve_wse[-1]),
        "width_curve_training_count": int(len(work)),
    }
    return model, diagnostics


def apply_station_monotonic_width_curve(
    frame: pd.DataFrame,
    curve_model: Mapping[str, Any],
    source: str,
    scope: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """把已拟合曲线应用到目标样本；目标 width 不参与曲线拟合。"""
    work = frame.copy().sort_values("time_dt").reset_index(drop=True)
    curve_wse = np.asarray(curve_model["curve_wse"], dtype=float)
    curve_width = np.asarray(curve_model["curve_width"], dtype=float)
    if len(curve_wse) == 0:
        raise ValueError("单调 WSE-width 曲线为空")
    if work.empty:
        work["width_global_monotonic"] = np.nan
        work["width_model"] = np.nan
        work["width_model_source"] = source
        work["width_curve_scope"] = scope
        work["width_curve_position"] = "empty"
        return work, {
            "target_count": 0,
            "target_below_curve_count": 0,
            "target_above_curve_count": 0,
        }

    target_wse = pd.to_numeric(work["wse_model"], errors="coerce").to_numpy(dtype=float)
    width_global = np.interp(
        target_wse,
        curve_wse,
        curve_width,
        left=curve_width[0],
        right=curve_width[-1],
    )
    width_global = np.maximum(width_global, WIDTH_FLOOR)
    work["width_global_monotonic"] = width_global
    work["width_model"] = width_global
    work["width_model_source"] = source
    work["width_curve_scope"] = scope
    work["width_curve_wse_min"] = float(curve_wse[0])
    work["width_curve_wse_max"] = float(curve_wse[-1])
    work["width_curve_training_count"] = int(curve_model["calibration_count"])
    work["width_curve_position"] = np.where(
        target_wse < curve_wse[0],
        "below_calibration_curve",
        np.where(
            target_wse > curve_wse[-1],
            "above_calibration_curve",
            "within_calibration_curve",
        ),
    )
    adjustment = width_global - pd.to_numeric(
        work["width"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_adjustment = adjustment[np.isfinite(adjustment)]
    raw_x, raw_y = collapse_wse_width(work["wse_model"], work["width"])
    model_x, model_y = collapse_wse_width(work["wse_model"], work["width_model"])
    return work, {
        "target_count": int(len(work)),
        "target_below_curve_count": int((target_wse < curve_wse[0]).sum()),
        "target_above_curve_count": int((target_wse > curve_wse[-1]).sum()),
        "raw_width_wse_decrease_count": (
            int(np.sum(np.diff(raw_y) < 0)) if len(raw_x) > 1 else 0
        ),
        "model_width_wse_decrease_count": (
            int(np.sum(np.diff(model_y) < -1e-10)) if len(model_x) > 1 else 0
        ),
        "global_width_adjustment_mean": (
            float(np.mean(finite_adjustment))
            if finite_adjustment.size else np.nan
        ),
        "global_width_adjustment_median_abs": (
            float(np.median(np.abs(finite_adjustment)))
            if finite_adjustment.size else np.nan
        ),
        "global_width_adjustment_max_abs": (
            float(np.max(np.abs(finite_adjustment)))
            if finite_adjustment.size else np.nan
        ),
    }


def apply_calibration_only_station_monotonic_width(
    calibration_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    enable_hampel: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """严格前向版本：曲线只使用率定折协变量，随后应用到目标折。"""
    model, fit_diag = fit_station_monotonic_width_curve(
        calibration_frame, enable_hampel=enable_hampel
    )
    transformed, apply_diag = apply_station_monotonic_width_curve(
        target_frame,
        model,
        source="calibration_only_monotonic_width_wse",
        scope="calibration_only",
    )
    return transformed, {**fit_diag, **apply_diag}


def apply_global_station_monotonic_width(frame: pd.DataFrame, enable_hampel: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    用当前站点全部卫星 WSE-width 建立唯一单调曲线，然后回填每条观测。

    这里只读取 WSE 和 width，不读取 Q。Hampel 仅决定哪些卫星点参与曲线拟合，
    不删除后续率定或验证记录；训练期是否排除异常记录仍由 prepare_training 决定。
    """
    if frame.empty:
        work = frame.copy()
        work["width_global_monotonic"] = np.nan
        work["width_model"] = np.nan
        work["width_model_source"] = "global_station_monotonic_width_wse"
        work["width_curve_scope"] = "all_station_covariates"
        return work, {"global_monotonic_anchor_count": 0}
    model, fit_diag = fit_station_monotonic_width_curve(
        frame, enable_hampel=enable_hampel
    )
    transformed, apply_diag = apply_station_monotonic_width_curve(
        frame,
        model,
        source="global_station_monotonic_width_wse",
        scope="all_station_covariates",
    )
    return transformed, {**fit_diag, **apply_diag}


def aggregate_overpasses(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in ("wse_model", "width", "slope", "Q") if c in frame.columns]
    other = [c for c in ("station_id", "source_file") if c in frame.columns]
    agg: Dict[str, str] = {c: "median" for c in numeric}
    agg.update({c: "first" for c in other})
    output = frame.groupby("time_dt", as_index=False, dropna=False).agg(agg)
    return output.sort_values("time_dt").reset_index(drop=True)


def load_station(path: Path, aggregate_nodes: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    station_id = normalize_station_id(path.stem)
    raw = read_csv_safely(path)
    raw_count = len(raw)
    wse_column = "wse_adj" if "wse_adj" in raw.columns else "wse"
    time_column = next((c for c in ("time_str", "time", "date") if c in raw.columns), None)
    required = {wse_column, "width", "slope", "Q"}
    missing = sorted(required - set(raw.columns))
    if missing or time_column is None:
        raise ValueError(f"缺少必要列: {missing}; time column={time_column}")

    frame = pd.DataFrame({
        "station_id": station_id,
        "source_file": str(path),
        "time_dt": pd.to_datetime(raw[time_column], errors="coerce", utc=True),
        "wse_model": pd.to_numeric(raw[wse_column], errors="coerce"),
        "width": pd.to_numeric(raw["width"], errors="coerce"),
        "slope": pd.to_numeric(raw["slope"], errors="coerce"),
        "Q": pd.to_numeric(raw["Q"], errors="coerce"),
    })
    parsed_time_count = int(frame["time_dt"].notna().sum())
    frame = frame.dropna(subset=["time_dt", "wse_model", "width", "Q"])
    frame = frame[(frame["width"] > 0) & (frame["Q"] > 0)]
    valid_before_aggregation = len(frame)
    if aggregate_nodes:
        frame = aggregate_overpasses(frame)
    frame = frame.sort_values("time_dt").reset_index(drop=True)
    frame["obs_id"] = [f"{station_id}|{t.isoformat()}|{i}" for i, t in enumerate(frame["time_dt"])]
    audit = {
        "station_id": station_id,
        "source_file": str(path),
        "raw_row_count": int(raw_count),
        "parsed_time_count": parsed_time_count,
        "valid_row_count_before_overpass_aggregation": int(valid_before_aggregation),
        "overpass_count": int(len(frame)),
        "wse_source_column": wse_column,
    }
    return frame, audit


def station_statistics(frame: pd.DataFrame, audit: Dict[str, Any], meta: Dict[str, Any],
                       enable_hampel: bool) -> Dict[str, Any]:
    out = dict(audit)
    out.update({
        "station_name": meta.get("Name", ""),
        "station_type": meta.get("type", ""),
        "longitude": meta.get("Lon_web", np.nan),
        "latitude": meta.get("Lat_web", np.nan),
        "n_reference": meta.get("N_River", np.nan),
        "start_time": frame["time_dt"].min() if len(frame) else pd.NaT,
        "end_time": frame["time_dt"].max() if len(frame) else pd.NaT,
    })
    for column in ("wse_model", "width", "slope", "Q"):
        x = pd.to_numeric(frame[column], errors="coerce")
        out[f"{column}_min"] = float(x.min()) if x.notna().any() else np.nan
        out[f"{column}_max"] = float(x.max()) if x.notna().any() else np.nan
        out[f"{column}_mean"] = float(x.mean()) if x.notna().any() else np.nan
        out[f"{column}_median"] = float(x.median()) if x.notna().any() else np.nan
        if enable_hampel and column in ("wse_model", "width", "slope"):
            out[f"hampel_{column}_candidate_count"] = int(hampel_mask(x).sum())
    out["wse_range"] = out.get("wse_model_max", np.nan) - out.get("wse_model_min", np.nan)
    out["q_range"] = out.get("Q_max", np.nan) - out.get("Q_min", np.nan)
    out["nonpositive_slope_count"] = int((pd.to_numeric(frame["slope"], errors="coerce") <= 0).sum())
    return out


# =============================================================================
# 评价指标与旧版综合目标
# =============================================================================

def calculate_metrics(obs: Sequence[float], sim: Sequence[float]) -> Dict[str, Any]:
    x, y = finite_xy(obs, sim)
    result: Dict[str, Any] = {
        "valid_count": int(len(x)), "R2": np.nan, "NSE": np.nan, "KGE": np.nan,
        "logR2": np.nan, "logNSE": np.nan, "RMSE": np.nan, "MAE": np.nan,
        "Bias": np.nan, "PBIAS": np.nan, "median_signed_log_ratio": np.nan,
        "median_abs_log_ratio": np.nan, "factor_error_median": np.nan,
    }
    if len(x) == 0:
        return result
    error = y - x
    result.update({
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "PBIAS": float(100.0 * np.sum(error) / np.sum(x)) if abs(np.sum(x)) > EPS else np.nan,
    })
    xp, yp = finite_xy(x, y, positive=True)
    if len(xp):
        lr = np.log10(yp / xp)
        result["median_signed_log_ratio"] = float(np.median(lr))
        result["median_abs_log_ratio"] = float(np.median(np.abs(lr)))
        result["factor_error_median"] = float(10.0 ** result["median_abs_log_ratio"])
    if len(x) >= MIN_METRIC_RECORDS and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        result["R2"] = r * r
        den = float(np.sum((x - np.mean(x)) ** 2))
        result["NSE"] = float(1.0 - np.sum(error ** 2) / den) if den > 0 else np.nan
        alpha = float(np.std(y) / np.std(x))
        mean_ratio = float(np.mean(y) / np.mean(x)) if abs(np.mean(x)) > EPS else np.nan
        result["KGE"] = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (mean_ratio - 1) ** 2))
    if len(xp) >= MIN_METRIC_RECORDS and np.std(np.log10(xp)) > 0 and np.std(np.log10(yp)) > 0:
        lx, ly = np.log10(xp), np.log10(yp)
        lr_corr = float(np.corrcoef(lx, ly)[0, 1])
        result["logR2"] = lr_corr * lr_corr
        den = float(np.sum((lx - np.mean(lx)) ** 2))
        result["logNSE"] = float(1.0 - np.sum((ly - lx) ** 2) / den) if den > 0 else np.nan
    return result


def low_flow_mask(q: Sequence[float]) -> np.ndarray:
    values = np.asarray(q, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    fixed = valid & (values <= 100.0)
    if fixed.sum() >= MIN_METRIC_RECORDS:
        return fixed
    if valid.sum() <= MIN_METRIC_RECORDS:
        return valid
    limit = float(np.nanquantile(values[valid], 0.40))
    fallback = valid & (values <= limit)
    if fallback.sum() < MIN_METRIC_RECORDS:
        order = np.where(valid)[0][np.argsort(values[valid])]
        fallback = np.zeros(len(values), dtype=bool)
        fallback[order[:MIN_METRIC_RECORDS]] = True
    return fallback


def clipped_metric(value: Any) -> float:
    return max(float(value), METRIC_CLIP_MIN_FOR_OBJECTIVE) if np.isfinite(value) else METRIC_CLIP_MIN_FOR_OBJECTIVE


def old_composite_score(obs: Sequence[float], sim: Sequence[float]) -> float:
    global_m = calculate_metrics(obs, sim)
    low = low_flow_mask(obs)
    low_m = calculate_metrics(np.asarray(obs)[low], np.asarray(sim)[low])
    score = (
        GLOBAL_LOG_NSE_WEIGHT * clipped_metric(global_m["logNSE"])
        + GLOBAL_LOG_R2_WEIGHT * (float(np.clip(global_m["logR2"], 0, 1)) if np.isfinite(global_m["logR2"]) else 0.0)
        + GLOBAL_KGE_WEIGHT * clipped_metric(global_m["KGE"])
        + LOW_LOG_NSE_WEIGHT * clipped_metric(low_m["logNSE"])
        + LOW_LOG_R2_WEIGHT * (float(np.clip(low_m["logR2"], 0, 1)) if np.isfinite(low_m["logR2"]) else 0.0)
        - GLOBAL_PBIAS_PENALTY_WEIGHT * (abs(global_m["PBIAS"]) / 100.0 if np.isfinite(global_m["PBIAS"]) else 10.0)
        - LOW_PBIAS_PENALTY_WEIGHT * (abs(low_m["PBIAS"]) / 100.0 if np.isfinite(low_m["PBIAS"]) else 10.0)
        - LOW_LOG_BIAS_PENALTY_WEIGHT * (abs(low_m["median_signed_log_ratio"]) if np.isfinite(low_m["median_signed_log_ratio"]) else 5.0)
    )
    ratio = np.asarray(sim, dtype=float) / np.asarray(obs, dtype=float)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if len(ratio):
        score -= MEDIAN_RATIO_PENALTY_WEIGHT * abs(float(np.log10(np.median(ratio))))
    return float(score)


# =============================================================================
# 训练期 QC、几何积分和参数率定
# =============================================================================

def prepare_training(frame: pd.DataFrame, enable_hampel: bool) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    work = frame.copy().sort_values("time_dt").reset_index(drop=True)
    diag: Dict[str, Any] = {"train_input_count": int(len(work))}
    if len(work) < MIN_TRAIN_RECORDS:
        return None, diag

    wse_bad = hampel_mask(work["wse_model"]) if enable_hampel else np.zeros(len(work), dtype=bool)
    width_bad = hampel_mask(work["width"]) if enable_hampel else np.zeros(len(work), dtype=bool)

    # slope 的 Hampel 异常不删除整条流量记录，而是与非正 slope 一起用训练期中位数修复。
    positive_slope = pd.to_numeric(work["slope"], errors="coerce").where(lambda x: x > 0)
    log_slope = np.log10(positive_slope)
    slope_bad = hampel_mask(log_slope) if enable_hampel else np.zeros(len(work), dtype=bool)
    slope_for_repair = pd.to_numeric(work["slope"], errors="coerce").to_numpy(dtype=float)
    slope_for_repair[slope_bad] = np.nan
    safe_slope, slope_diag = repair_slopes(slope_for_repair)
    work["slope_safe"] = safe_slope

    keep = ~(wse_bad | width_bad)
    work = work.loc[keep].copy().reset_index(drop=True)
    if "width_model" not in work.columns:
        work["width_model"] = np.maximum(pd.to_numeric(work["width"], errors="coerce"), WIDTH_FLOOR)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["wse_model", "width", "width_model", "Q", "slope_safe"]
    )
    work = work[(work["width"] > 0) & (work["width_model"] > 0) & (work["Q"] > 0)]
    diag.update({
        "hampel_wse_removed_count": int(wse_bad.sum()),
        "hampel_width_removed_count": int(width_bad.sum()),
        "hampel_slope_repaired_count": int(slope_bad.sum()),
        "train_after_hampel_count": int(len(work)),
        **slope_diag,
    })
    if len(work) < MIN_TRAIN_RECORDS:
        return None, diag
    return work, diag


def effective_min_width_bounds(
    anchor_width: Sequence[float],
    factor_low: float,
    factor_high: float,
    enforce_monotonic: bool = True,
) -> Tuple[float, float, float]:
    """
    返回最低 WSE 有效宽度的观测参考值和搜索边界。

    上界不超过第二个单调宽度锚点，从而改变最低锚点后仍保持整条
    width--WSE 曲线单调不减。只有一个锚点时仅使用倍率边界。
    """
    widths = np.maximum(np.asarray(anchor_width, dtype=float), WIDTH_FLOOR)
    if len(widths) == 0:
        raise ValueError("无法确定最低 WSE 宽度搜索范围")
    low_factor = max(float(factor_low), EPS)
    high_factor = max(float(factor_high), low_factor)
    observed = float(widths[0])
    lower = max(WIDTH_FLOOR, observed * low_factor)
    upper = max(lower, observed * high_factor)
    if enforce_monotonic and len(widths) > 1:
        upper = max(lower, min(upper, float(widths[1])))
    if upper <= lower:
        upper = lower * (1.0 + 1e-6)
    return observed, float(lower), float(upper)


def apply_effective_min_width(
    anchor_width: Sequence[float],
    effective_width: float,
    enforce_monotonic: bool = True,
) -> np.ndarray:
    """替换最低 WSE 宽度并再次执行单调保护。"""
    widths = np.maximum(np.asarray(anchor_width, dtype=float).copy(), WIDTH_FLOOR)
    if len(widths) == 0:
        return widths
    widths[0] = max(float(effective_width), WIDTH_FLOOR)
    return np.maximum.accumulate(widths) if enforce_monotonic else widths


def geometry_arrays(
    train: pd.DataFrame,
    depth0: float,
    min_width_effective: float,
    enforce_monotonic: bool = True,
) -> Dict[str, np.ndarray]:
    anchor_wse, anchor_width_observed = collapse_wse_width(
        train["wse_model"], train["width_model"]
    )
    if len(anchor_wse) == 0:
        raise ValueError("无有效 WSE-width 几何锚点")
    anchor_width = apply_effective_min_width(
        anchor_width_observed, min_width_effective, enforce_monotonic
    )
    wse0 = float(anchor_wse[0] - depth0)
    x = np.concatenate([[wse0], anchor_wse])
    # 河床起点宽度固定为 0；积分时保留精确的三角形几何。
    y = np.concatenate([[WIDTH0_FIXED], np.maximum(anchor_width, WIDTH_FLOOR)])
    area = np.zeros(len(x), dtype=float)
    if len(x) > 1:
        area[1:] = np.cumsum(np.maximum(np.diff(x), 0.0) * (y[:-1] + y[1:]) / 2.0)
    area = np.maximum(area, 0.0)
    return {
        "wse": x,
        "width": y,
        "area": area,
        "wse0": np.array([wse0]),
        "anchor_wse": anchor_wse,
        "anchor_width": anchor_width,
        "anchor_width_observed": anchor_width_observed,
    }


def area_at_observation(anchor_wse: np.ndarray, anchor_width: np.ndarray,
                        wse0: float, test_wse: float,
                        test_width: float) -> Tuple[float, str]:
    """插入当前 SWOT width 后，从 WSE0 到当前 WSE 做原始折线梯形积分。"""
    xt = float(test_wse)
    yt = max(float(test_width), WIDTH_FLOOR)
    if not np.isfinite(xt) or not np.isfinite(yt):
        return np.nan, "invalid"
    if xt <= wse0:
        return AREA_FLOOR, "below_wse0"
    if len(anchor_wse) and np.isclose(
        xt, float(anchor_wse[0]), rtol=1e-9, atol=1e-8
    ):
        # 最低 WSE 的宽度已经作为率定参数修正；在该锚点处保持面积计算一致。
        yt = max(float(anchor_width[0]), WIDTH_FLOOR)

    inside = (anchor_wse > wse0) & (anchor_wse < xt)
    x = np.concatenate([[wse0], anchor_wse[inside], [xt]])
    y = np.concatenate([[WIDTH0_FIXED], anchor_width[inside], [yt]])
    order = np.argsort(x)
    x, y = x[order], y[order]
    x, y = collapse_wse_width(x, y)
    area = float(np.sum(np.maximum(np.diff(x), 0.0) * (y[:-1] + y[1:]) / 2.0)) if len(x) > 1 else AREA_FLOOR
    if xt < float(np.min(anchor_wse)):
        position = "below_train_min_wse"
    elif xt > float(np.max(anchor_wse)):
        position = "above_train_max_wse"
    else:
        position = "within_train_wse_range"
    return max(area, AREA_FLOOR), position


def manning_q(area: np.ndarray, width: np.ndarray, slope: np.ndarray, n_value: float) -> np.ndarray:
    a = np.maximum(np.asarray(area, dtype=float), AREA_FLOOR)
    w = np.maximum(np.asarray(width, dtype=float), WIDTH_FLOOR)
    s = np.maximum(np.asarray(slope, dtype=float), EPS)
    n = max(float(n_value), EPS)
    return (1.0 / n) * a ** (5.0 / 3.0) * w ** (-2.0 / 3.0) * np.sqrt(s)


def training_predictions(train: pd.DataFrame, n_value: float, depth0: float,
                         min_width_effective: float,
                         enforce_monotonic: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    geom = geometry_arrays(
        train, depth0, min_width_effective, enforce_monotonic=enforce_monotonic
    )
    # 训练锚点的累计面积；Manning 通常使用各观测 width，但最低 WSE 锚点使用
    # 同一个率定后的有效宽度，避免面积与水面宽度采用互相矛盾的值。
    areas = np.interp(train["wse_model"].to_numpy(dtype=float), geom["wse"], geom["area"])
    width_for_q = train["width_model"].to_numpy(dtype=float).copy()
    minimum_mask = np.isclose(
        train["wse_model"].to_numpy(dtype=float),
        float(geom["anchor_wse"][0]),
        rtol=1e-9,
        atol=1e-8,
    )
    width_for_q[minimum_mask] = float(geom["anchor_width"][0])
    q = manning_q(areas, width_for_q,
                  train["slope_safe"].to_numpy(dtype=float), n_value)
    return q, geom


def objective_loss(
    params_log: np.ndarray,
    train: pd.DataFrame,
    objective: str,
    enforce_monotonic: bool = True,
) -> float:
    n_value, depth0, min_width_effective = np.exp(np.asarray(params_log, dtype=float))
    try:
        qcalc, _ = training_predictions(
            train,
            n_value,
            depth0,
            min_width_effective,
            enforce_monotonic=enforce_monotonic,
        )
        qobs = train["Q"].to_numpy(dtype=float)
        if not np.all(np.isfinite(qcalc)):
            return 1e12
        if objective == "old_composite":
            return -old_composite_score(qobs, qcalc)
        if objective == "normalized_mse":
            scale = max(float(np.std(qobs)), 0.1 * float(np.mean(qobs)), EPS)
            return float(np.mean(((qcalc - qobs) / scale) ** 2))
        if objective == "log_mse":
            return float(np.mean((np.log10(np.maximum(qcalc, EPS)) - np.log10(qobs)) ** 2))
        raise ValueError(f"未知目标函数: {objective}")
    except Exception:
        return 1e12


def hybrid_global_local_optimize(
    func: Any,
    bounds: Sequence[Tuple[float, float]],
    args: Tuple[Any, ...],
    seed: int,
    fast: bool = False,
    maxiter: int = 60,
    popsize: int = 8,
    tol: float = 1e-3,
    local_maxiter: int = 400,
) -> Dict[str, Any]:
    """
    先做差分进化全局搜索，再用有界 Powell 局部收敛。

    原复合目标含截断指标，未必满足光滑梯度假设；Powell 比默认的
    L-BFGS-B polish 更稳健。返回值同时保留两阶段状态，避免把
    “DE 达到最大迭代数”误当作整个率定失败。
    """
    de_maxiter = 10 if fast else max(1, int(maxiter))
    de_popsize = 5 if fast else max(5, int(popsize))
    de_result = differential_evolution(
        func,
        bounds=bounds,
        args=args,
        seed=int(seed),
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=float(tol),
        atol=1e-8,
        polish=False,
        updating="immediate",
        workers=1,
    )
    local_result = minimize(
        func,
        x0=np.asarray(de_result.x, dtype=float),
        args=args,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 80 if fast else max(50, int(local_maxiter)),
            "xtol": 1e-5,
            "ftol": 1e-7,
        },
    )
    candidates = [
        ("global", de_result),
        ("local", local_result),
    ]
    finite = [(name, result) for name, result in candidates if np.isfinite(result.fun)]
    if not finite:
        return {
            "x": np.asarray(de_result.x, dtype=float),
            "fun": np.inf,
            "success": False,
            "fit_accepted": False,
            "message": "No finite optimizer solution",
            "nfev": int(getattr(de_result, "nfev", 0)) + int(getattr(local_result, "nfev", 0)),
            "nit": int(getattr(de_result, "nit", 0)) + int(getattr(local_result, "nit", 0)),
            "global_success": bool(de_result.success),
            "local_success": bool(local_result.success),
        }
    selected_name, selected = min(finite, key=lambda item: float(item[1].fun))
    accepted = bool(de_result.success or local_result.success)
    return {
        "x": np.asarray(selected.x, dtype=float),
        "fun": float(selected.fun),
        "success": bool(de_result.success or local_result.success),
        "fit_accepted": accepted,
        "selected_stage": selected_name,
        "message": (
            f"global: {de_result.message}; local: {local_result.message}; "
            f"selected={selected_name}"
        ),
        "nfev": int(getattr(de_result, "nfev", 0)) + int(getattr(local_result, "nfev", 0)),
        "nit": int(getattr(de_result, "nit", 0)) + int(getattr(local_result, "nit", 0)),
        "global_success": bool(de_result.success),
        "local_success": bool(local_result.success),
        "global_fun": float(de_result.fun) if np.isfinite(de_result.fun) else np.nan,
        "local_fun": float(local_result.fun) if np.isfinite(local_result.fun) else np.nan,
    }


def fit_parameters(train_input: pd.DataFrame, cfg: Config, fit_seed: int) -> Optional[Dict[str, Any]]:
    train, qc = prepare_training(train_input, cfg.enable_hampel)
    if train is None:
        return None

    anchor_wse, anchor_width = collapse_wse_width(train["wse_model"], train["width_model"])
    wse_span = max(float(np.ptp(anchor_wse)), 0.1)
    depth_upper = min(DEPTH_ABSOLUTE_MAX, max(50.0, 20.0 * wse_span))
    min_width_observed, min_width_lower, min_width_upper = effective_min_width_bounds(
        anchor_width,
        cfg.min_width_factor_low,
        cfg.min_width_factor_high,
        enforce_monotonic=cfg.use_global_station_monotonic_width,
    )
    bounds = [
        (math.log(N_MIN), math.log(N_MAX)),
        (math.log(DEPTH_MIN), math.log(depth_upper)),
        (math.log(min_width_lower), math.log(min_width_upper)),
    ]
    result = hybrid_global_local_optimize(
        objective_loss,
        bounds=bounds,
        args=(train, cfg.objective, cfg.use_global_station_monotonic_width),
        seed=int(fit_seed),
        fast=cfg.fast,
        maxiter=cfg.optimizer_maxiter,
        popsize=cfg.optimizer_popsize,
        tol=cfg.optimizer_tol,
        local_maxiter=cfg.local_optimizer_maxiter,
    )
    if not np.isfinite(result["fun"]):
        return None
    n_value, depth0, min_width_effective = np.exp(result["x"])
    qtrain, geom = training_predictions(
        train,
        n_value,
        depth0,
        min_width_effective,
        enforce_monotonic=cfg.use_global_station_monotonic_width,
    )
    train_metrics = calculate_metrics(train["Q"], qtrain)
    wse0 = float(anchor_wse[0] - depth0)
    adjusted_anchor_width = np.asarray(geom["anchor_width"], dtype=float)
    min_width_effective = float(adjusted_anchor_width[0])
    a0 = float(0.5 * min_width_effective * depth0)
    tol = 0.01
    log_bounds = np.asarray(bounds, dtype=float)
    span = log_bounds[:, 1] - log_bounds[:, 0]
    hit_low = (result["x"] - log_bounds[:, 0]) <= tol * span
    hit_high = (log_bounds[:, 1] - result["x"]) <= tol * span
    return {
        "n": float(n_value), "WSE0": wse0, "width0": WIDTH0_FIXED,
        "depth0": float(depth0), "A0": a0,
        "train_wse_min": float(anchor_wse[0]), "train_wse_max": float(anchor_wse[-1]),
        "train_wse_range": float(np.ptp(anchor_wse)),
        "train_width_at_min_wse": min_width_effective,
        "min_width_observed": min_width_observed,
        "min_width_effective": min_width_effective,
        "min_width_factor": float(min_width_effective / max(min_width_observed, WIDTH_FLOOR)),
        "min_width_search_lower": min_width_lower,
        "min_width_search_upper": min_width_upper,
        "depth_to_wse_range_ratio": float(depth0 / wse_span),
        "width0_to_min_width_ratio": 0.0,
        "objective": cfg.objective, "objective_loss": float(result["fun"]),
        "optimizer_success": bool(result["success"]),
        "fit_accepted": bool(result["fit_accepted"]),
        "global_optimizer_success": bool(result["global_success"]),
        "local_optimizer_success": bool(result["local_success"]),
        "optimizer_selected_stage": result.get("selected_stage", ""),
        "optimizer_message": str(result["message"]),
        "optimizer_nfev": int(result["nfev"]), "optimizer_nit": int(result["nit"]),
        "n_in_typical_range": bool(N_TYPICAL_MIN <= n_value <= N_TYPICAL_MAX),
        "n_above_typical_range": bool(n_value > N_TYPICAL_MAX),
        "n_at_lower_bound": bool(hit_low[0]), "n_at_upper_bound": bool(hit_high[0]),
        "depth_at_lower_bound": bool(hit_low[1]), "depth_at_upper_bound": bool(hit_high[1]),
        "width0_fixed_zero": True,
        "width0_at_lower_bound": True, "width0_at_upper_bound": False,
        "min_width_at_lower_bound": bool(hit_low[2]),
        "min_width_at_upper_bound": bool(hit_high[2]),
        "anchor_wse": anchor_wse, "anchor_width": adjusted_anchor_width,
        "train_slope_fallback": float(qc["slope_fallback"]),
        "train_metrics": train_metrics, "qc": qc,
    }


def predict_frame(fit: Dict[str, Any], frame: pd.DataFrame, scheme: str,
                  fold_id: int, repeat_id: int, sample_role: str,
                  use_global_station_monotonic_width: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    anchor_wse = np.asarray(fit["anchor_wse"], dtype=float)
    anchor_width = np.asarray(fit["anchor_width"], dtype=float)
    for _, row in frame.iterrows():
        slope_raw = float(row["slope"]) if np.isfinite(row["slope"]) else np.nan
        slope_safe, slope_diag = repair_slopes([slope_raw], fallback=fit["train_slope_fallback"])
        width_raw = float(row["width"])
        if use_global_station_monotonic_width and "width_global_monotonic" in row.index:
            width_used = max(float(row["width_global_monotonic"]), WIDTH_FLOOR)
            width_source = str(row.get(
                "width_model_source", "global_station_monotonic_width_wse"
            ))
        else:
            width_used, width_source = max(width_raw, WIDTH_FLOOR), "observed_nonnegative_width"
        if len(anchor_wse) and np.isclose(
            float(row["wse_model"]), float(anchor_wse[0]), rtol=1e-9, atol=1e-8
        ):
            width_used = max(float(anchor_width[0]), WIDTH_FLOOR)
            width_source = f"{width_source}+optimized_minimum_wse_anchor"
        area, position = area_at_observation(
            anchor_wse, anchor_width, fit["WSE0"], float(row["wse_model"]), width_used,
        )
        qcalc = float(manning_q(np.array([area]), np.array([width_used]), slope_safe, fit["n"])[0])
        rows.append({
            "station_id": row["station_id"], "obs_id": row["obs_id"],
            "time": row["time_dt"], "scheme": scheme, "fold_id": fold_id,
            "repeat_id": repeat_id, "sample_role": sample_role,
            "WSE": float(row["wse_model"]), "width_raw": width_raw,
            "width_used": width_used, "width_source": width_source,
            "width_adjustment": float(width_used - width_raw),
            "width_adjustment_percent": float(100.0 * (width_used - width_raw) / width_raw) if width_raw > 0 else np.nan,
            "width_curve_scope": row.get("width_curve_scope", "not_recorded"),
            "width_curve_position": row.get("width_curve_position", "not_recorded"),
            "width_curve_wse_min": finite_number(row.get("width_curve_wse_min")),
            "width_curve_wse_max": finite_number(row.get("width_curve_wse_max")),
            "width_curve_training_count": finite_number(row.get("width_curve_training_count")),
            "slope_raw": slope_raw, "slope_used": float(slope_safe[0]),
            "slope_was_repaired": bool(slope_diag["slope_repaired_count"] > 0),
            "area": float(area), "area_position": position,
            "Q_observed": float(row["Q"]), "Q_calculated": qcalc,
            "n": fit["n"], "WSE0": fit["WSE0"], "width0": fit["width0"],
            "depth0": fit["depth0"], "A0": fit["A0"],
            "min_width_observed": fit.get("min_width_observed", np.nan),
            "min_width_effective": fit.get("min_width_effective", np.nan),
            "min_width_factor": fit.get("min_width_factor", np.nan),
        })
    return pd.DataFrame(rows)


def public_fit_row(fit: Dict[str, Any], station_id: str, scheme: str,
                   fold_id: int, repeat_id: int, train_count: int,
                   validation_count: int) -> Dict[str, Any]:
    excluded = {"anchor_wse", "anchor_width", "train_metrics", "qc"}
    row = {k: v for k, v in fit.items() if k not in excluded}
    row.update({
        "station_id": station_id, "scheme": scheme, "fold_id": fold_id,
        "repeat_id": repeat_id, "train_count_before_qc": int(train_count),
        "validation_count": int(validation_count), **fit["qc"],
        "optimization_unit": "station",
        "width_rule_all_stages": "global_station_isotonic" if USE_GLOBAL_STATION_MONOTONIC_WIDTH else "observed_nonnegative",
    })
    row.update({f"train_{k}": v for k, v in fit["train_metrics"].items()})
    return row


def metric_row(pred: pd.DataFrame, station_id: str, scheme: str,
               fold_id: int, repeat_id: int, train_count: int) -> Dict[str, Any]:
    row = {
        "station_id": station_id, "scheme": scheme, "fold_id": fold_id,
        "repeat_id": repeat_id, "train_count": int(train_count),
        "validation_count": int(len(pred)),
    }
    row.update(calculate_metrics(pred["Q_observed"], pred["Q_calculated"]))
    if len(pred):
        row["above_train_wse_count"] = int((pred["area_position"] == "above_train_max_wse").sum())
        row["below_train_wse_count"] = int((pred["area_position"] == "below_train_min_wse").sum())
        row["below_wse0_count"] = int((pred["area_position"] == "below_wse0").sum())
        row["repaired_validation_slope_count"] = int(pred["slope_was_repaired"].sum())
    return row


# =============================================================================
# 分组与站点内实验
# =============================================================================

def wse_stratified_split(frame: pd.DataFrame, train_fraction: float, strata_count: int,
                         seed: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    if len(frame) < MIN_TRAIN_RECORDS + 1:
        return None
    work = frame.copy().reset_index(drop=True)
    q = min(max(2, strata_count), len(work))
    try:
        strata = pd.qcut(work["wse_model"].rank(method="first"), q=q, labels=False, duplicates="drop")
    except Exception:
        strata = pd.cut(work["wse_model"].rank(method="first"), bins=q, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    validation_indices: List[int] = []
    for _, indices in work.groupby(strata, dropna=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        rng.shuffle(idx)
        if len(idx) == 1:
            train_indices.extend(idx.tolist())
            continue
        n_train = max(1, min(len(idx) - 1, int(round(train_fraction * len(idx)))))
        train_indices.extend(idx[:n_train].tolist())
        validation_indices.extend(idx[n_train:].tolist())
    if len(train_indices) < MIN_TRAIN_RECORDS or len(validation_indices) == 0:
        all_idx = np.arange(len(work))
        rng.shuffle(all_idx)
        n_train = max(MIN_TRAIN_RECORDS, min(len(work) - 1, int(round(train_fraction * len(work)))))
        train_indices = all_idx[:n_train].tolist()
        validation_indices = all_idx[n_train:].tolist()
    return work.iloc[sorted(train_indices)].copy(), work.iloc[sorted(validation_indices)].copy()


def temporal_forward_splits(frame: pd.DataFrame, requested_folds: int, initial_fraction: float = 0.70) -> List[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    work = frame.sort_values("time_dt").reset_index(drop=True)
    n = len(work)
    if n < MIN_TRAIN_RECORDS + MIN_TEMPORAL_TEST_RECORDS:
        return []
    initial = max(MIN_TRAIN_RECORDS, int(math.floor(initial_fraction * n)))
    initial = min(initial, n - 1)
    remaining = n - initial
    actual = min(requested_folds, max(1, remaining // MIN_TEMPORAL_TEST_RECORDS))
    boundaries = np.linspace(initial, n, actual + 1, dtype=int)
    output = []
    for fold in range(actual):
        train_end = int(boundaries[fold])
        test_end = int(boundaries[fold + 1])
        train = work.iloc[:train_end].copy()
        test = work.iloc[train_end:test_end].copy()
        if len(train) >= MIN_TRAIN_RECORDS and len(test) >= 1:
            output.append((fold, train, test))
    return output


def process_station(path_text: str, meta: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    path = Path(path_text)
    station_id = normalize_station_id(path.stem)
    result: Dict[str, Any] = {
        "statistics": None, "parameters": [], "predictions": [],
        "fold_metrics": [], "failures": [],
    }
    try:
        frame, audit = load_station(path, cfg.aggregate_nodes_by_overpass)
        if cfg.use_global_station_monotonic_width:
            frame, monotonic_diag = apply_global_station_monotonic_width(frame, cfg.enable_hampel)
            audit.update(monotonic_diag)
        else:
            frame["width_model"] = np.maximum(frame["width"].to_numpy(dtype=float), WIDTH_FLOOR)
        result["statistics"] = station_statistics(frame, audit, meta, cfg.enable_hampel)
        if len(frame) < MIN_TRAIN_RECORDS:
            result["failures"].append({"station_id": station_id, "scheme": "all", "reason": "insufficient_records"})
            return result

        if cfg.run_reconstruction or cfg.run_spatial_cv:
            fit = fit_parameters(frame, cfg, stable_seed(station_id, "full", base=cfg.seed))
            if fit is None:
                result["failures"].append({"station_id": station_id, "scheme": "station_reconstruction", "reason": "fit_failed"})
            else:
                result["parameters"].append(public_fit_row(fit, station_id, "station_reconstruction", 0, 0, len(frame), len(frame)))
                if cfg.run_reconstruction:
                    pred = predict_frame(
                        fit, frame, "station_reconstruction", 0, 0, "reconstruction",
                        cfg.use_global_station_monotonic_width,
                    )
                    result["predictions"].append(pred)
                    result["fold_metrics"].append(metric_row(pred, station_id, "station_reconstruction", 0, 0, len(frame)))

        if cfg.run_wse_cv:
            repeats = cfg.repeat_count if cfg.repeat_random_grouping else 1
            for repeat_id in range(repeats):
                split = wse_stratified_split(
                    frame, cfg.wse_train_fraction, cfg.wse_strata,
                    stable_seed(station_id, "wse", repeat_id, base=cfg.seed),
                )
                if split is None:
                    result["failures"].append({"station_id": station_id, "scheme": "wse_stratified", "repeat_id": repeat_id, "reason": "split_failed"})
                    continue
                train, test = split
                fit = fit_parameters(train, cfg, stable_seed(station_id, "wse_fit", repeat_id, base=cfg.seed))
                if fit is None:
                    result["failures"].append({"station_id": station_id, "scheme": "wse_stratified", "repeat_id": repeat_id, "reason": "fit_failed"})
                    continue
                pred = predict_frame(
                    fit, test, "wse_stratified", 0, repeat_id, "validation",
                    cfg.use_global_station_monotonic_width,
                )
                result["parameters"].append(public_fit_row(fit, station_id, "wse_stratified", 0, repeat_id, len(train), len(test)))
                result["predictions"].append(pred)
                result["fold_metrics"].append(metric_row(pred, station_id, "wse_stratified", 0, repeat_id, len(train)))

        if cfg.run_temporal_cv:
            splits = temporal_forward_splits(frame, cfg.temporal_folds)
            if not splits:
                result["failures"].append({"station_id": station_id, "scheme": "temporal_forward", "reason": "split_failed"})
            for fold_id, train, test in splits:
                fit = fit_parameters(train, cfg, stable_seed(station_id, "time_fit", fold_id, base=cfg.seed))
                if fit is None:
                    result["failures"].append({"station_id": station_id, "scheme": "temporal_forward", "fold_id": fold_id, "reason": "fit_failed"})
                    continue
                pred = predict_frame(
                    fit, test, "temporal_forward", fold_id, 0, "validation",
                    cfg.use_global_station_monotonic_width,
                )
                result["parameters"].append(public_fit_row(fit, station_id, "temporal_forward", fold_id, 0, len(train), len(test)))
                result["predictions"].append(pred)
                result["fold_metrics"].append(metric_row(pred, station_id, "temporal_forward", fold_id, 0, len(train)))
    except Exception as exc:
        result["failures"].append({
            "station_id": station_id, "scheme": "station_worker",
            "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8),
        })
    return result


# =============================================================================
# 空间块留站验证
# =============================================================================

def build_spatial_folds(full_params: pd.DataFrame, statistics: pd.DataFrame,
                        requested_folds: int, seed: int) -> pd.DataFrame:
    base = full_params[full_params["scheme"] == "station_reconstruction"][["station_id"]].drop_duplicates()
    coords = statistics[["station_id", "longitude", "latitude"]].copy()
    data = base.merge(coords, on="station_id", how="left")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    valid = data["longitude"].notna() & data["latitude"].notna()
    data["spatial_fold"] = -1
    n_valid = int(valid.sum())
    if n_valid < 2:
        return data
    k = min(max(2, requested_folds), n_valid)
    lon = data.loc[valid, "longitude"].to_numpy(dtype=float)
    lat = data.loc[valid, "latitude"].to_numpy(dtype=float)
    x = lon * np.cos(np.deg2rad(float(np.nanmean(lat))))
    features = np.column_stack([x, lat])
    scale = np.std(features, axis=0)
    scale[scale <= 0] = 1.0
    features = (features - np.mean(features, axis=0)) / scale
    if KMeans is not None and n_valid >= k:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(features)
    else:
        order = np.argsort(x)
        labels = np.empty(n_valid, dtype=int)
        for fold, indices in enumerate(np.array_split(order, k)):
            labels[indices] = fold
    data.loc[valid, "spatial_fold"] = labels.astype(int)
    return data


def satellite_anchor_frame(frame: pd.DataFrame, enable_hampel: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    width_column = "width_model" if "width_model" in frame.columns else "width"
    x, y = collapse_wse_width(frame["wse_model"], frame[width_column])
    return x, y, {
        "target_anchor_count": int(len(x)),
        "target_width_source": "global_station_isotonic" if width_column == "width_model" else "observed_nonnegative",
    }


def run_spatial_validation(files_by_station: Dict[str, str], metadata_by_station: Dict[str, Dict[str, Any]],
                           full_params: pd.DataFrame, statistics: pd.DataFrame,
                           cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = build_spatial_folds(full_params, statistics, cfg.spatial_folds, cfg.seed)
    predictions: List[pd.DataFrame] = []
    parameters: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    transfers: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    local = full_params[full_params["scheme"] == "station_reconstruction"].copy()
    local = local.merge(assignments[["station_id", "spatial_fold"]], on="station_id", how="left")
    folds = sorted(int(x) for x in local["spatial_fold"].dropna().unique() if int(x) >= 0)

    for fold in folds:
        train_stations = local[local["spatial_fold"] != fold]
        test_stations = local[local["spatial_fold"] == fold]
        if train_stations.empty or test_stations.empty:
            continue
        n_transfer = float(np.nanmedian(train_stations["n"]))
        depth_ratio = float(np.nanmedian(train_stations["depth_to_wse_range_ratio"]))
        min_width_factor = float(np.nanmedian(train_stations["min_width_factor"]))
        transfers.append({
            "spatial_fold": fold, "training_station_count": int(train_stations["station_id"].nunique()),
            "validation_station_count": int(test_stations["station_id"].nunique()),
            "n_transferred_median": n_transfer,
            "depth_to_wse_range_ratio_transferred_median": depth_ratio,
            "min_width_factor_transferred_median": min_width_factor,
        })
        fold_predictions: List[pd.DataFrame] = []
        for station_id in test_stations["station_id"].astype(str):
            try:
                path = Path(files_by_station[station_id])
                frame, _ = load_station(path, cfg.aggregate_nodes_by_overpass)
                if cfg.use_global_station_monotonic_width:
                    frame, target_global_diag = apply_global_station_monotonic_width(frame, cfg.enable_hampel)
                else:
                    frame["width_model"] = np.maximum(frame["width"].to_numpy(dtype=float), WIDTH_FLOOR)
                    target_global_diag = {}
                anchor_wse, anchor_width, anchor_diag = satellite_anchor_frame(frame, cfg.enable_hampel)
                if len(anchor_wse) < 2:
                    raise ValueError("目标站有效卫星几何锚点不足 2 个")
                target_span = max(float(np.ptp(anchor_wse)), 0.1)
                depth0 = max(MIN_EFFECTIVE_DEPTH, depth_ratio * target_span)
                min_width_observed, min_width_lower, min_width_upper = effective_min_width_bounds(
                    anchor_width,
                    cfg.min_width_factor_low,
                    cfg.min_width_factor_high,
                )
                min_width_effective = float(np.clip(
                    min_width_factor * min_width_observed,
                    min_width_lower,
                    min_width_upper,
                ))
                anchor_width = apply_effective_min_width(anchor_width, min_width_effective)
                positive_slopes = pd.to_numeric(frame["slope"], errors="coerce")
                positive_slopes = positive_slopes[np.isfinite(positive_slopes) & (positive_slopes > 0)]
                slope_fallback = float(np.nanmedian(positive_slopes)) if len(positive_slopes) else SLOPE_GLOBAL_FALLBACK
                fit = {
                    "n": n_transfer, "depth0": depth0,
                    "WSE0": float(anchor_wse[0] - depth0), "width0": WIDTH0_FIXED,
                    "A0": float(0.5 * min_width_effective * depth0),
                    "min_width_observed": min_width_observed,
                    "min_width_effective": min_width_effective,
                    "min_width_factor": min_width_effective / max(min_width_observed, WIDTH_FLOOR),
                    "anchor_wse": anchor_wse, "anchor_width": anchor_width,
                    "train_slope_fallback": slope_fallback,
                }
                pred = predict_frame(
                    fit, frame, "spatial_block", fold, 0, "spatial_validation",
                    cfg.use_global_station_monotonic_width,
                )
                predictions.append(pred)
                fold_predictions.append(pred)
                parameters.append({
                    "station_id": station_id, "scheme": "spatial_block", "fold_id": fold,
                    "repeat_id": 0, "n": n_transfer, "WSE0": fit["WSE0"],
                    "width0": WIDTH0_FIXED, "depth0": depth0, "A0": fit["A0"],
                    "min_width_observed": min_width_observed,
                    "min_width_effective": min_width_effective,
                    "min_width_factor": fit["min_width_factor"],
                    "train_count_before_qc": int(train_stations["station_id"].nunique()),
                    "validation_count": len(frame), "parameter_source": "training_station_median_transfer",
                    "optimization_unit": "station_then_spatial_parameter_transfer",
                    "width_rule_all_stages": "global_target_station_satellite_isotonic",
                    **anchor_diag, **target_global_diag,
                })
                fold_metrics.append(metric_row(pred, station_id, "spatial_block", fold, 0,
                                               int(train_stations["station_id"].nunique())))
            except Exception as exc:
                failures.append({"station_id": station_id, "scheme": "spatial_block", "fold_id": fold,
                                 "reason": f"{type(exc).__name__}: {exc}"})
        if fold_predictions:
            pooled = pd.concat(fold_predictions, ignore_index=True)
            row = metric_row(pooled, f"__SPATIAL_FOLD_{fold}__", "spatial_block_pooled_fold", fold, 0,
                             int(train_stations["station_id"].nunique()))
            fold_metrics.append(row)
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(parameters), pd.DataFrame(fold_metrics), pd.DataFrame(transfers),
        pd.DataFrame(failures),
    )


# =============================================================================
# 汇总与输出
# =============================================================================

METRIC_COLUMNS = [
    "R2", "NSE", "KGE", "logR2", "logNSE", "RMSE", "MAE", "Bias", "PBIAS",
    "median_signed_log_ratio", "median_abs_log_ratio", "factor_error_median",
]


def build_station_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    keys = ["station_id", "scheme", "repeat_id"]
    for key, group in predictions.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["fold_count"] = int(group["fold_id"].nunique())
        row.update(calculate_metrics(group["Q_observed"], group["Q_calculated"]))
        row["above_train_wse_count"] = int((group["area_position"] == "above_train_max_wse").sum())
        row["below_train_wse_count"] = int((group["area_position"] == "below_train_min_wse").sum())
        row["slope_repaired_count"] = int(group["slope_was_repaired"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_overall_summary(predictions: pd.DataFrame, station_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    for (scheme, repeat_id), group in predictions.groupby(["scheme", "repeat_id"], dropna=False):
        row: Dict[str, Any] = {
            "scheme": scheme, "repeat_id": repeat_id,
            "station_count": int(group["station_id"].nunique()),
            "prediction_count": int(len(group)),
        }
        pooled = calculate_metrics(group["Q_observed"], group["Q_calculated"])
        row.update({f"pooled_{k}": v for k, v in pooled.items()})
        station_group = station_metrics[(station_metrics["scheme"] == scheme) &
                                        (station_metrics["repeat_id"] == repeat_id)]
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(station_group.get(metric), errors="coerce").dropna()
            row[f"station_median_{metric}"] = float(values.median()) if len(values) else np.nan
            row[f"station_mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"station_valid_{metric}_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def build_repeat_uncertainty(station_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"scheme", "station_id", "repeat_id"}
    if station_metrics.empty or not required.issubset(station_metrics.columns):
        return pd.DataFrame()
    repeated = station_metrics[station_metrics["scheme"] == "wse_stratified"]
    if repeated.empty or repeated["repeat_id"].nunique() <= 1:
        return pd.DataFrame()
    rows = []
    for station_id, group in repeated.groupby("station_id"):
        row: Dict[str, Any] = {"station_id": station_id, "repeat_count": int(group["repeat_id"].nunique())}
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_q05"] = float(values.quantile(0.05)) if len(values) else np.nan
            row[f"{metric}_q95"] = float(values.quantile(0.95)) if len(values) else np.nan
            row[f"{metric}_valid_repeat_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def metadata_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    meta = read_csv_safely(path)
    if "ID" not in meta.columns:
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for _, row in meta.iterrows():
        output[normalize_station_id(row["ID"])] = row.to_dict()
    return output


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="n + WSE0 + 最低 WSE 有效宽度 SWOT 流量交叉验证"
    )
    parser.add_argument("--data-dir", default=os.environ.get("SWOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
    parser.add_argument("--metadata-file", default=os.environ.get("SWOT_METADATA_FILE", str(DEFAULT_METADATA_FILE)))
    parser.add_argument("--output-dir", default=os.environ.get("SWOT_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--repeat-random-grouping", dest="repeat", action="store_true",
                        help="启用重复 WSE 分层随机分组")
    parser.add_argument("--no-repeat-random-grouping", dest="repeat", action="store_false")
    parser.set_defaults(repeat=RUN_REPEATED_RANDOM_GROUPING)
    parser.add_argument("--repeat-count", type=int, default=REPEATED_RANDOM_GROUP_COUNT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--objective", choices=("old_composite", "normalized_mse", "log_mse"),
                        default="old_composite")
    parser.add_argument("--no-hampel", action="store_true")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--no-wse-cv", action="store_true")
    parser.add_argument("--no-temporal-cv", action="store_true")
    parser.add_argument("--no-spatial-cv", action="store_true")
    parser.add_argument("--temporal-folds", type=int, default=TEMPORAL_FOLDS)
    parser.add_argument("--spatial-folds", type=int, default=SPATIAL_FOLDS)
    parser.add_argument(
        "--min-width-factor-low", type=float, default=MIN_WIDTH_FACTOR_LOW_DEFAULT,
        help="最低 WSE 有效宽度相对观测单调宽度的搜索下限倍率",
    )
    parser.add_argument(
        "--min-width-factor-high", type=float, default=MIN_WIDTH_FACTOR_HIGH_DEFAULT,
        help="最低 WSE 有效宽度相对观测单调宽度的搜索上限倍率",
    )
    parser.add_argument("--optimizer-maxiter", type=int, default=60)
    parser.add_argument("--optimizer-popsize", type=int, default=8)
    parser.add_argument("--optimizer-tol", type=float, default=1e-3)
    parser.add_argument("--local-optimizer-maxiter", type=int, default=400)
    args = parser.parse_args()
    if args.min_width_factor_low <= 0:
        parser.error("--min-width-factor-low 必须大于 0")
    if args.min_width_factor_high <= args.min_width_factor_low:
        parser.error("--min-width-factor-high 必须大于 --min-width-factor-low")
    return Config(
        data_dir=args.data_dir, metadata_file=args.metadata_file, output_dir=args.output_dir,
        repeat_random_grouping=bool(args.repeat), repeat_count=max(1, args.repeat_count),
        enable_hampel=not args.no_hampel,
        run_reconstruction=not args.no_reconstruction, run_wse_cv=not args.no_wse_cv,
        run_temporal_cv=not args.no_temporal_cv, run_spatial_cv=not args.no_spatial_cv,
        temporal_folds=max(1, args.temporal_folds), spatial_folds=max(2, args.spatial_folds),
        workers=max(1, args.workers), use_parallel=not args.serial,
        max_stations=args.max_stations, fast=bool(args.fast), objective=args.objective,
        min_width_factor_low=float(args.min_width_factor_low),
        min_width_factor_high=float(args.min_width_factor_high),
        optimizer_maxiter=max(1, int(args.optimizer_maxiter)),
        optimizer_popsize=max(5, int(args.optimizer_popsize)),
        optimizer_tol=max(float(args.optimizer_tol), 1e-10),
        local_optimizer_maxiter=max(50, int(args.local_optimizer_maxiter)),
    )


def main() -> int:
    cfg = parse_args()
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output_dir / "run.log", level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", encoding="utf-8",
    )
    with open(output_dir / "run_configuration.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据文件夹不存在: {data_dir}")
    files = sorted(data_dir.glob("*.csv"))
    if cfg.max_stations is not None:
        files = files[:max(0, cfg.max_stations)]
    if not files:
        raise FileNotFoundError(f"没有找到 CSV: {data_dir}")

    metadata = metadata_lookup(Path(cfg.metadata_file))
    files_by_station = {normalize_station_id(p.stem): str(p) for p in files}
    jobs = [(str(p), metadata.get(normalize_station_id(p.stem), {}), cfg) for p in files]
    print(f"站点数: {len(jobs)}; 并行={cfg.use_parallel}; workers={cfg.workers}", flush=True)
    print(f"重复随机分组: {cfg.repeat_random_grouping}; 次数={cfg.repeat_count if cfg.repeat_random_grouping else 1}", flush=True)

    results: List[Dict[str, Any]] = []
    if cfg.use_parallel and cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(process_station, *job): job[0] for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                try:
                    results.append(future.result())
                except Exception as exc:
                    sid = normalize_station_id(Path(futures[future]).stem)
                    results.append({"statistics": None, "parameters": [], "predictions": [], "fold_metrics": [],
                                    "failures": [{"station_id": sid, "scheme": "process_pool", "reason": str(exc)}]})
                if completed % 20 == 0 or completed == len(futures):
                    print(f"站点内实验进度: {completed}/{len(futures)}", flush=True)
    else:
        for i, job in enumerate(jobs, start=1):
            results.append(process_station(*job))
            if i % 10 == 0 or i == len(jobs):
                print(f"站点内实验进度: {i}/{len(jobs)}", flush=True)

    stats = pd.DataFrame([r["statistics"] for r in results if r.get("statistics") is not None])
    params = pd.DataFrame([row for r in results for row in r.get("parameters", [])])
    fold_metrics = pd.DataFrame([row for r in results for row in r.get("fold_metrics", [])])
    failures = pd.DataFrame([row for r in results for row in r.get("failures", [])])
    pred_parts = [x for r in results for x in r.get("predictions", []) if isinstance(x, pd.DataFrame) and not x.empty]
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    assignments = pd.DataFrame()
    transfers = pd.DataFrame()
    if cfg.run_spatial_cv and not params.empty and not stats.empty:
        print("开始空间块留站验证……", flush=True)
        assignments = build_spatial_folds(params, stats, cfg.spatial_folds, cfg.seed)
        spatial_pred, spatial_params, spatial_metrics, transfers, spatial_failures = run_spatial_validation(
            files_by_station, metadata, params, stats, cfg,
        )
        if not spatial_pred.empty:
            predictions = pd.concat([predictions, spatial_pred], ignore_index=True)
        if not spatial_params.empty:
            params = pd.concat([params, spatial_params], ignore_index=True, sort=False)
        if not spatial_metrics.empty:
            fold_metrics = pd.concat([fold_metrics, spatial_metrics], ignore_index=True, sort=False)
        if not spatial_failures.empty:
            failures = pd.concat([failures, spatial_failures], ignore_index=True, sort=False)

    station_metrics = build_station_metrics(predictions)
    overall = build_overall_summary(predictions, station_metrics)
    repeat_uncertainty = build_repeat_uncertainty(station_metrics)

    save_table(stats, tables_dir / "station_data_statistics.csv")
    save_table(params, tables_dir / "parameter_estimates.csv")
    save_table(predictions, tables_dir / "validation_predictions.csv")
    save_table(fold_metrics, tables_dir / "validation_metrics_by_fold.csv")
    save_table(station_metrics, tables_dir / "validation_metrics_by_station.csv")
    save_table(overall, tables_dir / "experiment_overall_summary.csv")
    save_table(repeat_uncertainty, tables_dir / "wse_repeat_uncertainty_by_station.csv")
    save_table(assignments, tables_dir / "spatial_fold_assignments.csv")
    save_table(transfers, tables_dir / "spatial_transfer_parameters.csv")
    save_table(failures, tables_dir / "failed_experiments.csv")

    print(f"完成。输出目录: {output_dir}", flush=True)
    print(f"站点统计: {tables_dir / 'station_data_statistics.csv'}", flush=True)
    print(f"站点级验证: {tables_dir / 'validation_metrics_by_station.csv'}", flush=True)
    print(f"总体汇总: {tables_dir / 'experiment_overall_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        raise SystemExit(130)
