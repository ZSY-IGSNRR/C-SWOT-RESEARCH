#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWOT 逐站流量复现的 beta + (WSE0, width0) 水力核心。

本模块复用 ``swot_n_wse0_width0_nonnegative_cv.py`` 中已经验证过的
数据读取、Hampel、坡度修复、全站单调 width、分组和指标函数，只替换
河床起算几何和参数率定部分。

几何定义
--------
``(WSE0, width0)`` 是拟定河床起点，最低训练 WSE 处的宽度来自该站
全局单调 width--WSE 折线。两点之间采用梯形：

    A0 = 0.5 * (width0 + width_min) * (WSE_min - WSE0)

最低训练 WSE 以上的面积增量由单调 width--WSE 折线做梯形积分，
beta 仅缩放这部分增量：

    A(WSE) = A0 + beta * integral[WSE_min, WSE] width(h) dh

这样 beta 描述“卫星宽度积分对有效过水面积增量的转换”，不会与 n
对整个面积形成完全等价的常数缩放。对低于训练最小 WSE、但高于
WSE0 的观测，直接使用河床起点至当前观测的梯形，不使用 beta，
从而保证在 WSE0 处面积连续地归零。

Manning 宽河道近似：

    Q = (1/n) * A**(5/3) * width**(-2/3) * slope**(1/2)

重要：这些参数是站点级“有效参数”。尤其 WSE0 与 width0 仍可能
具有等效性；应结合 A0、边界命中和多初值诊断解释，不能直接当作
唯一真实河床测量值。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import quality_geometry as quality
import numpy as np
import pandas as pd

import swot_n_wse0_width0_nonnegative_cv as _base
from swot_n_wse0_width0_nonnegative_cv import *  # noqa: F401,F403


BETA_MIN_DEFAULT = 0.25
BETA_MAX_DEFAULT = 4.00
BETA_PRIOR_WEIGHT_DEFAULT = 0.06
WIDTH0_RATIO_MIN_DEFAULT = 0.0
WIDTH0_RATIO_MAX_DEFAULT = 1.0


@dataclass
class Config(_base.Config):
    """在成熟基础配置上增加 beta 与河床宽度搜索设置。"""

    beta_min: float = BETA_MIN_DEFAULT
    beta_max: float = BETA_MAX_DEFAULT
    beta_prior_weight: float = BETA_PRIOR_WEIGHT_DEFAULT
    width0_ratio_min: float = WIDTH0_RATIO_MIN_DEFAULT
    width0_ratio_max: float = WIDTH0_RATIO_MAX_DEFAULT


def prepare_training(
    frame: pd.DataFrame,
    enable_hampel: bool,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    训练期 QC。

    与基础版本逻辑一致，但显式复制 NumPy 数组，兼容 pandas 3 默认
    返回只读 ``to_numpy`` 视图的行为。
    """
    # Global QC was applied consistently to each method before fitting.
    if "global_qc_pass" in frame.columns:
        enable_hampel = False
    work = frame.copy().sort_values(
        "time_dt"
    ).reset_index(drop=True)
    diag: Dict[str, Any] = {
        "train_input_count": int(len(work))
    }
    if len(work) < MIN_TRAIN_RECORDS:
        return None, diag

    wse_bad = (
        hampel_mask(work["wse_model"])
        if enable_hampel
        else np.zeros(len(work), dtype=bool)
    )
    width_bad = (
        hampel_mask(work["width"])
        if enable_hampel
        else np.zeros(len(work), dtype=bool)
    )
    positive_slope = pd.to_numeric(
        work["slope"], errors="coerce"
    ).where(lambda values: values > 0)
    log_slope = np.log10(positive_slope)
    slope_bad = (
        hampel_mask(log_slope)
        if enable_hampel
        else np.zeros(len(work), dtype=bool)
    )
    slope_for_repair = pd.to_numeric(
        work["slope"], errors="coerce"
    ).to_numpy(dtype=float).copy()
    slope_for_repair[slope_bad] = np.nan
    safe_slope, slope_diag = repair_slopes(slope_for_repair)
    work["slope_safe"] = safe_slope

    keep = ~(wse_bad | width_bad)
    work = work.loc[keep].copy().reset_index(drop=True)
    if "width_model" not in work.columns:
        work["width_model"] = np.maximum(
            pd.to_numeric(work["width"], errors="coerce"),
            WIDTH_FLOOR,
        )
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "wse_model", "width", "width_model", "Q", "slope_safe"
        ]
    )
    work = work[
        (work["width"] > 0)
        & (work["width_model"] > 0)
        & (work["Q"] > 0)
    ]
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


def _objective_from_predictions(obs: np.ndarray, sim: np.ndarray, objective: str) -> float:
    if not np.all(np.isfinite(sim)):
        return 1e12
    if objective == "old_composite":
        return -old_composite_score(obs, sim)
    if objective == "normalized_mse":
        scale = max(float(np.std(obs)), 0.1 * float(np.mean(obs)), EPS)
        return float(np.mean(((sim - obs) / scale) ** 2))
    if objective == "log_mse":
        return float(np.mean(
            (np.log10(np.maximum(sim, EPS)) - np.log10(np.maximum(obs, EPS))) ** 2
        ))
    raise ValueError(f"未知目标函数: {objective}")


def _clamp_beta(value: float, cfg: Config) -> float:
    return float(np.clip(float(value), float(cfg.beta_min), float(cfg.beta_max)))


def _width0_limits(minimum_width: float, cfg: Config) -> Tuple[float, float]:
    width_min = max(float(minimum_width), WIDTH_FLOOR)
    low_ratio = float(np.clip(cfg.width0_ratio_min, 0.0, 1.0))
    high_ratio = float(np.clip(cfg.width0_ratio_max, low_ratio, 1.0))
    return width_min * low_ratio, width_min * high_ratio


def geometry_arrays(
    train: pd.DataFrame,
    depth0: float,
    width0: float,
    beta: float = 1.0,
    enforce_monotonic: bool = True,
) -> Dict[str, np.ndarray]:
    """
    建立河床点、观测 width 锚点及其有效累计面积。

    ``area`` 在最低 WSE 以下采用真实梯形 A0，在最低 WSE 以上采用
    ``A0 + beta * delta_area``。因此 beta 是增量面积系数，而不是
    对整个断面面积的常数倍乘。
    """
    anchor_wse, anchor_width = collapse_wse_width(
        train["wse_model"], train["width_model"]
    )
    if len(anchor_wse) == 0:
        raise ValueError("无有效 WSE-width 几何锚点")
    anchor_width = np.maximum(np.asarray(anchor_width, dtype=float), WIDTH_FLOOR)
    if enforce_monotonic:
        anchor_width = np.maximum.accumulate(anchor_width)

    depth = max(float(depth0), DEPTH_MIN)
    bed_width = float(np.clip(float(width0), 0.0, float(anchor_width[0])))
    beta_value = max(float(beta), EPS)
    wse0 = float(anchor_wse[0] - depth)
    a0 = 0.5 * (bed_width + float(anchor_width[0])) * depth

    delta = np.zeros(len(anchor_wse), dtype=float)
    if len(anchor_wse) > 1:
        delta[1:] = np.cumsum(
            np.maximum(np.diff(anchor_wse), 0.0)
            * (anchor_width[:-1] + anchor_width[1:])
            / 2.0
        )
    effective_area = np.maximum(a0 + beta_value * delta, AREA_FLOOR)
    return {
        "wse": np.concatenate([[wse0], anchor_wse]),
        "width": np.concatenate([[bed_width], anchor_width]),
        "area": np.concatenate([[0.0], effective_area]),
        "wse0": np.array([wse0]),
        "width0": np.array([bed_width]),
        "A0": np.array([a0]),
        "beta": np.array([beta_value]),
        "anchor_wse": anchor_wse,
        "anchor_width": anchor_width,
        "delta_area": delta,
    }


def _positive_increment_from_min(
    anchor_wse: np.ndarray,
    anchor_width: np.ndarray,
    test_wse: float,
    test_width: float,
) -> float:
    """从最低训练 WSE 到当前 WSE 的正向折线梯形积分。"""
    wse_min = float(anchor_wse[0])
    xt = float(test_wse)
    yt = max(float(test_width), WIDTH_FLOOR)
    if xt <= wse_min:
        return 0.0
    inside = (anchor_wse > wse_min) & (anchor_wse < xt)
    x = np.concatenate([[wse_min], anchor_wse[inside], [xt]])
    y = np.concatenate([[float(anchor_width[0])], anchor_width[inside], [yt]])
    order = np.argsort(x)
    x, y = collapse_wse_width(x[order], y[order])
    if len(x) <= 1:
        return 0.0
    return float(np.sum(
        np.maximum(np.diff(x), 0.0) * (y[:-1] + y[1:]) / 2.0
    ))


def area_at_observation(
    anchor_wse: np.ndarray,
    anchor_width: np.ndarray,
    wse0: float,
    width0: float,
    beta: float,
    test_wse: float,
    test_width: float,
) -> Tuple[float, str, float]:
    """
    计算单个观测的有效面积，并返回未缩放增量面积。

    返回 ``(area, position, delta_area_raw)``。低于训练最小 WSE 时，
    ``delta_area_raw`` 为 NaN，因为该段使用河床点至观测点的直接
    梯形而不是 beta 增量模型。
    """
    xt = float(test_wse)
    yt = max(float(test_width), WIDTH_FLOOR)
    bed_wse = float(wse0)
    bed_width = max(float(width0), 0.0)
    if not np.isfinite(xt) or not np.isfinite(yt):
        return np.nan, "invalid", np.nan
    if xt <= bed_wse:
        return AREA_FLOOR, "below_wse0", np.nan

    wse_min = float(anchor_wse[0])
    width_min = float(anchor_width[0])
    if xt < wse_min:
        # 河床点到当前水面的梯形；保证 WSE0 处面积严格为 0。
        area = 0.5 * (bed_width + yt) * (xt - bed_wse)
        return max(float(area), AREA_FLOOR), "below_train_min_wse", np.nan

    depth0 = max(wse_min - bed_wse, 0.0)
    a0 = 0.5 * (bed_width + width_min) * depth0
    delta = _positive_increment_from_min(
        anchor_wse, anchor_width, xt, yt
    )
    area = a0 + max(float(beta), EPS) * delta
    position = (
        "above_train_max_wse"
        if xt > float(anchor_wse[-1])
        else "within_train_wse_range"
    )
    return max(float(area), AREA_FLOOR), position, float(delta)


def training_predictions(
    train: pd.DataFrame,
    n_value: float,
    depth0: float,
    width0: float,
    beta: float = 1.0,
    enforce_monotonic: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    geom = geometry_arrays(
        train, depth0, width0, beta, enforce_monotonic=enforce_monotonic
    )
    wse = train["wse_model"].to_numpy(dtype=float)
    areas = np.interp(wse, geom["wse"], geom["area"])
    width_for_q = train["width_model"].to_numpy(dtype=float).copy()
    minimum_mask = np.isclose(
        wse, float(geom["anchor_wse"][0]), rtol=1e-9, atol=1e-8
    )
    width_for_q[minimum_mask] = float(geom["anchor_width"][0])
    q = manning_q(
        areas,
        width_for_q,
        train["slope_safe"].to_numpy(dtype=float),
        n_value,
    )
    return q, geom


def fit_parameters(
    train_input: pd.DataFrame,
    cfg: Config,
    fit_seed: int,
    *,
    n_fixed_value: Optional[float] = None,
    beta_fixed_value: Optional[float] = None,
    depth0_fixed_value: Optional[float] = None,
    width0_fixed_value: Optional[float] = None,
    parameter_source: str = "station_training_Q_joint_n_beta_bed_point",
) -> Optional[Dict[str, Any]]:
    """
    逐站联合率定 n、beta、WSE0/depth0 与 width0。

    通过可选 fixed 参数，同一函数也支持 beta=1、固定参考 n 和
    仅优化 n 的严格消融，确保各方法使用一致的目标与数值求解器。
    """
    train, qc = prepare_training(train_input, cfg.enable_hampel)
    if train is None:
        return None

    anchor_wse, anchor_width = collapse_wse_width(
        train["wse_model"], train["width_model"]
    )
    anchor_width = np.maximum(np.asarray(anchor_width, dtype=float), WIDTH_FLOOR)
    if cfg.use_global_station_monotonic_width:
        anchor_width = np.maximum.accumulate(anchor_width)
    min_width = float(anchor_width[0])
    width0_lower, width0_upper = _width0_limits(min_width, cfg)
    wse_span = max(float(np.ptp(anchor_wse)), 0.1)
    depth_upper = min(DEPTH_ABSOLUTE_MAX, max(50.0, 20.0 * wse_span))

    names = []
    bounds = []
    if n_fixed_value is None:
        names.append("log_n")
        bounds.append((math.log(N_MIN), math.log(N_MAX)))
    if depth0_fixed_value is None:
        names.append("log_depth0")
        bounds.append((math.log(DEPTH_MIN), math.log(depth_upper)))
    if width0_fixed_value is None:
        names.append("width0_ratio")
        bounds.append((
            float(np.clip(cfg.width0_ratio_min, 0.0, 1.0)),
            float(np.clip(cfg.width0_ratio_max, 0.0, 1.0)),
        ))
    if beta_fixed_value is None:
        names.append("log_beta")
        bounds.append((math.log(cfg.beta_min), math.log(cfg.beta_max)))

    def decode(vector: Sequence[float]) -> Dict[str, float]:
        values = dict(zip(names, np.asarray(vector, dtype=float)))
        n_value = (
            float(np.clip(n_fixed_value, N_MIN, N_MAX))
            if n_fixed_value is not None
            else math.exp(values["log_n"])
        )
        depth0 = (
            max(float(depth0_fixed_value), DEPTH_MIN)
            if depth0_fixed_value is not None
            else math.exp(values["log_depth0"])
        )
        width0 = (
            float(np.clip(width0_fixed_value, width0_lower, width0_upper))
            if width0_fixed_value is not None
            else min_width * values["width0_ratio"]
        )
        beta = (
            _clamp_beta(beta_fixed_value, cfg)
            if beta_fixed_value is not None
            else math.exp(values["log_beta"])
        )
        return {
            "n": n_value,
            "depth0": depth0,
            "width0": width0,
            "beta": beta,
        }

    obs = train["Q"].to_numpy(dtype=float)

    def loss(vector: np.ndarray) -> float:
        try:
            p = decode(vector)
            sim, _ = training_predictions(
                train,
                p["n"],
                p["depth0"],
                p["width0"],
                p["beta"],
                enforce_monotonic=cfg.use_global_station_monotonic_width,
            )
            value = _objective_from_predictions(obs, sim, cfg.objective)
            if beta_fixed_value is None and cfg.beta_prior_weight > 0:
                # 与旧 beta 框架一致的弱先验，仅抑制无证据的极端 beta。
                value += float(cfg.beta_prior_weight) * abs(math.log10(p["beta"]))
            return float(value) if np.isfinite(value) else 1e12
        except Exception:
            return 1e12

    if bounds:
        result = hybrid_global_local_optimize(
            loss,
            bounds=bounds,
            args=(),
            seed=int(fit_seed),
            fast=cfg.fast,
            maxiter=cfg.optimizer_maxiter,
            popsize=cfg.optimizer_popsize,
            tol=cfg.optimizer_tol,
            local_maxiter=cfg.local_optimizer_maxiter,
        )
        if not np.isfinite(result["fun"]):
            return None
        fitted = decode(result["x"])
    else:
        fitted = decode([])
        fixed_loss = loss(np.array([], dtype=float))
        result = {
            "x": np.array([], dtype=float),
            "fun": fixed_loss,
            "success": np.isfinite(fixed_loss),
            "fit_accepted": np.isfinite(fixed_loss),
            "global_success": True,
            "local_success": True,
            "selected_stage": "fixed",
            "message": "all parameters fixed",
            "nfev": 1,
            "nit": 0,
        }

    qtrain, geom = training_predictions(
        train,
        fitted["n"],
        fitted["depth0"],
        fitted["width0"],
        fitted["beta"],
        enforce_monotonic=cfg.use_global_station_monotonic_width,
    )
    train_metrics = calculate_metrics(train["Q"], qtrain)
    n_value = float(fitted["n"])
    depth0 = float(fitted["depth0"])
    width0 = float(fitted["width0"])
    beta = float(fitted["beta"])
    wse0 = float(anchor_wse[0] - depth0)
    a0 = float(0.5 * (width0 + min_width) * depth0)

    active_values = dict(zip(names, np.asarray(result["x"], dtype=float)))
    bound_hits: Dict[str, Tuple[bool, bool]] = {}
    for name, (low, high) in zip(names, bounds):
        value = active_values[name]
        span = max(high - low, EPS)
        bound_hits[name] = (
            bool((value - low) <= 0.01 * span),
            bool((high - value) <= 0.01 * span),
        )

    n_hits = bound_hits.get("log_n", (False, False))
    depth_hits = bound_hits.get("log_depth0", (False, False))
    width_hits = bound_hits.get("width0_ratio", (False, False))
    beta_hits = bound_hits.get("log_beta", (False, False))
    return {
        **quality.curve_metadata(train_input),
        "model_kind": "hydraulic",
        "parameter_source": parameter_source,
        "n": n_value,
        "beta": beta,
        "WSE0": wse0,
        "width0": width0,
        "depth0": depth0,
        "A0": a0,
        "train_wse_min": float(anchor_wse[0]),
        "train_wse_max": float(anchor_wse[-1]),
        "train_wse_range": float(np.ptp(anchor_wse)),
        "train_width_at_min_wse": min_width,
        # 保留兼容列：最低观测宽度不再参与率定。
        "min_width_observed": min_width,
        "min_width_effective": min_width,
        "min_width_factor": 1.0,
        "depth_to_wse_range_ratio": float(depth0 / wse_span),
        "width0_to_min_width_ratio": float(width0 / max(min_width, WIDTH_FLOOR)),
        "beta_prior_penalty": float(
            cfg.beta_prior_weight * abs(math.log10(beta))
            if beta_fixed_value is None else 0.0
        ),
        "objective": cfg.objective,
        "objective_loss": float(result["fun"]),
        "optimizer_success": bool(result["success"]),
        "fit_accepted": bool(result["fit_accepted"]),
        "global_optimizer_success": bool(result["global_success"]),
        "local_optimizer_success": bool(result["local_success"]),
        "optimizer_selected_stage": result.get("selected_stage", ""),
        "optimizer_message": str(result["message"]),
        "optimizer_nfev": int(result["nfev"]),
        "optimizer_nit": int(result["nit"]),
        "n_in_typical_range": bool(N_TYPICAL_MIN <= n_value <= N_TYPICAL_MAX),
        "n_above_typical_range": bool(n_value > N_TYPICAL_MAX),
        "n_fixed": bool(n_fixed_value is not None),
        "beta_fixed": bool(beta_fixed_value is not None),
        "depth0_fixed": bool(depth0_fixed_value is not None),
        "width0_fixed": bool(width0_fixed_value is not None),
        "n_at_lower_bound": bool(n_hits[0]),
        "n_at_upper_bound": bool(n_hits[1]),
        "beta_at_lower_bound": bool(beta_hits[0]),
        "beta_at_upper_bound": bool(beta_hits[1]),
        "depth_at_lower_bound": bool(depth_hits[0]),
        "depth_at_upper_bound": bool(depth_hits[1]),
        "width0_at_lower_bound": bool(width_hits[0]),
        "width0_at_upper_bound": bool(width_hits[1]),
        "width0_fixed_zero": bool(
            width0_fixed_value is not None and abs(float(width0_fixed_value)) <= EPS
        ),
        "min_width_at_lower_bound": False,
        "min_width_at_upper_bound": False,
        "n_search_lower": N_MIN,
        "n_search_upper": N_MAX,
        "beta_search_lower": float(cfg.beta_min),
        "beta_search_upper": float(cfg.beta_max),
        "depth0_search_lower": DEPTH_MIN,
        "depth0_search_upper": float(depth_upper),
        "width0_search_lower": float(width0_lower),
        "width0_search_upper": float(width0_upper),
        "A0_search_lower": float(
            0.5 * (width0_lower + min_width) * DEPTH_MIN
        ),
        "A0_search_upper": float(
            0.5 * (width0_upper + min_width) * depth_upper
        ),
        "bed_point_definition": "(WSE0,width0); trapezoid to minimum training WSE",
        "beta_definition": "multiplier of width-integrated area above minimum training WSE",
        "width_curve_scope": (
            str(train_input["width_curve_scope"].iloc[0])
            if "width_curve_scope" in train_input.columns and len(train_input)
            else "not_recorded"
        ),
        "width_curve_training_count": (
            float(pd.to_numeric(
                train_input["width_curve_training_count"], errors="coerce"
            ).dropna().iloc[0])
            if "width_curve_training_count" in train_input.columns
            and pd.to_numeric(
                train_input["width_curve_training_count"], errors="coerce"
            ).notna().any()
            else np.nan
        ),
        "anchor_wse": np.asarray(geom["anchor_wse"], dtype=float),
        "anchor_width": np.asarray(geom["anchor_width"], dtype=float),
        "train_slope_fallback": float(qc["slope_fallback"]),
        "train_metrics": train_metrics,
        "qc": qc,
    }


def predict_frame(
    fit: Mapping[str, Any],
    frame: pd.DataFrame,
    scheme: str,
    fold_id: int,
    repeat_id: int,
    sample_role: str,
    use_global_station_monotonic_width: bool = True,
) -> pd.DataFrame:
    rows = []
    anchor_wse = np.asarray(fit["anchor_wse"], dtype=float)
    anchor_width = np.asarray(fit["anchor_width"], dtype=float)
    beta = float(fit.get("beta", 1.0))
    for _, row in frame.iterrows():
        slope_raw = (
            float(row["slope"]) if np.isfinite(row["slope"]) else np.nan
        )
        slope_safe, slope_diag = repair_slopes(
            [slope_raw], fallback=fit["train_slope_fallback"]
        )
        width_raw = float(row["width"])
        if (
            use_global_station_monotonic_width
            and "width_global_monotonic" in row.index
        ):
            width_used = max(
                float(row["width_global_monotonic"]), WIDTH_FLOOR
            )
            width_source = str(row.get(
                "width_model_source", "global_station_monotonic_width_wse"
            ))
        else:
            width_used = max(width_raw, WIDTH_FLOOR)
            width_source = "observed_nonnegative_width"

        area, position, delta_area_raw = area_at_observation(
            anchor_wse,
            anchor_width,
            float(fit["WSE0"]),
            float(fit["width0"]),
            beta,
            float(row["wse_model"]),
            width_used,
        )
        qcalc = float(manning_q(
            np.array([area]),
            np.array([width_used]),
            slope_safe,
            float(fit["n"]),
        )[0])
        rows.append({
            "station_id": row["station_id"],
            "obs_id": row["obs_id"],
            "time": row["time_dt"],
            "scheme": scheme,
            "fold_id": fold_id,
            "repeat_id": repeat_id,
            "sample_role": sample_role,
            "WSE": float(row["wse_model"]),
            "width_raw": width_raw,
            "width_used": width_used,
            "width_source": width_source,
            "width_adjustment": float(width_used - width_raw),
            "width_adjustment_percent": (
                float(100.0 * (width_used - width_raw) / width_raw)
                if width_raw > 0 else np.nan
            ),
            "width_curve_scope": row.get("width_curve_scope", "not_recorded"),
            "width_curve_position": row.get("width_curve_position", "not_recorded"),
            "width_curve_wse_min": finite_number(row.get("width_curve_wse_min")),
            "width_curve_wse_max": finite_number(row.get("width_curve_wse_max")),
            "width_curve_training_count": finite_number(
                row.get("width_curve_training_count")
            ),
            "slope_raw": slope_raw,
            "slope_used": float(slope_safe[0]),
            "slope_was_repaired": bool(
                slope_diag["slope_repaired_count"] > 0
            ),
            "area": float(area),
            "delta_area_raw": delta_area_raw,
            "area_position": position,
            "Q_observed": float(row["Q"]),
            "Q_calculated": qcalc,
            "n": float(fit["n"]),
            "beta": beta,
            "WSE0": float(fit["WSE0"]),
            "width0": float(fit["width0"]),
            "depth0": float(fit["depth0"]),
            "A0": float(fit["A0"]),
            "min_width_observed": float(
                fit.get("min_width_observed", anchor_width[0])
            ),
            "min_width_effective": float(
                fit.get("min_width_effective", anchor_width[0])
            ),
            "min_width_factor": float(fit.get("min_width_factor", 1.0)),
        })
    return pd.DataFrame(rows)


def q_for_beta(
    prediction_rows: pd.DataFrame,
    beta_test: float,
) -> np.ndarray:
    """
    在 n、河床点、width 和 slope 固定时重算指定 beta 的 Q。

    该函数用于 beta 敏感性表。低于训练最小 WSE 的记录按定义不使用
    beta，因此保留其原面积。
    """
    beta_value = max(float(beta_test), EPS)
    a0 = pd.to_numeric(prediction_rows["A0"], errors="coerce").to_numpy(float)
    area_old = pd.to_numeric(
        prediction_rows["area"], errors="coerce"
    ).to_numpy(float)
    delta = pd.to_numeric(
        prediction_rows.get("delta_area_raw"), errors="coerce"
    ).to_numpy(float)
    scalable = np.isfinite(delta)
    area_new = area_old.copy()
    area_new[scalable] = a0[scalable] + beta_value * delta[scalable]
    return manning_q(
        area_new,
        pd.to_numeric(
            prediction_rows["width_used"], errors="coerce"
        ).to_numpy(float),
        pd.to_numeric(
            prediction_rows["slope_used"], errors="coerce"
        ).to_numpy(float),
        # 本函数按一个站点调用；n 在组内为常数。
        float(pd.to_numeric(
            prediction_rows["n"], errors="coerce"
        ).dropna().iloc[0]),
    )
