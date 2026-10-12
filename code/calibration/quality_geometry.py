"""Satellite-only full-record QC and auditable, calibration-derived width rules.

Global QC is retrospective: centered windows may include later observations.
Neither QC nor curve construction reads gauge discharge or prediction error.
All limits are configurable experimental assumptions, not universal river laws.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class Rules:
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


RULES = Rules()


def configure(cfg):
    global RULES
    RULES = Rules(**{name: getattr(cfg, name, default) for name, default in asdict(Rules()).items()})


def hampel_flags(values, floor, rules=None):
    """Exact local MAD; truncated endpoint windows; finite floor for zero MAD."""
    rules = rules or RULES
    x = np.asarray(values, dtype=float)
    bad = np.zeros(len(x), dtype=bool)
    centers = np.full(len(x), np.nan)
    scales = np.full(len(x), np.nan)
    for i, value in enumerate(x):
        window = x[max(0, i-rules.hampel_half_window):i+rules.hampel_half_window+1]
        window = window[np.isfinite(window)]
        if len(window) < rules.hampel_min_points or not np.isfinite(value):
            continue
        med = float(np.median(window))
        scale = max(1.4826 * float(np.median(np.abs(window-med))), floor)
        centers[i], scales[i] = med, scale
        bad[i] = abs(value-med) > rules.hampel_sigma * scale
    return bad, centers, scales


def annotate(frame):
    work = frame.copy().sort_values('time_dt').reset_index(drop=True)
    values = {
        'wse': (pd.to_numeric(work['wse_model'], errors='coerce').to_numpy(float), RULES.wse_scale_floor_m),
        'width': (pd.to_numeric(work['width'], errors='coerce').to_numpy(float), RULES.width_scale_floor_m),
        'log_slope': (np.log10(pd.to_numeric(work['slope'], errors='coerce').where(lambda x:x>0)).to_numpy(float), RULES.slope_log_scale_floor),
    }
    for name, (arr, floor) in values.items():
        bad, center, scale = hampel_flags(arr, floor)
        work['global_hampel_'+name] = bad
        work['global_hampel_'+name+'_median'] = center
        work['global_hampel_'+name+'_scale'] = scale
    invalid = ~np.isfinite(work['slope'].to_numpy(float)) | (work['slope'].to_numpy(float) <= 0)
    work['global_invalid_slope'] = invalid
    fields = ['global_hampel_wse','global_hampel_width','global_hampel_log_slope','global_invalid_slope']
    work['global_qc_pass'] = ~work[fields].any(axis=1)
    work['global_qc_reason'] = [';'.join(f for f in fields if bool(row[f])) for _,row in work.iterrows()]
    return work


def retain(frame, method='full'):
    if 'global_qc_pass' not in frame:
        raise ValueError('Full-record QC must be annotated before splitting.')
    # Invalid slopes are a shared basic eligibility rule. The ablation changes
    # only Hampel rejection; it does not gain a separate invalid-slope fallback.
    return frame.loc[~frame.global_invalid_slope].copy() if method=='no_hampel' else frame.loc[frame.global_qc_pass].copy()


def curve(frame, initial=True, growth=True):
    """Build geometry from calibration covariates only, preserving each edit."""
    data = frame[['wse_model','width']].replace([np.inf,-np.inf],np.nan).dropna()
    data = data[data.width>0].groupby('wse_model',sort=True).width.median()
    x, raw = data.index.to_numpy(float), data.to_numpy(float)
    if not len(x):
        return None
    edited = raw.copy()
    neighbors = np.arange(1, min(len(x),RULES.initial_neighbor_count+1))
    neighbors = neighbors[x[neighbors]-x[0] <= RULES.initial_neighbor_radius_m]
    reference = float(np.median(raw[neighbors])) if len(neighbors) else np.nan
    first_changed = False
    if initial and len(neighbors)>=RULES.initial_min_neighbors and raw[0] > RULES.initial_width_ratio*reference:
        # Correct only a locally inconsistent first width. A consistently wide
        # channel remains wide; no arbitrary small initial width is imposed.
        edited[0] = reference
        first_changed = True
    isotonic = IsotonicRegression(increasing=True).fit_transform(x,edited) if len(x)>1 else edited.copy()
    bounded = np.asarray(isotonic,dtype=float).copy()
    if growth:
        for i in range(1,len(x)):
            gradient = min(RULES.width_max_growth_m_per_m, RULES.width_max_relative_growth_per_m*bounded[i-1])
            bounded[i] = min(bounded[i],bounded[i-1]+gradient*(x[i]-x[i-1]))
    return {'curve_wse':x,'curve_width':bounded,'calibration_count':len(frame),
            'initial_guard':initial,'growth_guard':growth,'first_width_corrected':first_changed,
            'initial_neighbor_count':len(neighbors),'initial_neighbor_width':reference,
            'raw_width':raw,'initial_corrected_width':edited,'isotonic_width':isotonic,
            'growth_capped':bounded < isotonic-1e-9,'rules':asdict(RULES)}


def apply_curve(frame, model, training=True, use_validation_curve=True):
    work = frame.copy()
    if model is None:
        work['width_model'] = work['width'].astype(float)
        work['width_curve_scope'] = 'unavailable'
        return work
    x,y = model['curve_wse'],model['curve_width']
    predicted = np.interp(work['wse_model'].to_numpy(float),x,y)
    use = training or use_validation_curve
    work['width_model'] = predicted if use else work['width'].to_numpy(float)
    if use:
        work['width_global_monotonic'] = predicted
    else:
        work = work.drop(columns=['width_global_monotonic'],errors='ignore')
    work['width_model_source'] = 'calibration_bounded_width_curve' if use else 'observed_validation_width_training_curve_only'
    work['width_curve_scope'] = 'calibration_only'
    work['width_curve_wse_min'],work['width_curve_wse_max'] = float(x[0]),float(x[-1])
    work['width_curve_training_count'] = model['calibration_count']
    work['width_curve_position'] = np.where(work.wse_model<x[0],'below_calibration_curve',np.where(work.wse_model>x[-1],'above_calibration_curve','within_calibration_curve'))
    work.attrs['width_curve_json'] = json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()},allow_nan=True)
    return work


def curve_metadata(frame):
    encoded=frame.attrs.get('width_curve_json')
    model=json.loads(encoded) if encoded else None
    if model is None:return {}
    keys=['curve_wse','raw_width','initial_corrected_width','isotonic_width','curve_width','growth_capped']
    return {'width_curve_parameters_json':json.dumps({k:np.asarray(model[k]).tolist() for k in keys},allow_nan=False),
            'width_rules_json':json.dumps(model['rules']),
            'initial_width_guard':model['initial_guard'],'width_growth_guard':model['growth_guard'],
            'initial_width_corrected':model['first_width_corrected'],
            'initial_neighbor_count':model['initial_neighbor_count'],
            'width_growth_capped_anchor_count':int(np.sum(model['growth_capped']))}
