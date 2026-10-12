"""Shared, explicit metrics for the SWOT-5 post-processing package."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

SCHEMES = ['wse_stratified', 'temporal_forward']
OBS = ['station_id', 'validation_scheme', 'obs_id']
UNIT = ['station_id', 'validation_scheme', 'repeat_id']
METRICS = ['NSE', 'R2', 'MAE', 'RMSE', 'MdAPE', 'KGE', 'factor2_fraction']
GROUPS = {
    'flow_group': ['Recorded zero', '(0,10]', '(10,100]', '(100,1000]', '>1000'],
    'width_group': ['<100', '100-<200', '200-<500', '>=500'],
    'event_flow_group': ['(0,10]', '(10,100]', '(100,1000]', '>1000'],
    'region': ['NW', 'NE', 'SW', 'SE'],
}
PRIMARY = ['baseline', 'slope10', 'error10']
RULES = ['baseline', 'slope5', 'slope10', 'slope20', 'error5', 'error10', 'error20', 'absolute_top1']

def normalize(s):
    return s.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)

def truth(s):
    return s.astype('string').str.lower().isin(['true', '1', '1.0'])

def read_csv(path, **kwargs):
    d = pd.read_csv(path, dtype={'station_id': 'string'}, **kwargs)
    if 'station_id' in d:
        d['station_id'] = normalize(d.station_id)
    return d

def save(d, directory, name):
    d.to_csv(Path(directory) / (name + '.csv'), index=False, encoding='utf-8-sig')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def metrics(observed, calculated, min_pairs=5):
    """All metrics share a finite-pair mask; Qobs must be positive.

    Variance metrics follow the main experiment's nonconstant-observed AND
    nonconstant-predicted rule. Counts are retained even below min_pairs.
    """
    o, p = np.asarray(observed, float), np.asarray(calculated, float)
    use = np.isfinite(o) & np.isfinite(p) & (o > 0)
    o, p = o[use], p[use]
    r = {'n': len(o), **dict.fromkeys(METRICS, np.nan)}
    if len(o) < min_pairs:
        return r
    e = p - o
    r.update(MAE=float(np.mean(abs(e))), RMSE=float(np.sqrt(np.mean(e**2))),
             MdAPE=float(np.median(100 * abs(e) / o)),
             factor2_fraction=float(np.mean((p / o >= .5) & (p / o <= 2))))
    if len(o) >= 2 and np.std(o) > 0 and np.std(p) > 0:
        cor = np.corrcoef(o, p)[0, 1]
        r.update(NSE=float(1 - np.sum(e**2) / np.sum((o-o.mean())**2)),
                 R2=float(cor**2), KGE=float(1-np.sqrt((cor-1)**2 +
                 (np.std(p)/np.std(o)-1)**2 + (p.mean()/o.mean()-1)**2)))
    return r

def interval(values, iterations=2000, seed=922):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {'stations': 0, 'median': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
    r = {'stations': len(v), 'median': float(np.median(v)), 'ci_low': np.nan, 'ci_high': np.nan}
    if len(v) >= 5 and iterations > 0:
        rng = np.random.default_rng(seed)
        bs = np.concatenate([np.median(rng.choice(v, (min(200, iterations-i), len(v))), axis=1)
                             for i in range(0, iterations, 200)])
        r.update(zip(['ci_low', 'ci_high'], np.quantile(bs, [.025, .975])))
    return r

def aggregate_units(frame, group_cols=UNIT, pred='Q_calculated', min_pairs=5):
    rows = [dict(zip(group_cols, key if isinstance(key, tuple) else (key,)),
                 **metrics(g.Q_observed, g[pred], min_pairs))
            for key, g in frame.groupby(group_cols, sort=True)]
    return pd.DataFrame(rows, columns=group_cols + ['n'] + METRICS)

def assign_groups(meta):
    d = meta.copy()
    d['flow_group'] = pd.cut(d.gauge_Q50, [0, 10, 100, 1000, np.inf],
                            labels=GROUPS['flow_group'][1:]).astype(object)
    d.loc[d.gauge_Q50.eq(0), 'flow_group'] = 'Recorded zero'
    d.loc[d.gauge_days.lt(270), 'flow_group'] = np.nan
    d['width_group'] = pd.cut(d.sword_width_m.where(d.sword_width_m.gt(0)),
                             [0, 100, 200, 500, np.inf], right=False,
                             labels=GROUPS['width_group']).astype(object)
    d['region'] = np.where(d.latitude.ge(35), 'N', 'S') + np.where(d.longitude.ge(105), 'E', 'W')
    d.loc[d.longitude.isna() | d.latitude.isna(), 'region'] = np.nan
    return d

def keep_rule(d, rule):
    return d if rule == 'baseline' else d.loc[~d[rule].astype(bool)]

def grouped_summary(stations, iterations, dimensions=('flow_group', 'width_group', 'region')):
    rows = []
    for key, d in stations.groupby(['validation_scheme', 'comparison', 'variant', 'metric']):
        info = dict(zip(['validation_scheme', 'comparison', 'variant', 'metric'], key))
        for variable in ('all', *dimensions):
            groups = [('All', d)] if variable == 'all' else d.groupby(variable, dropna=False)
            for label, g in groups:
                rows.append(dict(info, group_variable=variable, group=str(label),
                                 **interval(g.value, iterations)))
    return pd.DataFrame(rows)
