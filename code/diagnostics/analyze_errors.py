"""Compare saved held-out predictions on identical residual-screened rows.

Arguments: project root, output directory. No model is refitted.
"""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'postprocessing'))
from common import OBS, UNIT, METRICS, metrics, interval, normalize, truth, keep_rule

rules = ['baseline', 'error10', 'absolute_top1', 'slope10']
flags = rules[1:]
obs = pd.read_csv(root / 'postprocess_results/tables/observation_diagnostics.csv',
    usecols=OBS + ['Q_observed', 'Q_model', 'absolute_error', 'squared_error',
                  'primary_cohort'] + flags,
    dtype={'station_id': 'string', 'obs_id': 'string'})
obs['station_id'] = normalize(obs.station_id)
for col in flags + ['primary_cohort']:
    obs[col] = truth(obs[col])
obs = obs[obs.validation_scheme.eq('wse_stratified')].copy()
assert not obs.duplicated(OBS).any()

columns = OBS + ['repeat_id', 'method', 'Q_observed', 'Q_calculated']
parts = []
for chunk in pd.read_csv(root / 'results/tables/validation_predictions.csv',
    usecols=columns, dtype={'station_id': 'string', 'obs_id': 'string'},
    chunksize=100000):
    selected = chunk[chunk.validation_scheme.eq('wse_stratified') &
                     chunk.method.isin(['full', 'empirical_rating_curve'])].copy()
    if len(selected):
        selected['station_id'] = normalize(selected.station_id)
        parts.append(selected)
pred = pd.concat(parts, ignore_index=True)
key = OBS + ['repeat_id']
f = pred[pred.method.eq('full')].copy()
e = pred[pred.method.eq('empirical_rating_curve')].copy()
assert not f.duplicated(key).any() and not e.duplicated(key).any()
d = f.merge(e[key + ['Q_observed', 'Q_calculated']], on=key,
            suffixes=('', '_empirical'), validate='one_to_one')
assert len(d) == len(f) == len(e)
assert np.allclose(d.Q_observed, d.Q_observed_empirical)
d = d.merge(obs[OBS + flags + ['primary_cohort']], on=OBS, validate='many_to_one')
assert len(d) == len(f)
assert d[['Q_observed', 'Q_calculated', 'Q_calculated_empirical']].notna().all().all()

# Each station-overpass is represented once by the median over saved repeats.
collapsed = d.groupby(OBS, as_index=False).agg(
    Q_observed=('Q_observed', 'first'),
    Q_calculated=('Q_calculated', 'median'),
    Q_calculated_empirical=('Q_calculated_empirical', 'median'))
collapsed = collapsed.merge(obs[OBS + flags + ['primary_cohort', 'Q_model']],
                            on=OBS, validate='one_to_one')
assert np.allclose(collapsed.Q_calculated, collapsed.Q_model)
cohort = collapsed[collapsed.primary_cohort].copy()
assert len(cohort) == 6152
pooled = []
for rule in rules:
    g = keep_rule(cohort, rule)
    for model, col in [('hydraulic', 'Q_calculated'),
                       ('empirical', 'Q_calculated_empirical')]:
        pooled.append(dict(rule=rule, model=model,
            observations_removed=len(cohort)-len(g), retention=len(g)/len(cohort),
            **metrics(g.Q_observed, g[col], min_pairs=1)))
pooled = pd.DataFrame(pooled)
saved = pd.read_csv(root / 'postprocess_results/tables/pooled_screening_metrics.csv')
saved = saved[saved.validation_scheme.eq('wse_stratified')]
for row in pooled[pooled.model.eq('hydraulic')].itertuples():
    reference = saved[saved.rule.eq(row.rule)].iloc[0]
    for metric in METRICS:
        assert np.isclose(getattr(row, metric), reference[metric], rtol=1e-9, atol=1e-9)
pooled.to_csv(out / 'pooled_same_observations.csv', index=False, encoding='utf-8-sig')

# Paired station-repeat metrics, then station medians across repeats. Stations
# have equal weight. Error-metric advantages are reversed so positive is better.
records = []
for rule in rules:
    for unit, g in keep_rule(d, rule).groupby(UNIT, sort=True):
        a = metrics(g.Q_observed, g.Q_calculated, min_pairs=5)
        b = metrics(g.Q_observed, g.Q_calculated_empirical, min_pairs=5)
        for metric in METRICS:
            if np.isfinite(a[metric]) and np.isfinite(b[metric]):
                sign = -1 if metric in ['MAE', 'RMSE', 'MdAPE'] else 1
                records.append(dict(station_id=unit[0], repeat_id=unit[2],
                    rule=rule, metric=metric, hydraulic=a[metric], empirical=b[metric],
                    advantage=sign*(a[metric]-b[metric])))
paired = pd.DataFrame(records)
station = paired.groupby(['station_id', 'rule', 'metric'], as_index=False).agg(
    hydraulic=('hydraulic', 'median'), empirical=('empirical', 'median'),
    advantage=('advantage', 'median'), valid_repeats=('repeat_id', 'nunique'))
summary = []
for (rule, metric), g in station.groupby(['rule', 'metric'], sort=True):
    ci = interval(g.advantage, iterations=2000, seed=922)
    summary.append(dict(rule=rule, metric=metric, stations=len(g),
        hydraulic_median=float(g.hydraulic.median()),
        empirical_median=float(g.empirical.median()),
        paired_advantage=ci['median'], ci_low=ci['ci_low'], ci_high=ci['ci_high'],
        hydraulic_better_fraction=float(g.advantage.gt(0).mean())))
summary = pd.DataFrame(summary)
saved_comparison = pd.read_csv(root / 'postprocess_results/tables/empirical_scale_comparison.csv')
saved_comparison = saved_comparison[saved_comparison.group_variable.eq('all')]
for row in summary[summary.rule.isin(['baseline', 'error10', 'slope10'])].itertuples():
    reference = saved_comparison[(saved_comparison.rule.eq(row.rule)) &
                                 (saved_comparison.metric.eq(row.metric))].iloc[0]
    for fresh, saved_col in [('hydraulic_median', 'full_median'),
                             ('empirical_median', 'empirical_median'),
                             ('paired_advantage', 'median'),
                             ('ci_low', 'ci_low'), ('ci_high', 'ci_high'),
                             ('hydraulic_better_fraction', 'full_better_fraction')]:
        assert np.isclose(getattr(row, fresh), reference[saved_col], rtol=1e-9, atol=1e-9)
    assert row.stations == reference.stations
checks = {'paired_prediction_rows': len(d), 'unique_paired_observations': len(collapsed),
          'primary_cohort_observations': len(cohort),
          'primary_cohort_stations': int(cohort.station_id.nunique()),
          'all_predictions_paired': True, 'observed_Q_equal_between_models': True,
          'hydraulic_pooled_metrics_reproduced': True,
          'existing_paired_comparisons_reproduced': True,
          'saved_comparison_columns': saved_comparison.columns.tolist()}
summary.to_csv(out / 'paired_station_summary.csv', index=False, encoding='utf-8-sig')

primary_obs = obs[obs.primary_cohort]
influence = []
for rule in rules[1:]:
    removed = primary_obs[primary_obs[rule]]
    influence.append(dict(rule=rule, removed_observations=len(removed),
        removed_fraction=len(removed)/len(primary_obs),
        removed_squared_error_fraction=float(removed.squared_error.sum()/primary_obs.squared_error.sum())))
influence = pd.DataFrame(influence)
influence.to_csv(out / 'error_influence.csv', index=False, encoding='utf-8-sig')

result = {'executed_at': datetime.now(timezone.utc).isoformat(),
          'checks': checks, 'pooled': pooled.to_dict(orient='records'),
          'station_paired': summary.to_dict(orient='records'),
          'influence': influence.to_dict(orient='records')}
(out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False))
print(pooled[['rule','model','n','NSE','MAE','RMSE','MdAPE']].to_string(index=False))
print(summary[summary.metric.isin(['NSE','MdAPE','KGE'])].to_string(index=False))
print(influence.to_string(index=False))
