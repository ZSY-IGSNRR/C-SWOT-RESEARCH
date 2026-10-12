"""Post-process completed SWOT-5 predictions. Never fit the discharge model.

Use code/run.py postprocess, or inspect python analyze_postprocess.py --help.
"""
import argparse
from pathlib import Path
import json
import sys
import traceback
import warnings
import numpy as np
import pandas as pd
from common import (SCHEMES, OBS, UNIT, METRICS, RULES, PRIMARY, GROUPS, read_csv,
                    save, sha256, normalize, truth, metrics, interval,
                    aggregate_units, assign_groups, keep_rule, grouped_summary)

HERE = Path(__file__).resolve().parent
SOS_ROOT = Path('inputs/comparison_with_sos_spatial_audit_20260908')

def log(msg):
    print(msg, flush=True)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    p.add_argument('--output-dir', type=Path, default=Path('postprocess_results'))
    p.add_argument('--metadata', type=Path, default=HERE/'station_metadata.csv')
    p.add_argument('--station-audit', type=Path, default=HERE/'station_spatial_audit.csv')
    p.add_argument('--sos-raw-dir', type=Path, default=SOS_ROOT/'sos_v16_mapped_raw')
    p.add_argument('--boundary-file', type=Path, default=Path('inputs/Class2_hydrounits.shp'))
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--min-pairs', type=int, default=5)
    p.add_argument('--sos-min-pairs', type=int, default=4)
    p.add_argument('--max-stations', type=int, default=0, help='Software smoke test only; 0 uses all stations')
    p.add_argument('--skip-plots', action='store_true')
    return p.parse_args()

def load_inputs(args, out):
    path = args.results_dir/'tables/validation_predictions.csv'
    cols = ['station_id', 'obs_id', 'time', 'method', 'validation_scheme', 'repeat_id', 'fold_id',
            'Q_observed', 'Q_calculated', 'slope_used', 'WSE', 'width_raw', 'width_used']
    chunks = []
    for c in pd.read_csv(path, usecols=cols, chunksize=100000, dtype={'station_id':'string'}):
        c['station_id']=normalize(c.station_id)
        chunks.append(c[c.method.isin(['full', 'empirical_rating_curve']) & c.validation_scheme.isin(SCHEMES)])
    pred = pd.concat(chunks, ignore_index=True)
    if args.max_stations:
        # Spread a software-test subset over the sorted input station inventory.
        ids = sorted(pred.loc[pred.method.eq('full'), 'station_id'].unique())
        ix = np.unique(np.linspace(0, len(ids)-1, min(args.max_stations, len(ids))).astype(int))
        pred = pred[pred.station_id.isin([ids[i] for i in ix])].copy()
    assert len(pred), 'No held-out predictions found'
    assert not pred.duplicated(OBS+['method', 'repeat_id']).any(), 'Duplicate prediction keys'
    assert (np.isfinite(pred.Q_observed) & np.isfinite(pred.Q_calculated) & pred.Q_observed.gt(0)).all()
    pred['time'] = pd.to_datetime(pred.time, utc=True, errors='raise')
    assert pred.groupby(OBS).Q_observed.nunique().eq(1).all(), 'Qobs changes across methods/repeats'
    assert pred.groupby(OBS).time.nunique().eq(1).all(), 'Observation key has multiple times'
    meta = assign_groups(read_csv(args.metadata))
    assert meta.station_id.is_unique
    assert set(pred.station_id) <= set(meta.station_id), 'Station metadata missing'
    # Do not import old outcome metrics or old evaluation eligibility.
    params = read_csv(args.results_dir/'tables/parameter_estimates.csv', usecols=[
        'station_id', 'method', 'validation_scheme', 'repeat_id', 'fold_id',
        'train_slope_fallback', 'slope_positive_count', 'slope_used_global_fallback'])
    params = params[params.method.eq('full') & params.validation_scheme.isin(SCHEMES)]
    assert not params.duplicated(UNIT+['fold_id']).any()
    save(meta, out, 'station_metadata')
    return pred, meta, params

def make_flags(pred, params, out, min_pairs):
    full = pred[pred.method.eq('full')].copy()
    full = full.merge(params.drop(columns='method'), on=UNIT+['fold_id'], how='left',
                      validate='many_to_one', indicator=True)
    assert full._merge.eq('both').all(), 'Missing fold-specific slope reference'
    full = full.drop(columns='_merge')
    full['slope_reference_valid'] = (full.slope_positive_count.ge(5) &
        ~truth(full.slope_used_global_fallback) & full.train_slope_fallback.gt(0) &
        np.isfinite(full.train_slope_fallback) & full.slope_used.gt(0))
    ratio = full.slope_used / full.train_slope_fallback
    full['slope_factor'] = np.maximum(ratio, 1/ratio).where(full.slope_reference_valid)
    for threshold in (5, 10, 20):
        full[f'fold_slope{threshold}'] = full.slope_factor.gt(threshold)
    save(full[OBS+['repeat_id', 'fold_id', 'slope_used', 'train_slope_fallback',
                   'slope_positive_count', 'slope_reference_valid', 'slope_factor',
                   'fold_slope5', 'fold_slope10', 'fold_slope20']], out, 'slope_reference_audit')
    aggs = dict(time=('time', 'first'), Q_observed=('Q_observed', 'first'),
                Q_model=('Q_calculated', 'median'), prediction_appearances=('repeat_id', 'size'),
                reference_valid_fraction=('slope_reference_valid', 'mean'),
                WSE=('WSE', 'median'), width_raw=('width_raw', 'median'), width_used=('width_used', 'median'))
    for threshold in (5, 10, 20):
        aggs[f'slope{threshold}_fraction'] = (f'fold_slope{threshold}', 'mean')
    obs = full.groupby(OBS, as_index=False).agg(**aggs)
    # Nonpositive model Q is retained in evaluation and counted as an extreme underprediction.
    ratio = obs.Q_model / obs.Q_observed
    obs['factor_error'] = np.where(ratio.gt(0), np.maximum(ratio, 1/ratio), np.inf)
    for threshold in (5, 10, 20):
        obs[f'slope{threshold}'] = obs[f'slope{threshold}_fraction'].ge(.5)
        obs[f'error{threshold}'] = obs.factor_error.gt(threshold)
    obs['over10'] = ratio.gt(10)
    obs['under10'] = ratio.lt(.1)
    obs['absolute_error'] = (obs.Q_model-obs.Q_observed).abs()
    obs['squared_error'] = (obs.Q_model-obs.Q_observed)**2
    baseline = aggregate_units(full, min_pairs=min_pairs)
    eligible = baseline[baseline.n.ge(min_pairs)][['station_id', 'validation_scheme']].drop_duplicates()
    obs = obs.merge(eligible.assign(primary_cohort=True), on=['station_id', 'validation_scheme'],
                    how='left', validate='many_to_one')
    obs['primary_cohort'] = obs.primary_cohort.eq(True)
    obs['absolute_top1'] = False
    thresholds = []
    for scheme in SCHEMES:
        mask = obs.validation_scheme.eq(scheme) & obs.primary_cohort
        cutoff = obs.loc[mask, 'absolute_error'].quantile(.99)
        obs.loc[obs.validation_scheme.eq(scheme), 'absolute_top1'] = obs.absolute_error.gt(cutoff)
        thresholds.append(dict(validation_scheme=scheme, error_99th_m3s=cutoff,
                               reference_observations=int(mask.sum()), strict_greater=True))
    save(pd.DataFrame(thresholds), out, 'absolute_error_thresholds')
    save(baseline, out, 'baseline_all_station_repeat_metrics')
    full = full.merge(obs[OBS+RULES[1:]+['primary_cohort']], on=OBS, validate='many_to_one')
    return full, obs, baseline

def screening_analysis(full, obs, meta, out, args):
    p = full[full.primary_cohort].copy()
    o = obs[obs.primary_cohort].copy()
    units = []
    for rule in RULES:
        log(f'  Screening metrics: {rule}')
        r = aggregate_units(keep_rule(p, rule), min_pairs=args.min_pairs)
        r['rule'] = rule
        units.append(r)
    rm = pd.concat(units, ignore_index=True)
    save(rm, out, 'station_repeat_metrics_before_after')
    retention, pooled = [], []
    for sc, original in o.groupby('validation_scheme'):
        before = rm[rm.validation_scheme.eq(sc) & rm.rule.eq('baseline')]
        for rule in RULES:
            kept = keep_rule(original, rule)
            valid = rm[rm.validation_scheme.eq(sc) & rm.rule.eq(rule) & rm.n.ge(args.min_pairs)]
            retention.append(dict(validation_scheme=sc, rule=rule,
                observations_before=len(original), observations_kept=len(kept),
                observation_retention=len(kept)/len(original),
                stations_before=original.station_id.nunique(), stations_with_any=kept.station_id.nunique(),
                stations_evaluable=valid.station_id.nunique(),
                station_retention=valid.station_id.nunique()/original.station_id.nunique(),
                prediction_repeat_units_before=len(before),
                repeat_units_before=int(before.n.ge(args.min_pairs).sum()), repeat_units_kept=len(valid)))
            pooled.append(dict(validation_scheme=sc, rule=rule, aggregation='one_median_prediction_per_overpass',
                               **metrics(kept.Q_observed, kept.Q_model, min_pairs=1)))
    save(pd.DataFrame(retention), out, 'retention')
    save(pd.DataFrame(pooled), out, 'pooled_screening_metrics')
    # Native outcomes: which evaluable stations remain under each separate rule?
    native = rm[rm.n.ge(args.min_pairs)].groupby(['station_id', 'validation_scheme', 'rule'])[METRICS].median().reset_index()
    native = native.melt(id_vars=['station_id', 'validation_scheme', 'rule'], value_vars=METRICS,
                         var_name='metric', value_name='value').rename(columns={'rule': 'variant'})
    native['comparison'] = 'native_available'
    # Primary comparison holds station-repeat units fixed across all three variants.
    comparisons = [('primary_three', PRIMARY)] + [(f'baseline_vs_{r}', ['baseline', r]) for r in RULES[1:]]
    rows = []
    for name, variants in comparisons:
        for metric in METRICS:
            pv = rm[rm.rule.isin(variants)].pivot(index=UNIT, columns='rule', values=metric).reindex(columns=variants).dropna()
            if pv.empty:
                continue
            for key, g in pv.groupby(level=[0, 1]):
                for rule in variants:
                    rows.append(dict(station_id=key[0], validation_scheme=key[1], comparison=name,
                        variant=rule, metric=metric, value=g[rule].median(), common_repeats=len(g),
                        delta_from_baseline=(g[rule]-g.baseline).median()))
    common = pd.DataFrame(rows, columns=['station_id', 'validation_scheme', 'comparison', 'variant',
                                        'metric', 'value', 'common_repeats', 'delta_from_baseline'])
    common = common.merge(meta, on='station_id', validate='many_to_one')
    native = native.merge(meta, on='station_id', validate='many_to_one')
    save(common, out, 'common_station_metrics')
    save(native, out, 'available_station_metrics')
    summaries = pd.concat([grouped_summary(common, args.bootstrap), grouped_summary(native, args.bootstrap)], ignore_index=True)
    save(summaries, out, 'scale_spatial_performance_summary')
    # Coverage by scale uses all baseline-evaluable observations as the denominator.
    coverage = []
    for sc, d in o.groupby('validation_scheme'):
        for variable in ('flow_group', 'width_group', 'region'):
            for label, g in d.groupby(variable, dropna=False):
                for rule in RULES:
                    kept = keep_rule(g, rule)
                    ids = set(rm.loc[rm.validation_scheme.eq(sc) & rm.rule.eq(rule) & rm.n.ge(args.min_pairs), 'station_id'])
                    coverage.append(dict(validation_scheme=sc, group_variable=variable, group=str(label), rule=rule,
                        observations_before=len(g), observations_kept=len(kept), retention=len(kept)/len(g),
                        stations_before=g.station_id.nunique(), stations_evaluable=len(set(g.station_id)&ids)))
    save(pd.DataFrame(coverage), out, 'retention_by_scale_region')
    return common

def occurrence_analysis(obs, out, args):
    flags = ['error5', 'error10', 'error20', 'slope5', 'slope10', 'slope20', 'over10', 'under10', 'absolute_top1']
    rates, station_rates = [], []
    rng = np.random.default_rng(922)
    for population in ('all_heldout', 'baseline_evaluable'):
        frame = obs if population == 'all_heldout' else obs[obs.primary_cohort]
        for sc, d in frame.groupby('validation_scheme'):
            for sid, g in d.groupby('station_id'):
                station_rates.append(dict(station_id=sid, validation_scheme=sc, population=population,
                    observations=len(g), squared_error=g.squared_error.sum(),
                    **{flag+'_rate':g[flag].mean() for flag in flags}))
            for variable in ('all', 'flow_group', 'width_group', 'event_flow_group', 'region'):
                groups = [('All', d)] if variable == 'all' else d.groupby(variable, dropna=False)
                for label, g in groups:
                    for flag in flags:
                        z = g.groupby('station_id')[flag].agg(['sum', 'size'])
                        rate = z['sum'].sum()/z['size'].sum()
                        lo, hi = np.nan, np.nan
                        if len(z) >= 5 and args.bootstrap:
                            v = z.to_numpy()
                            ix = rng.integers(0, len(z), (args.bootstrap, len(z)))
                            draws = v[ix, 0].sum(axis=1)/v[ix, 1].sum(axis=1)
                            lo, hi = np.quantile(draws, [.025, .975])
                        rates.append(dict(validation_scheme=sc, population=population, group_variable=variable,
                            group=str(label), flag=flag, observations=len(g), stations=len(z),
                            flagged=int(g[flag].sum()), rate=rate, ci_low=lo, ci_high=hi,
                            equal_station_rate=(z['sum']/z['size']).mean()))
    save(pd.DataFrame(rates), out, 'occurrence_rates')
    save(pd.DataFrame(station_rates), out, 'station_occurrence')
    save(obs.sort_values('squared_error', ascending=False).groupby('validation_scheme').head(30), out, 'largest_errors')

def association_analysis(obs, out):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    estimates, support = [], []
    formula = 'outcome ~ log_width + log_Q50 + log_relative_Q + log_distance + C(region)'
    for sc, raw in obs[obs.primary_cohort].groupby('validation_scheme'):
        d = raw[raw.gauge_days.ge(270) & raw.gauge_Q50.gt(0) & raw.sword_width_m.gt(0) & raw.nearest_node_distance_km.ge(0)].copy()
        d = d.dropna(subset=['longitude', 'latitude', 'region'])
        d['log_width'] = np.log10(d.sword_width_m)
        d['log_Q50'] = np.log10(d.gauge_Q50)
        d['log_relative_Q'] = np.log10(d.Q_observed/d.gauge_Q50)
        d['log_distance'] = np.log10(1+d.nearest_node_distance_km)
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_width', 'log_Q50', 'log_relative_Q', 'log_distance'])
        d['spatial_block'] = (np.floor(d.longitude/5).astype(int).astype(str)+'_'+np.floor(d.latitude/5).astype(int).astype(str))
        d['outcome'] = d.error10.astype(int)
        for cluster in ('station_id', 'spatial_block'):
            info = dict(validation_scheme=sc, cluster=cluster, stations=d.station_id.nunique(),
                        observations=len(d), events=int(d.outcome.sum()), clusters=d[cluster].nunique(), formula=formula)
            if d.outcome.nunique() < 2 or d[cluster].nunique() < 5 or len(d) < 30:
                support.append(dict(info, status='insufficient_support'))
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model = smf.glm(formula, data=d, family=sm.families.Binomial())
                    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
                        support.append(dict(info, status='rank_deficient'))
                        continue
                    coefficient_count=model.exog.shape[1]
                    info['coefficient_count']=coefficient_count
                    if min(d.outcome.sum(),len(d)-d.outcome.sum()) < 5*coefficient_count or d[cluster].nunique() <= coefficient_count:
                        support.append(dict(info,status='insufficient_events_or_clusters'))
                        continue
                    if d.groupby('region').outcome.nunique().lt(2).any():
                        support.append(dict(info,status='separation_in_region'))
                        continue
                    fit = model.fit(cov_type='cluster', cov_kwds={'groups':d[cluster]})
                notes = '; '.join(str(w.message) for w in caught)
                if not fit.converged or any('separation' in str(w.message).lower() for w in caught):
                    support.append(dict(info, status='unstable_fit', warnings=notes))
                    continue
                ci = fit.conf_int()
                information=-model.hessian(fit.params)
                if (not np.isfinite(ci.to_numpy()).all() or not np.isfinite(fit.bse).all()
                        or (fit.bse <= 0).any() or np.linalg.cond(information)>1e12):
                    support.append(dict(info,status='unstable_covariance_or_information',warnings=notes))
                    continue
                for term in fit.params.index:
                    estimates.append(dict(validation_scheme=sc, cluster=cluster, term=term,
                        log_odds=fit.params[term], odds_ratio=np.exp(fit.params[term]),
                        ci_low=np.exp(ci.loc[term,0]), ci_high=np.exp(ci.loc[term,1]), p_value=fit.pvalues[term]))
                support.append(dict(info, status='ok', warnings=notes))
            except (ValueError, np.linalg.LinAlgError) as exc:
                support.append(dict(info, status='not_estimable', reason=str(exc)))
    save(pd.DataFrame(estimates, columns=['validation_scheme','cluster','term','log_odds','odds_ratio','ci_low','ci_high','p_value']), out, 'adjusted_associations')
    save(pd.DataFrame(support), out, 'association_model_support')

def empirical_comparison(pred, obs, meta, out, args):
    f = pred[pred.method.eq('full') & pred.validation_scheme.eq('wse_stratified')]
    e = pred[pred.method.eq('empirical_rating_curve') & pred.validation_scheme.eq('wse_stratified')]
    key = OBS+['repeat_id']
    d = f.merge(e[key+['Q_observed','Q_calculated']], on=key, suffixes=('', '_empirical'), validate='one_to_one')
    assert np.allclose(d.Q_observed, d.Q_observed_empirical)
    d = d.merge(obs[OBS+['slope10','error10']], on=OBS, validate='many_to_one')
    coverage, records = [], []
    for repeat, g in f.groupby('repeat_id'):
        coverage.append(dict(repeat_id=repeat, full_rows=len(g), empirical_rows=int(e.repeat_id.eq(repeat).sum()),
                             common_rows=int(d.repeat_id.eq(repeat).sum())))
    for rule in PRIMARY:
        for key, g in keep_rule(d, rule).groupby(UNIT):
            a = metrics(g.Q_observed, g.Q_calculated, args.min_pairs)
            b = metrics(g.Q_observed, g.Q_calculated_empirical, args.min_pairs)
            for metric in METRICS:
                sign = -1 if metric in ['MAE','RMSE','MdAPE'] else 1
                records.append(dict(zip(UNIT,key), rule=rule, metric=metric, n=a['n'],
                    full=a[metric], empirical=b[metric], full_advantage=sign*(a[metric]-b[metric])))
    rp = pd.DataFrame(records)
    save(rp, out, 'empirical_paired_repeat_metrics')
    save(pd.DataFrame(coverage), out, 'empirical_pair_coverage')
    st = rp.dropna(subset=['full','empirical']).groupby(['station_id','validation_scheme','rule','metric']).agg(
        full=('full','median'), empirical=('empirical','median'), full_advantage=('full_advantage','median'),
        valid_repeats=('repeat_id','nunique')).reset_index().merge(meta, on='station_id', validate='many_to_one')
    save(st, out, 'empirical_paired_station_metrics')
    rows=[]
    for (sc, rule, metric), frame in st.groupby(['validation_scheme','rule','metric']):
        for variable in ('all','flow_group','width_group'):
            groups = [('All',frame)] if variable=='all' else frame.groupby(variable,dropna=False)
            for label,g in groups:
                rows.append(dict(validation_scheme=sc,rule=rule,metric=metric,group_variable=variable,group=str(label),
                    full_better_fraction=g.full_advantage.gt(0).mean(), full_median=g.full.median(),
                    empirical_median=g.empirical.median(), **interval(g.full_advantage,args.bootstrap)))
    save(pd.DataFrame(rows),out,'empirical_scale_comparison')

def verify_baseline(rm, args):
    archived = read_csv(args.results_dir/'tables/validation_metrics_by_station.csv')
    archived = archived[archived.method.eq('full') & archived.validation_scheme.isin(SCHEMES)]
    j = rm.merge(archived[UNIT+['valid_count','R2','NSE','MAE']],on=UNIT,suffixes=('_new','_saved'),validate='one_to_one')
    assert len(j)==len(rm), 'Baseline metric units missing from original outputs'
    assert (j.n==j.valid_count).all()
    for name in ('R2','NSE','MAE'):
        mask=j.n.ge(args.min_pairs)
        assert np.allclose(j.loc[mask,name+'_new'],j.loc[mask,name+'_saved'],equal_nan=True,rtol=1e-7,atol=1e-8), name
    return len(j)

def main():
    args=parse_args()
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if not str(args.output_dir.resolve()).isascii():
        raise ValueError('Output directory must be ASCII only')
    if args.bootstrap<0 or args.min_pairs<2 or args.sos_min_pairs<2 or args.max_stations<0:
        raise ValueError('Invalid bootstrap/sample-size options')
    # Preflight every required source before creating result artifacts.
    required=[args.metadata,args.station_audit,args.boundary_file,args.results_dir/'run_configuration.json',
              *[args.results_dir/'tables'/f for f in ['validation_predictions.csv','parameter_estimates.csv','validation_metrics_by_station.csv']]]
    for p in required:
        if not p.is_file():raise FileNotFoundError(p)
    if not args.sos_raw_dir.is_dir():raise FileNotFoundError(args.sos_raw_dir)
    if args.output_dir.exists():raise FileExistsError(f'Choose a new output directory: {args.output_dir}')
    # Import plotting dependencies now so a missing dependency does not strand a long run.
    if not args.skip_plots:
        import matplotlib, geopandas
    import statsmodels.api
    args.output_dir.mkdir(parents=True)
    out=args.output_dir/'tables';out.mkdir()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(smoke_test=bool(args.max_stations),main_model_refitted=False,
        Q50_definition='daily medians, 2023-04-06 through 2024-03-07; at least 270 of 337 days',
        width_definition='fixed nearest-node SWORD v17b prior, not fitted or gauge cross-section width',
        slope_definition='current SWOT-5 saved calibration slope median, >=5 positive values, no global fallback; >=half repeat appearances flagged',
        selection_warning='Residual exclusions are post hoc diagnostics, not independent model improvement',
        sources={str(p):sha256(p) for p in required if p.suffix!='.shp'})
    (args.output_dir/'configuration.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        log('[1/7] Reading predictions and static station metadata')
        pred,meta,params=load_inputs(args,out)
        full,obs,baseline=make_flags(pred,params,out,args.min_pairs)
        checked=verify_baseline(baseline,args)
        obs=obs.merge(meta,on='station_id',validate='many_to_one')
        obs['event_flow_group']=pd.cut(obs.Q_observed,[0,10,100,1000,np.inf],labels=GROUPS['event_flow_group']).astype(object)
        save(obs,out,'observation_diagnostics')
        log('[2/7] Screening, retention, scale and spatial summaries')
        screening_analysis(full,obs,meta,out,args)
        log('[3/7] Error occurrence and conditional associations')
        occurrence_analysis(obs,out,args)
        association_analysis(obs,out)
        log('[4/7] Empirical WSE-Q comparison on identical held-out observations')
        empirical_comparison(pred,obs,meta,out,args)
        log('[5/7] Rebuilding audited SoS common support from cached raw files')
        from sos_analysis import analyze_sos
        sos_checks=analyze_sos(obs,args,out)
        log('[6/7] Rendering figures')
        figures=[]
        if not args.skip_plots:
            from plot_postprocess import plot_all
            figures=plot_all(args.output_dir,args.boundary_file)
        log('[7/7] Final source-integrity checks')
        for path,digest in config['sources'].items():
            assert sha256(path)==digest, f'Source changed during processing: {path}'
        verification=dict(passed=True,smoke_test=bool(args.max_stations),baseline_units_reconciled=checked,
            independent_observations=len(obs),main_model_refitted=False,figure_count=len(figures),
            original_sources_unchanged=True,**sos_checks)
        (args.output_dir/'verification_report.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
        log(json.dumps(verification,indent=2))
        log(f'Finished. Results: {args.output_dir}')
    except Exception:
        (args.output_dir/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':
    main()
