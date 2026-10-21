"""Gauge-level OOF scores and exploratory between-gauge associations.

No model is fitted. Observation-level Q never leaves private/. Derived gauge
statistics are retained for interpreting score heterogeneity. Spearman rho is
descriptive, with gauge bootstrap confidence intervals. No p value is computed.
"""
from pathlib import Path
import json
from _paths import work_root
ROOT = work_root()
import numpy as np
import pandas as pd

SEED = 20261019
N_BOOT = 1000


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dist(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if not len(x):
        return {'n': 0}
    q25, med, q75 = np.quantile(x, [.25, .5, .75])
    return dict(n=len(x), median=med, q25=q25, q75=q75, half_IQR=(q75-q25)/2,
                minimum=x.min(), maximum=x.max())


def boolean(x):
    return x.astype(str).str.lower().isin(['true', '1', '1.0'])


def ratio(a, b):
    return a / b if np.isfinite(b) and b > 0 else np.nan


def safe_spearman(x, y):
    if len(x) < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    a = pd.Series(x).rank(method='average').to_numpy()
    b = pd.Series(y).rank(method='average').to_numpy()
    return float(np.corrcoef(a,b)[0,1]), np.nan


def boot_ci(x, y, rng):
    if len(x) < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    ix = rng.integers(0,len(x),size=(N_BOOT,len(x)))
    a = pd.DataFrame(x[ix]).rank(axis=1,method='average').to_numpy(copy=True)
    b = pd.DataFrame(y[ix]).rank(axis=1,method='average').to_numpy(copy=True)
    a -= a.mean(axis=1,keepdims=True)
    b -= b.mean(axis=1,keepdims=True)
    denominator = np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))
    vals = np.divide(np.sum(a*b,axis=1),denominator,out=np.full(N_BOOT,np.nan),where=denominator>0)
    vals = vals[np.isfinite(vals)]
    return tuple(np.quantile(vals, [.025, .975])) if len(vals) else (np.nan, np.nan)


def main():
    pred = pd.read_csv(ROOT/'private/validation_predictions_RESTRICTED.csv',
                       dtype={'station_id': 'string', 'obs_id': 'string'}, low_memory=False)
    pred = pred.loc[pred.method.eq('full')].copy()
    assert len(pred) == 7615
    assert pred.station_id.nunique() == 422
    assert not pred.duplicated(['station_id', 'obs_id']).any()
    par = pd.read_csv(ROOT/'private/parameter_estimates_RESTRICTED.csv',
                      dtype={'station_id': 'string'}, low_memory=False)
    par = par.loc[par.method.eq('full')].copy()
    elig = pd.read_csv(ROOT/'checks/all_station_assignment_eligibility.csv',
                       dtype={'station_id': 'string'}).set_index('station_id')
    boundary = pd.read_csv(ROOT/'tables/parameter_boundary_by_gauge.csv',
                           dtype={'station_id': 'string'})
    boundary['boundary_hit_fraction'] = boundary.either_hit_folds / boundary.fitted_folds
    bwide = boundary.pivot(index='station_id', columns='parameter', values='boundary_hit_fraction')
    bwide.columns = [f'{c}_boundary_hit_fraction' for c in bwide.columns]
    rows = []
    for station, g in pred.groupby('station_id', sort=True):
        obs = g.Q_observed.to_numpy(float)
        calc = g.Q_calculated.to_numpy(float)
        valid = np.isfinite(obs) & np.isfinite(calc)
        assert valid.all()
        err = calc-obs
        std = obs.std(ddof=0)
        sst = float(np.dot(obs-obs.mean(), obs-obs.mean()))
        sse = float(np.dot(err, err))
        nse = 1-sse/sst if len(obs) >= 5 and np.ptp(obs) > 0 and sst > 0 else np.nan
        rmse = np.sqrt(np.mean(err**2))
        width = g.width_raw.to_numpy(float)
        wse = g.WSE.to_numpy(float)
        slope = g.slope_raw.to_numpy(float)
        correction = np.abs(g.width_used.to_numpy(float)-width)
        out = dict(station_id=station, n_oof=len(obs), nse_defined=np.isfinite(nse),
                   NSE=nse, RMSE=rmse, Q_mean=obs.mean(), Q_sd_population=std,
                   Q_sd_sample=obs.std(ddof=1), Q_CV=ratio(std, obs.mean()),
                   Q_range=np.ptp(obs), Q_unique=len(np.unique(obs)),
                   normalized_RMSE_Q_sd=ratio(rmse, std),
                   WSE_span_m=np.ptp(wse), WSE_sd_m=np.std(wse),
                   raw_width_median_m=np.median(width), raw_width_CV=ratio(np.std(width),np.mean(width)),
                   median_absolute_width_correction_m=np.median(correction),
                   median_absolute_width_correction_percent=np.median(100*correction/width),
                   width_correction_fraction=np.mean(correction > 1.e-8),
                   slope_median=np.median(slope), slope_CV=ratio(np.std(slope),np.mean(slope)),
                   above_range_fraction=g.domain.eq('above').mean(),
                   below_range_fraction=g.domain.eq('below').mean(),
                   outside_range_fraction=g.domain.isin(['above','below']).mean(),
                   median_training_WSE_span_m=g.calibration_wse_span.median(),
                   max_normalized_extrapolation_distance=g.normalized_extrapolation_distance.max())
        # Sorted raw widths are a diagnostic; they are never used to fit an
        # association with validation Q or to select/remove individual points.
        ordered = g.sort_values(['WSE', 'obs_id'])
        delta_w = np.diff(ordered.width_raw.to_numpy(float))
        delta_h = np.diff(ordered.WSE.to_numpy(float))
        nonzero_h = delta_h > 0
        out['raw_WSE_width_decrease_fraction'] = (np.mean(delta_w[nonzero_h] < 0)
                                                 if nonzero_h.any() else np.nan)
        fit = par.loc[par.station_id.eq(station)]
        assert len(fit) == 5
        for c in ['n','beta','depth0','width0_to_min_width_ratio']:
            out[f'median_{c}'] = fit[c].median()
        hitcols = ['n_at_lower_bound','n_at_upper_bound','beta_at_lower_bound','beta_at_upper_bound',
                   'depth_at_lower_bound','depth_at_upper_bound','width0_at_lower_bound','width0_at_upper_bound']
        hitmat = np.column_stack([boolean(fit[c]).to_numpy() for c in hitcols])
        out['any_parameter_boundary_hit_fraction'] = hitmat.any(axis=1).mean()
        e = elig.loc[station]
        out['QC_excluded_fraction'] = ratio(float(e.qc_fail_observations),float(e.complete_observations))
        for c in ['wse','width','log_slope']:
            out[f'Hampel_{c}_fraction'] = ratio(float(e[f'global_hampel_{c}_count']),float(e.complete_observations))
        if station in bwide.index:
            out.update(bwide.loc[station].to_dict())
        rows.append(out)
    gauges = pd.DataFrame(rows)
    assert len(gauges) == 422 and gauges.NSE.notna().sum() == 418
    # Q-derived gauge summaries can disclose exact Q at constant-flow gauges.
    # Keep this detailed driver table local with the restricted inputs.
    gauges.to_csv(ROOT/'private/station_analysis_drivers_RESTRICTED.csv',index=False,encoding='utf-8-sig')
    feature_groups = {
        'n_oof':'sample_size', 'Q_mean':'discharge_distribution', 'Q_sd_population':'discharge_distribution',
        'Q_CV':'discharge_distribution', 'WSE_span_m':'hydraulic_observation',
        'raw_width_CV':'hydraulic_observation', 'raw_width_median_m':'hydraulic_observation',
        'median_absolute_width_correction_percent':'width_correction',
        'raw_WSE_width_decrease_fraction':'width_correction',
        'QC_excluded_fraction':'quality', 'Hampel_wse_fraction':'quality',
        'Hampel_width_fraction':'quality', 'Hampel_log_slope_fraction':'quality',
        'slope_CV':'hydraulic_observation', 'above_range_fraction':'WSE_coverage',
        'below_range_fraction':'WSE_coverage', 'outside_range_fraction':'WSE_coverage',
        'max_normalized_extrapolation_distance':'WSE_coverage',
        'any_parameter_boundary_hit_fraction':'parameter_boundary',
        'n_boundary_hit_fraction':'parameter_boundary', 'beta_boundary_hit_fraction':'parameter_boundary',
        'depth0_boundary_hit_fraction':'parameter_boundary', 'width0_boundary_hit_fraction':'parameter_boundary',
    }
    feature_groups = {k:v for k,v in feature_groups.items() if k in gauges}
    correlation = []
    rng = np.random.default_rng(SEED)
    for minimum in [5,10,20]:
        subset = gauges.loc[gauges.n_oof.ge(minimum)]
        for metric in ['NSE','RMSE']:
            for feature, group in feature_groups.items():
                valid = subset[[metric,feature]].replace([np.inf,-np.inf],np.nan).dropna()
                x,y = valid[feature].to_numpy(float),valid[metric].to_numpy(float)
                rho,p = safe_spearman(x,y)
                low,high = boot_ci(x,y,rng)
                correlation.append(dict(minimum_oof_pairs=minimum,metric=metric,feature=feature,
                                        feature_group=group,n_gauges=len(valid),Spearman_rho=rho,
                                        p_value=p,bootstrap_95ci_low=low,bootstrap_95ci_high=high,
                                        mathematical_coupling=(metric=='NSE' and feature in ['Q_sd_population','Q_CV'])))
    corr = pd.DataFrame(correlation)
    corr['BH_q_value'] = np.nan
    corr.to_csv(ROOT/'tables/station_analysis_correlations.csv',index=False,encoding='utf-8-sig')
    matrix_columns = ['NSE','RMSE',*feature_groups]
    gauges[matrix_columns].corr(method='spearman', min_periods=8).to_csv(
        ROOT/'tables/station_analysis_Spearman_matrix.csv',encoding='utf-8-sig')
    sensitivity = []
    for minimum in [5,10,20]:
        subset = gauges.loc[gauges.n_oof.ge(minimum)]
        score = subset.NSE.dropna()
        sensitivity.append(dict(minimum_oof_pairs=minimum,n_gauges=len(subset),
                                n_defined_NSE=len(score),NSE=dist(score),RMSE=dist(subset.RMSE),
                                fraction_NSE_positive=float(score.gt(0).mean()),
                                fraction_NSE_at_least_half=float(score.ge(.5).mean())))
    bins = []
    groupings = {
        'n_oof': pd.cut(gauges.n_oof,[6,9,19,np.inf],labels=['7-9','10-19','20+']),
        'Q_CV': pd.qcut(gauges.Q_CV,4,duplicates='drop'),
        'median_absolute_width_correction_percent': pd.qcut(gauges.median_absolute_width_correction_percent,4,duplicates='drop'),
        'outside_range_fraction': pd.qcut(gauges.outside_range_fraction,4,duplicates='drop'),
    }
    for feature, groups in groupings.items():
        for label, subset in gauges.groupby(groups,observed=True):
            out = dict(feature=feature,group=str(label),n_gauges=len(subset),
                       feature_median=subset[feature].median(),fraction_NSE_positive=subset.NSE.dropna().gt(0).mean())
            for metric in ['NSE','RMSE']:
                out.update({f'{metric}_{key}':value for key,value in dist(subset[metric]).items()})
            bins.append(out)
    pd.DataFrame(bins).to_csv(ROOT/'tables/station_analysis_groups.csv',index=False,encoding='utf-8-sig')
    identity = gauges.dropna(subset=['NSE','normalized_RMSE_Q_sd'])
    summary = dict(
        definition='One OOF prediction per observation; all five held-out folds combined within each gauge.',
        n_gauges=422,n_observations=7615,n_defined_NSE=418,
        undefined_NSE_reason='Four gauges have constant observed discharge (SST = 0). Their RMSE is retained.',
        NSE=dist(gauges.NSE),RMSE=dist(gauges.RMSE),sample_size=dist(gauges.n_oof),
        sensitivity=sensitivity,
        NSE_RMSE_identity='NSE = 1 - (RMSE / population_standard_deviation_of_Q)^2',
        NSE_RMSE_identity_max_abs_error=np.max(np.abs(identity.NSE-(1-identity.normalized_RMSE_Q_sd**2))),
        statistical_notes=[
            'Spearman correlations are exploratory associations across gauges, not causal evidence.',
            'NSE is mathematically coupled to Q variance and RMSE; a Q-variability association partly reflects its definition.',
            'RMSE is sensitive to discharge scale; compare it together with Q distribution and NSE.',
            'No p values or multiplicity-adjusted significance tests are computed; associated columns are intentionally empty.',
            'Bootstrap confidence intervals resample gauges, with 1000 resamples and seed 20261019.',
            'Thresholds >=5, >=10 and >=20 select different gauge populations; this is not an experiment changing sample count at fixed gauges.',
            'Uncertainty in scores from few observations and spatial dependence can affect these associations.',
            'The five fitted parameter boundary fractions are model diagnostics, not proof of a physical cause.',
        ],
        strongest_NSE_associations=corr.loc[corr.minimum_oof_pairs.eq(5)&corr.metric.eq('NSE')]
            .assign(abs_rho=lambda x:x.Spearman_rho.abs()).sort_values('abs_rho',ascending=False)
            .drop(columns='abs_rho').head(10).to_dict('records'),
    )
    (ROOT/'tables/station_analysis_drivers_summary.json').write_text(
        json.dumps(clean(summary),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(clean({k:summary[k] for k in ['n_gauges','n_observations','n_defined_NSE','NSE','RMSE','sensitivity','strongest_NSE_associations']}),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
