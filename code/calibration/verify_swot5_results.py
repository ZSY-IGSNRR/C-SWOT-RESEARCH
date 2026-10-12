"""Verify the executed SWOT-5 protocol, assignments, coverage and saved curves."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(results):
    cfg=json.loads((results/'run_configuration.json').read_text(encoding='utf-8'))
    protocol=json.loads((results/'method_and_optimizer_definition.json').read_text(encoding='utf-8'))
    tables=results/'tables'
    pred=pd.read_csv(tables/'validation_predictions.csv',dtype={'station_id':str})
    assign=pd.read_csv(tables/'split_assignments.csv',dtype={'station_id':str})
    params=pd.read_csv(tables/'parameter_estimates.csv',dtype={'station_id':str})
    def optional_table(name):
        try:return pd.read_csv(tables/name)
        except pd.errors.EmptyDataError:return pd.DataFrame()
    paired=optional_table('ablation_paired_station_differences.csv')
    coverage=optional_table('ablation_common_support_coverage.csv')
    failures=optional_table('failed_experiments.csv')
    methods={'full',*protocol['ablation_methods']} if cfg['run_ablation'] else {'full'}
    data=pred[pred.validation_scheme=='wse_stratified']
    repeats=set(range(cfg['repeat_count']))
    checks={
        'ascii_output_path':str(results).isascii(),
        'no_unexpected_worker_failures':failures.empty or not failures['method'].isin(['station_worker','process_pool']).any(),
        'large_region_spatial_transfer_disabled':not cfg['run_spatial_cv'] and not list(tables.glob('spatial_validation*')),
        'all_methods_present':methods.issubset(set(data.method)),
        'all_wse_repeats_present_by_method':all(set(data.loc[data.method==m,'repeat_id'])==repeats for m in methods),
        'no_duplicate_predictions':not pred.duplicated(['station_id','method','validation_scheme','repeat_id','fold_id','obs_id']).any(),
        'screened_predictions_pass_global_qc':bool(pred.loc[pred.method!='no_hampel','global_qc_pass'].eq(True).all()),
        'training_and_test_disjoint':not assign.groupby(['station_id','validation_scheme','repeat_id','fold_id','obs_id']).role.nunique().gt(1).any(),
        'paired_comparisons_use_common_observations':paired.empty or paired.support.eq('exact_common_validation_observations').all(),
        'paired_counts_do_not_exceed_native_counts':coverage.empty or bool(((coverage.common_count<=coverage.full_native_count)&(coverage.common_count<=coverage.comparator_native_count)).all()),
    }
    bad_curves=0;curves=0
    for row in params.to_dict('records'):
        value=row.get('width_curve_parameters_json')
        if not isinstance(value,str):continue
        c=json.loads(value);x=np.asarray(c['curve_wse']);y=np.asarray(c['curve_width'])
        rules=json.loads(row['width_rules_json']);curves+=1
        ok=np.all(np.diff(x)>0) and np.all(y>0) and np.all(np.diff(y)>=-1e-8)
        if row.get('width_growth_guard')==True:
            bounds=np.minimum(rules['width_max_growth_m_per_m'],rules['width_max_relative_growth_per_m']*y[:-1])*np.diff(x)
            ok=ok and np.all(np.diff(y)<=bounds+1e-7)
        bad_curves+=not ok
    checks['saved_curves_satisfy_recorded_constraints']=curves>0 and bad_curves==0
    temporal=assign[assign.validation_scheme=='temporal_forward']
    temporal_ok=True
    for _,g in temporal.groupby(['station_id','fold_id']):
        train=g[g.role=='training'];test=g[g.role=='validation']
        if len(train) and len(test):
            temporal_ok &= pd.to_datetime(train.time_dt).max()<=pd.to_datetime(test.time_dt).min()
    checks['temporal_training_precedes_test']=bool(temporal_ok)
    fraction_ok=True
    for _,g in temporal.groupby('station_id'):
        n=g.obs_id.nunique();first=g[g.fold_id==0]
        expected=min(max(5,int(np.floor(cfg['temporal_train_fraction']*n))),n-1)
        fraction_ok &= int((first.role=='training').sum())==expected
    checks['temporal_initial_fraction_matches_configuration']=bool(fraction_ok)
    a=params[(params.method=='full')&(params.validation_scheme=='wse_stratified')]
    b=params[(params.method=='width_train_only')&(params.validation_scheme=='wse_stratified')]
    match=a.merge(b,on=['station_id','repeat_id'],suffixes=('_a','_b'))
    checks['training_only_ablation_has_identical_fitted_parameters']=bool(len(match)>0 and all(np.allclose(match[c+'_a'],match[c+'_b'],rtol=1e-11,atol=1e-11) for c in ['n','beta','WSE0','width0'])) if cfg['run_ablation'] else True
    checks['all_ablation_repeats_present']=set(paired.repeat_id)==set(range(cfg['ablation_repeat_count'])) if cfg['run_ablation'] else True
    report={'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},
            'wse_stations':int(data.station_id.nunique()),'methods':len(methods),'saved_curve_count':curves,
            'recorded_failed_or_unavailable_experiments':len(failures),
            'interpretation':'Global full-record QC is retrospective. Native support and paired common support are distinct.'}
    (results/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,required=True);a=p.parse_args()
    raise SystemExit(0 if verify(a.results_dir)['passed'] else 1)
