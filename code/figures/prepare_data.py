from pathlib import Path
import json, hashlib, sys, argparse
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
P=argparse.ArgumentParser();P.add_argument('--data-root',type=Path,default=Path('.'));P.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);P.add_argument('--cache',type=Path,default=Path(__file__).parent/'cache');A=P.parse_args()
ROOT=A.data_root;WORK=A.cache;WORK.mkdir(parents=True,exist_ok=True);OUT=A.out
for part in ['main','supplement','diagnostics','editable','source','assets']:(OUT/part).mkdir(parents=True,exist_ok=True)
manifest=[]
def read(folder,name,usecols=None):
    f=ROOT/folder/'tables'/f'{name}.csv'
    d=pd.read_csv(f,usecols=usecols,dtype={'station_id':str},low_memory=False)
    manifest.append({'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'rows':len(d),'columns':list(d.columns)})
    return d
data={}
small={'results':['global_qc_station_summary','station_data_statistics','ablation_paired_test_summary','parameter_multistart_summary','n_sensitivity_summary','beta_sensitivity_summary','sample_size_learning_curve_summary','time_gap_metrics','flow_regime_metrics','peak_flow_metrics_by_station','wse_position_metrics','wse_repeat_uncertainty_by_station','experiment_overall_summary','validation_metrics_by_station'],
       'postprocess_results':['station_metadata','station_occurrence','sos_paired_summary','sos_paired_station_metrics','sos_retention','sos_pooled_metrics','empirical_scale_comparison','scale_spatial_performance_summary','association_model_support','adjusted_associations','retention','pooled_screening_metrics','common_station_metrics']}
for folder,names in small.items():
    for name in names:
        d=read(folder,name)
        if folder=='results' and 'method' in d:d=d[d.method.eq('full')].copy()
        data[name]=d
cols=['station_id','obs_id','time','method','validation_scheme','repeat_id','Q_observed','Q_calculated','wse_position_class','time_gap_days']
pred=read('results','validation_predictions',cols);pred=pred[pred.method.eq('full')].copy()
data['predictions_full']=pred
keys=['station_id','obs_id','time','Q_observed']
w=pred[pred.validation_scheme.eq('wse_stratified')].groupby(keys,dropna=False,as_index=False).agg(Q_calculated=('Q_calculated','median'),repeat_count=('repeat_id','nunique'))
w['validation_scheme']='wse_stratified'
data['predictions_collapsed']=pd.concat([w,pred[~pred.validation_scheme.eq('wse_stratified')]],ignore_index=True)
cols=['station_id','method','validation_scheme','n','beta','width0','depth0','A0','width0_to_min_width_ratio','objective_loss','n_at_lower_bound','n_at_upper_bound','beta_at_lower_bound','beta_at_upper_bound','depth_at_lower_bound','depth_at_upper_bound','width0_at_lower_bound','width0_at_upper_bound','width_growth_capped_anchor_count']
p=read('results','parameter_estimates',cols);p=p[p.method.eq('full')&p.validation_scheme.eq('station_reconstruction')].copy()
assert not p.station_id.duplicated().any();data['parameters']=p
cols=['station_id','method','validation_scheme','curve_wse','raw_width','initial_corrected_width','isotonic_width','curve_width','growth_capped']
curves=read('results','width_curve_anchor_audit',cols)
ids=p.sort_values(['width_growth_capped_anchor_count','station_id'],ascending=[False,True]).head(4).station_id.tolist()
data['width_curves']=curves[curves.method.eq('full')&curves.validation_scheme.eq('station_reconstruction')&curves.station_id.isin(ids)]
data['width_example_ids']=ids
qa=read('results','overpass_qc_audit',['station_id','time_dt','global_qc_pass'])
data['overpass_qc']=qa
data['station_metrics']=data['validation_metrics_by_station'].groupby(['validation_scheme','station_id'],as_index=False)[['NSE','R2','KGE','MAE']].median()
counts=data['predictions_collapsed'].groupby('validation_scheme').size().to_dict()
assert counts=={'station_reconstruction':8268,'temporal_forward':2338,'wse_stratified':7069},counts
assert len(p)==542
assert int(data['global_qc_station_summary'].complete_observations.sum())==10756
assert int(data['global_qc_station_summary'].retained_observations.sum())==8859
pd.to_pickle(data,WORK/'data.pkl')
(OUT/'source/source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prediction support',counts)
print('Prepared',len(data),'frames')
