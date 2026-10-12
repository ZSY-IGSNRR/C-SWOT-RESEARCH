from pathlib import Path
import sys,json,os
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
root=Path(os.environ['SWOT_RELEASE_DATA_ROOT']);out=Path(os.environ['SWOT_RELEASE_OUTPUT'])
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'postprocessing'))
from common import OBS,UNIT,metrics,interval,normalize,truth
methods=['full','width_train_only','empirical_rating_curve']
cols=OBS+['repeat_id','method','Q_observed','Q_calculated','WSE','width_raw','width_used','slope_used']
chunks=[]
for p in pd.read_csv(root/'results/tables/validation_predictions.csv',usecols=cols,
                     dtype={'station_id':'string','obs_id':'string'},chunksize=100000):
    chunks.append(p[p.method.isin(methods)&p.validation_scheme.eq('wse_stratified')])
p=pd.concat(chunks,ignore_index=True);p.station_id=normalize(p.station_id)
obs=pd.read_csv(root/'postprocess_results/tables/observation_diagnostics.csv',
                usecols=OBS+['primary_cohort','squared_error'],dtype={'station_id':'string','obs_id':'string'})
obs.station_id=normalize(obs.station_id);obs.primary_cohort=truth(obs.primary_cohort)
key=OBS+['repeat_id']
full=p[p.method.eq('full')]
raw=p[p.method.eq('width_train_only')]
emp=p[p.method.eq('empirical_rating_curve')]
d=full.merge(raw[key+['Q_calculated','Q_observed','WSE','slope_used','width_used']],on=key,
             suffixes=('','_raw'),validate='one_to_one')
d=d.merge(emp[key+['Q_calculated']],on=key,suffixes=('','_emp'),validate='one_to_one')
assert len(d)==len(full)==len(raw)==len(emp)
assert np.allclose(d.WSE,d.WSE_raw) and np.allclose(d.Q_observed,d.Q_observed_raw)
assert np.allclose(d.slope_used,d.slope_used_raw)
d=d.merge(obs[OBS+['primary_cohort']],on=OBS,validate='many_to_one')
g=d.groupby(OBS,as_index=False).agg(Q_observed=('Q_observed','first'),
    full=('Q_calculated','median'),raw=('Q_calculated_raw','median'),
    empirical=('Q_calculated_emp','median'),primary_cohort=('primary_cohort','first'))
g=g[g.primary_cohort]
summary=[]
for col in ['raw','full','empirical']:
    ratio=g[col]/g.Q_observed
    factor=np.where(ratio>0,np.maximum(ratio,1/ratio),np.inf)
    summary.append({'model':col,'stations':int(g.station_id.nunique()),
                    'tenfold_errors':int((factor>10).sum()),'tenfold_rate':float(np.mean(factor>10)),
                    **metrics(g.Q_observed,g[col],min_pairs=1)})
pd.DataFrame(summary).to_csv(out/'width_mapping_summary.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(summary).to_string(index=False))
# Station-level paired occurrence differences give each eligible station one vote.
station=g[['station_id']].copy()
for col in ['raw','full','empirical']:
    ratio=g[col]/g.Q_observed
    station[col]=np.where(ratio>0,np.maximum(ratio,1/ratio),np.inf)>10
station=station.groupby('station_id')[['raw','full','empirical']].mean()
station_summary=[]
for a,b in [('raw','full'),('full','empirical')]:
    s=interval(100*(station[a]-station[b]),2000)
    station_summary.append({'comparison':a+' minus '+b,'raw_or_full_median_rate':float(100*station[a].median()),
                            'full_or_empirical_median_rate':float(100*station[b].median()),**s})
print('Station-balanced rate differences',station_summary)
param_path=root/'results/tables/parameter_estimates.csv'
header=pd.read_csv(param_path,nrows=0).columns.tolist()
print('Parameter columns',header)
chosen=UNIT+['fold_id','method','n','beta','WSE0','width0','width_curve_parameters_json','train_slope_fallback']
params=pd.read_csv(param_path,usecols=chosen,dtype={'station_id':'string'})
params.station_id=normalize(params.station_id)
params=params[params.validation_scheme.eq('wse_stratified') & params.method.isin(['full','width_train_only'])]
pf=params[params.method.eq('full')];pr=params[params.method.eq('width_train_only')]
paired=pf.merge(pr,on=UNIT+['fold_id'],suffixes=('','_raw'),validate='one_to_one')
checks={v:bool(np.allclose(paired[v],paired[v+'_raw'],rtol=1e-10,atol=1e-10)) for v in ['n','beta','WSE0','width0']}
checks['calibration_width_curve']=bool((paired.width_curve_parameters_json==paired.width_curve_parameters_json_raw).all())
checks['calibration_slope_reference']=bool(np.allclose(paired.train_slope_fallback,paired.train_slope_fallback_raw))
print('Identical calibration parameter checks',checks,'paired_fits',len(paired))
data={'pooled':summary,'station_rate_differences':station_summary,
      'same_WSE':True,'same_slope':True,'paired_prediction_rows':len(d),
      'parameter_equal':checks,'paired_fits':len(paired)}
(out/'width_analysis.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
assert all(checks.values())
