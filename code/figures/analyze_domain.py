from pathlib import Path
import json, hashlib, os
import numpy as np
import pandas as pd

ROOT=Path(os.environ['SWOT_RELEASE_OUTPUT']).resolve()
ROOT.mkdir(exist_ok=True)
D=pd.read_pickle(Path(os.environ['SWOT_RELEASE_BASE_CACHE']))
f=Path(os.environ['SWOT_RELEASE_DATA_ROOT'])/'results/tables/validation_predictions.csv'
cols=['station_id','obs_id','time','method','validation_scheme','repeat_id','fold_id','WSE','width_curve_wse_min','width_curve_wse_max','Q_observed','Q_calculated','wse_position_class','flow_regime']
parts=[]
for z in pd.read_csv(f,usecols=cols,dtype={'station_id':str},chunksize=80000,low_memory=False):
    parts.append(z[z.method.eq('full')].copy())
p=pd.concat(parts,ignore_index=True)
p['span']=p.width_curve_wse_max-p.width_curve_wse_min
p['side']=np.select([p.WSE<p.width_curve_wse_min,p.WSE>p.width_curve_wse_max],['Below','Above'],default='Inside')
p['distance_m']=np.maximum(np.maximum(p.width_curve_wse_min-p.WSE,p.WSE-p.width_curve_wse_max),0)
p['distance_norm']=np.where(p.span>0,p.distance_m/p.span,np.nan)
p['log_ratio']=np.log10(p.Q_calculated/p.Q_observed)
p['abs_log_ratio']=p.log_ratio.abs()
p['APE']=100*np.abs(p.Q_calculated-p.Q_observed)/p.Q_observed
assert ((p.side.eq('Inside')) == p.wse_position_class.eq('interpolation')).all()
assert p[['Q_observed','Q_calculated']].gt(0).all().all()

def metrics(z):
    o=z.Q_observed.to_numpy(); q=z.Q_calculated.to_numpy()
    denom=np.sum((o-o.mean())**2)
    return {'observations':len(z),'stations':z.station_id.nunique(),'NSE':1-np.sum((q-o)**2)/denom if denom>0 else np.nan,'R2':np.corrcoef(o,q)[0,1]**2 if o.std()>0 and q.std()>0 else np.nan,'MdAPE':z.APE.median(),'factor':10**z.abs_log_ratio.median(),'signed_log_ratio':z.log_ratio.median()}

rs=[]
for (sc,r,side),z in p[p.validation_scheme.ne('station_reconstruction')].groupby(['validation_scheme','repeat_id','side']):
    rs.append({'scheme':sc,'repeat':r,'side':side,**metrics(z)})
rs=pd.DataFrame(rs);rs.to_csv(ROOT/'domain_summary.csv',index=False)
inside=[]
for (sc,sta,r),z in p[p.side.eq('Inside')&p.validation_scheme.ne('station_reconstruction')].groupby(['validation_scheme','station_id','repeat_id']):
    if len(z)>=5 and z.Q_observed.std()>0 and z.Q_calculated.std()>0:
        inside.append({'validation_scheme':sc,'station_id':sta,'repeat_id':r,**metrics(z)})
ins=pd.DataFrame(inside)
inside_station=ins.groupby(['validation_scheme','station_id'],as_index=False)[['NSE','R2','MdAPE','factor']].median()
inside_station.to_csv(ROOT/'inside_station_metrics.csv',index=False)

# Distance-bin errors use station-balanced summaries, with repeated assignments
# reduced within station before the across-station median and bootstrap.
edges=[0,.25,.5,1,np.inf];labs=['0–0.25','0.25–0.5','0.5–1','>1']
out=p[p.side.ne('Inside')&p.validation_scheme.ne('station_reconstruction')&p.span.gt(0)].copy()
out['bin']=pd.cut(out.distance_norm,edges,labels=labs,include_lowest=False)
bst=[];details=[];rng=np.random.default_rng(20261007)
for (sc,side,b),z in out.groupby(['validation_scheme','side','bin'],observed=True):
    a=z.groupby(['station_id','repeat_id'])[['abs_log_ratio','log_ratio']].median().groupby('station_id').median()
    v=a.abs_log_ratio.to_numpy();n=len(v)
    if n>=5:
        boot=np.median(v[rng.integers(0,n,size=(2000,n))],axis=1);lo,hi=np.quantile(boot,[.025,.975])
    else:lo=hi=np.nan
    exposures=z.groupby('repeat_id').size()
    # Reindex all ten assignments, so zero-count assignments contribute to support.
    if sc=='wse_stratified':exposures=exposures.reindex(range(10),fill_value=0)
    row={'scheme':sc,'side':side,'bin':str(b),'stations':n,'unique_observations':z[['station_id','obs_id']].drop_duplicates().shape[0],'median_predictions_per_assignment':exposures.median(),'median_abs_log_ratio':np.median(v),'factor':10**np.median(v),'factor_ci_low':10**lo,'factor_ci_high':10**hi,'median_signed_log_ratio':a.log_ratio.median(),'median_distance_m':z.distance_m.median(),'median_distance_norm':z.distance_norm.median()}
    bst.append(row)
    for sta,rowsta in a.iterrows():details.append({'scheme':sc,'side':side,'bin':str(b),'station_id':sta,**rowsta.to_dict()})
bs=pd.DataFrame(bst);bs.to_csv(ROOT/'distance_bins.csv',index=False)
pd.DataFrame(details).to_csv(ROOT/'distance_station_summary.csv',index=False)
audit={'source':str(f),'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'full_prediction_rows':len(p),'distance_exclusions_zero_span':int((p.validation_scheme.ne('station_reconstruction')&p.side.ne('Inside')&p.span.le(0)).sum()),'domain_medians':rs.groupby(['scheme','side']).median(numeric_only=True).reset_index().to_dict('records'),'inside_station_summary':[]}
for sc,z in inside_station.groupby('validation_scheme'):
    audit['inside_station_summary'].append({'scheme':sc,'stations':len(z),'NSE':z.NSE.median(),'R2':z.R2.median(),'positive_NSE':int(z.NSE.gt(0).sum()),'positive_fraction':z.NSE.gt(0).mean()})
audit['distance_bins']=bs.to_dict('records')
audit['distance_units']='boundary distance divided by positive calibration WSE span; no refitting'
(ROOT/'domain_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
D.update(predictions_domain=p,domain_summary=rs,inside_station=inside_station,distance_bins=bs)
pd.to_pickle(D,ROOT/'data.pkl')
print(json.dumps(audit,ensure_ascii=True,indent=2))
