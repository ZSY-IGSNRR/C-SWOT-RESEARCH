"""Equal-gauge analysis of the saved fivefold OOF validation predictions."""
from pathlib import Path
import hashlib,json,sys
from _paths import work_root
ROOT=work_root()
import numpy as np
import pandas as pd

T=ROOT/'tables';F=ROOT/'figures';C=ROOT/'checks'
def dist(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return {'N':0,'median':np.nan,'q25':np.nan,'q75':np.nan,'half_IQR':np.nan}
    q=np.quantile(x,[.25,.5,.75])
    return dict(N=len(x),median=q[1],q25=q[0],q75=q[2],half_IQR=(q[2]-q[0])/2)
def read(name):return pd.read_csv(T/(name+'.csv'),dtype={'station_id':str})
def format_number(x,pos=None):
    if abs(x)>=1000:
        k=int(np.round(np.log10(abs(x))))
        return rf'${"-" if x<0 else ""}10^{{{k}}}$'
    return f'{x:g}'

def main():
    source=T/'station_unique_observation_metrics.csv'
    raw=read('station_unique_observation_metrics')
    d=raw.loc[raw.scheme.eq('wse_stratified_5fold')&raw.support.eq('native')].copy()
    assert not d.duplicated(['method','station_id','domain']).any()
    d['observed_sd']=np.sqrt(d.SST/d.nobs)
    d['normalized_RMSE_by_observed_sd']=d.RMSE/d.observed_sd.where(d.observed_sd.gt(0))
    check=d.NSE.notna()
    assert np.allclose(d.loc[check,'NSE'],1-d.loc[check,'normalized_RMSE_by_observed_sd']**2,rtol=1e-10,atol=1e-10)
    rows=[]
    for (domain,method),g in d.groupby(['domain','method'],sort=False):
        valid=g.loc[g.NSE.notna()]
        row=dict(domain=domain,method=method,stations=len(g),NSE_evaluable_stations=len(valid),
            positive_NSE_stations=int(valid.NSE.gt(0).sum()),
            positive_NSE_fraction=float(valid.NSE.gt(0).mean()) if len(valid) else np.nan,
            NSE_ge_0_5_stations=int(valid.NSE.ge(.5).sum()),
            NSE_below_minus_1_stations=int(valid.NSE.lt(-1).sum()))
        for metric,values in [('NSE',valid.NSE),('RMSE_same_NSE_gauges',valid.RMSE),
            ('RMSE_all_gauges',g.RMSE),('nobs',g.nobs),('normalized_RMSE',valid.normalized_RMSE_by_observed_sd)]:
            row.update({metric+'_'+k:v for k,v in dist(values).items()})
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(T/'station_analysis_summary.csv',index=False,encoding='utf-8-sig')
    d.to_csv(T/'station_analysis_gauge_metrics.csv',index=False,encoding='utf-8-sig')
    # Pairwise comparisons on identical observation support and identical gauges.
    common=raw.loc[raw.scheme.eq('wse_stratified_5fold')&raw.support.eq('all_methods_common')&raw.domain.eq('all')]
    main_model=common.loc[common.method.eq('full')]
    gains=[]
    for method,g in common.groupby('method'):
        if method=='full':continue
        z=main_model.merge(g,on='station_id',suffixes=('_full','_other'),validate='one_to_one')
        z=z.loc[z.NSE_full.notna()&z.NSE_other.notna()]
        assert (z.nobs_full==z.nobs_other).all()
        row={'comparator':method,'common_evaluable_gauges':len(z),
            'fraction_full_NSE_better':float((z.NSE_full>z.NSE_other).mean()),
            'fraction_full_NSE_tied':float(np.isclose(z.NSE_full,z.NSE_other,rtol=0,atol=1e-10).mean()),
            'fraction_full_RMSE_lower':float((z.RMSE_full<z.RMSE_other).mean())}
        row.update({'paired_NSE_gain_'+k:v for k,v in dist(z.NSE_full-z.NSE_other).items()})
        gains.append(row)
    pd.DataFrame(gains).to_csv(T/'station_analysis_paired_gains.csv',index=False,encoding='utf-8-sig')
    audit={'status':'PASS','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'unit':'one gauge, all its mutually exclusive five-fold OOF predictions combined',
        'national_summary':'equal gauge weights; median +/- half IQR across gauge scores',
        'NSE_qualification':'>=5 observations, observed and predicted variation; constant-prediction sensitivity separately verified',
        'RMSE_main_cohort':'same NSE-evaluable gauges; all-gauge RMSE provided separately',
        'tail_NSE_warning':'Most gauges have only one above-range and one below-range OOF event; do not infer per-gauge tail NSE from a single event.',
        'identity_NSE_equals_one_minus_normalized_RMSE_squared':True,'models_refitted':False,'raw_Q_exported':False}
    (C/'station_analysis_method.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.loc[summary.method.eq('full'),['domain','stations','NSE_evaluable_stations','NSE_median','NSE_half_IQR','RMSE_same_NSE_gauges_median','RMSE_same_NSE_gauges_half_IQR','positive_NSE_fraction']].to_string(index=False))
    print(pd.DataFrame(gains).to_string(index=False))

if __name__=='__main__':main()
