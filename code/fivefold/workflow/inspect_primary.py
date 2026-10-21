"""Independent primary-score calculation (NumPy only, no summary helpers)."""
from pathlib import Path
import json,sys
from _paths import work_root
ROOT=work_root()
import numpy as np
import pandas as pd
p=pd.read_csv(ROOT/'private/validation_predictions_RESTRICTED.csv',
    dtype={'station_id':str,'obs_id':str},usecols=['station_id','obs_id','method','fold_id','domain','Q_observed','Q_calculated'])
p=p.loc[p.method.eq('full')]
assert len(p)==7615 and not p.duplicated(['station_id','obs_id']).any()
result=[]
for domain in ['all','within','above','below']:
    sub=p if domain=='all' else p.loc[p.domain.eq(domain)]
    scores=[]
    for fold,g in sub.groupby('fold_id'):
        obs=g.Q_observed.to_numpy();sim=g.Q_calculated.to_numpy()
        sse=np.dot(sim-obs,sim-obs);sst=np.dot(obs-obs.mean(),obs-obs.mean())
        scores.append({'fold':int(fold),'n':len(g),'NSE':float(1-sse/sst),'RMSE':float(np.sqrt(sse/len(g)))})
    obs=sub.Q_observed.to_numpy();sim=sub.Q_calculated.to_numpy()
    sse=np.dot(sim-obs,sim-obs);sst=np.dot(obs-obs.mean(),obs-obs.mean())
    row={'domain':domain,'n':len(sub),'station_count':int(sub.station_id.nunique()),'fold_scores':scores,
        'OOF_NSE':float(1-sse/sst),'OOF_RMSE':float(np.sqrt(sse/len(sub)))}
    for metric in ['NSE','RMSE']:
        q=np.percentile([s[metric] for s in scores],[25,50,75])
        row[metric+'_median']=float(q[1]);row[metric+'_half_IQR']=float((q[2]-q[0])/2)
    result.append(row)
(ROOT/'checks/primary_independent_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in row.items() if k!='fold_scores'} for row in result],indent=2))
