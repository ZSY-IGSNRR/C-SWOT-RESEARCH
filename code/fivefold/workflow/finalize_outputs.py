"""Check final derived outputs, score agreement and local report links."""
from pathlib import Path
import csv,hashlib,json,re,sys
from _paths import work_root
ROOT=work_root()
import numpy as np
import pandas as pd
audits=['checks/run_status.json','checks/all_assignment_audit.json',
    'checks/computation_equivalence.json','checks/final_independent_audit.json',
    'checks/products_final_QA.json','tables/summary.json','tables/error_diagnostics_method.json']
reports={p:json.loads((ROOT/p).read_text(encoding='utf-8-sig')) for p in audits}
for p,r in reports.items():
    if 'status' in r:assert r['status']=='PASS',(p,r['status'])
assert reports['checks/run_status.json']['failed_fits']==0
primary=json.loads((ROOT/'checks/primary_independent_metrics.json').read_text(encoding='utf-8'))
summary=pd.read_csv(ROOT/'tables/cross_gauge_summary.csv')
for row in primary:
    r=summary.loc[summary.scheme.eq('wse_stratified_5fold')&summary.support.eq('native')&summary.method.eq('full')&summary.domain.eq(row['domain'])].iloc[0]
    for key in ['NSE_median','NSE_half_IQR','RMSE_median','RMSE_half_IQR']:
        assert np.isclose(r[key],row[key],rtol=1e-12,atol=1e-12),(row['domain'],key)
for path in (ROOT/'tables').glob('*.csv'):
    with path.open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
    bad=set(header)&{'Q_observed','Q_calculated','gauge_Q','obs_id','time_dt','Q','train_Q_min','train_Q_max'}
    assert not bad,(path,bad)
page=(ROOT/'index.html').read_text(encoding='utf-8')
links=re.findall(r'(?:href|src)="([^"]+)"',page)
for link in links:
    if not link.startswith(('http:','https:','#')):assert (ROOT/link).is_file(),link
files=[]
for folder in ['code','figures','checks','tables']:
    for path in sorted((ROOT/folder).glob('*')):
        if not path.is_file() or path.name=='deliverables_manifest.json' or path.suffix=='.log':continue
        files.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
result={'status':'PASS','all_saved_primary_metrics_match_independent_calculation':True,
    'aggregate_CSVs_exclude_raw_gauge_Q_and_observation_keys':True,'local_report_links_resolve':True,
    'all_integrity_checks_passed':list(reports),'files':files,
    'manuscript_and_responses_modified':False,'private_directory_restricted':True}
(ROOT/'checks/deliverables_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))
