"""Standard WSE-stratified 5-fold CV using the unchanged frozen V15 model.

Observation-level records and fitted model checkpoints contain restricted gauge
information and remain in private/.  No manuscript or original file is edited.
"""
from __future__ import annotations
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='1'
from pathlib import Path
import argparse, copy, hashlib, json, pickle, platform, shutil, sys, time, traceback
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from _paths import work_root
ROOT=work_root()
from _paths import CALIBRATION, load_configuration
sys.path.insert(0,str(CALIBRATION))
import numpy as np
import pandas as pd
import run_beta_wse0_width0_experiments as v15
core=v15.core
quality=v15.quality
SCHEME='wse_stratified_5fold'
METHODS=['full',*v15.ABLATION_METHODS,'observed_fixed','corrected_fixed']
EQUIVALENT_JOINT={'full','width_train_only','no_initial_width_guard','no_width_growth_guard','monotonic_only'}
v15.METHOD_LABELS.update(observed_fixed='Observed width; all four parameters fixed',
    corrected_fixed='Corrected width; all four parameters fixed')
CFG=SimpleNamespace(**load_configuration())
v15._install_overrides()
quality.configure(CFG)

# Cache deterministic preprocessing only. Each miss calls the unchanged V15
# function; copies protect cached outputs. The objective and optimizer are not
# replaced. This avoids reconstructing the same pandas anchor table at every
# optimizer evaluation, without changing a floating-point operation.
_original_collapse=core.collapse_wse_width
@lru_cache(maxsize=128)
def _cached_collapse(xbytes,ybytes):
    x=np.frombuffer(xbytes,dtype=np.float64);y=np.frombuffer(ybytes,dtype=np.float64)
    return _original_collapse(x,y)
def cached_collapse(wse,width):
    if isinstance(wse,pd.Series) and isinstance(width,pd.Series) and not wse.index.equals(width.index):
        return _original_collapse(wse,width)
    x=np.asarray(wse,dtype=np.float64);y=np.asarray(width,dtype=np.float64)
    a,b=_cached_collapse(x.tobytes(),y.tobytes())
    return a.copy(),b.copy()
if os.environ.get('V15_FIVEFOLD_PURE_CACHE','1')=='1':core.collapse_wse_width=cached_collapse

def safe(o):
    if isinstance(o,Path):return str(o)
    if isinstance(o,np.generic):return o.item()
    if isinstance(o,np.ndarray):return o.tolist()
    return str(o)

def dump_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=safe),encoding='utf-8')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def assign_group(frame,seed):
    """Rank strata, shuffle in strata, cyclic labels with a carried offset."""
    ordered=frame.sort_values(['wse_model','obs_id'],kind='stable')
    rng=np.random.default_rng(seed)
    labels=rng.permutation(5)
    offset=int(rng.integers(0,5))
    rows=[]
    for stratum,idx in enumerate(np.array_split(np.arange(len(ordered)),5)):
        idx=rng.permutation(idx)
        for j,k in enumerate(idx):
            record=ordered.iloc[int(k)]
            rows.append({'obs_id':record.obs_id,'fold_id':int(labels[(offset+j)%5]),
                'wse_stratum':stratum})
        offset=(offset+len(idx))%5
    return pd.DataFrame(rows,columns=['obs_id','fold_id','wse_stratum'])

def split_station(base,sid):
    assert not base.obs_id.duplicated().any(),sid
    passed=base.loc[base.global_qc_pass].copy()
    if len(passed)<7:return None
    records=[]
    for qc_status,frame in [('pass',passed),('fail',base.loc[~base.global_qc_pass])]:
        assigned=assign_group(frame,v15.pipeline.stable_seed(sid,SCHEME,'assignment',qc_status,base=CFG.seed))
        assigned['qc_group']=qc_status
        records.append(assigned)
    assignment=pd.concat(records,ignore_index=True)
    assert len(assignment)==len(base) and not assignment.obs_id.duplicated().any()
    counts=assignment.loc[assignment.qc_group.eq('pass')].fold_id.value_counts().reindex(range(5),fill_value=0)
    assert counts.min()>=1 and counts.max()-counts.min()<=1
    for _,a in assignment.loc[assignment.qc_group.eq('pass')].groupby('wse_stratum'):
        c=a.fold_id.value_counts().reindex(range(5),fill_value=0)
        assert c.max()-c.min()<=1
    return assignment

def fixed_fit(train,n_ref,corrected):
    prepared,_=core.prepare_training(train,True)
    if prepared is None:return None
    anchor,_=core.collapse_wse_width(prepared.wse_model,prepared.width_model)
    fit=core.fit_parameters(train,v15.make_core_config(CFG,True,corrected),CFG.seed,
        n_fixed_value=n_ref,beta_fixed_value=1.,depth0_fixed_value=max(.5,float(np.ptp(anchor))),
        width0_fixed_value=0.,parameter_source='V15_reference_n_and_fixed_geometry_no_Q_fitting')
    if fit is None:return None
    assert fit['optimizer_selected_stage']=='fixed' and fit['optimizer_nit']==0 and fit['optimizer_nfev']==1
    assert all(fit[k] for k in ['n_fixed','beta_fixed','depth0_fixed','width0_fixed'])
    fit.update(quality.curve_metadata(train))
    return fit

def fitting_input(frame):
    # Exact equality only. The cache is local to one new station/fold, with one
    # seed and one identical joint-parameter core configuration. No old fit is
    # used. All original training inputs and prepared numerical inputs match.
    cols=['obs_id','time_dt','wse_model','width','width_model','Q','slope']
    cols += sorted(c for c in frame if c.startswith('global_'))
    raw=frame[cols].reset_index(drop=True)
    prepared,_=core.prepare_training(frame,True)
    numerical=None if prepared is None else prepared[['obs_id','wse_model','width','width_model','Q','slope_safe']].reset_index(drop=True)
    return raw,numerical

def station_job(path_text,n_ref,n_source,selected_methods,pilot_fold=None):
    start=time.monotonic();path=Path(path_text);sid=core.normalize_station_id(path.stem)
    station_dir=ROOT/'private/checkpoints'/sid
    station_dir.mkdir(parents=True,exist_ok=True)
    base,audit,_,overpass=v15.load_station_with_global_qc(path,CFG.aggregate_nodes_by_overpass)
    assignment=split_station(base,sid)
    eligibility={'station_id':sid,'source_rows':len(base),'qc_pass_rows':int(base.global_qc_pass.sum()),
        'eligible':assignment is not None,'reason':'eligible' if assignment is not None else 'fewer_than_7_QC_pass_records',
        'n_reference':n_ref,'n_reference_source':n_source}
    if assignment is None:
        dump_json(station_dir/'eligibility.json',eligibility)
        return {'eligibility':eligibility,'elapsed_s':time.monotonic()-start,'fits_completed':0}
    eligibility.update(qc_pass_wse_unique=base.loc[base.global_qc_pass,'wse_model'].nunique(),
        qc_pass_wse_span=float(np.ptp(base.loc[base.global_qc_pass,'wse_model'])))
    assignment=assignment.merge(base[['obs_id','global_qc_pass','wse_model','time_dt']],on='obs_id',validate='one_to_one')
    assignment['station_id']=sid;assignment['validation_scheme']=SCHEME;assignment['repeat_id']=0
    assignment.to_csv(station_dir/'observation_fold_assignment_RESTRICTED.csv',index=False,encoding='utf-8-sig')
    overpass.to_pickle(station_dir/'overpass_QC_audit_RESTRICTED.pkl')
    fits_completed=0;failures=0
    for fold in range(5):
        if pilot_fold is not None and fold!=pilot_fold:continue
        train_ids=assignment.loc[assignment.fold_id.ne(fold),'obs_id'].tolist()
        test_ids=assignment.loc[assignment.fold_id.eq(fold),'obs_id'].tolist()
        assert not set(train_ids)&set(test_ids) and set(train_ids)|set(test_ids)==set(base.obs_id)
        pairs,diag=v15.make_split_frame_versions(base,train_ids,test_ids)
        pairs['observed_fixed']=pairs['no_monotonic']
        pairs['corrected_fixed']=pairs['full']
        ptr,pte=pairs['full']
        prepared,_=core.prepare_training(ptr,True)
        assert prepared is not None and len(prepared)>=5 and len(pte)>=1,(sid,fold)
        fit_cache=[]
        for method in selected_methods:
            checkpoint=station_dir/f'fold_{fold}_{method}.pkl'
            tr,te=pairs[method];t0=time.monotonic()
            inputs=fitting_input(tr) if method in EQUIVALENT_JOINT else None
            if checkpoint.exists():
                if inputs is not None:
                    with checkpoint.open('rb') as f:existing=pickle.load(f)
                    if existing['success']:fit_cache.append((method,inputs,existing['fit']))
                continue
            outcome={'station_id':sid,'method':method,'fold_id':fold,'repeat_id':0,'validation_scheme':SCHEME}
            try:
                seed=v15.pipeline.stable_seed(sid,'wse_5fold_fit',fold,base=CFG.seed)
                reused=None
                if inputs is not None:
                    for cached_method,cached_input,cached_fit in fit_cache:
                        if inputs[0].equals(cached_input[0]) and inputs[1] is not None and inputs[1].equals(cached_input[1]):
                            reused=(cached_method,cached_fit);break
                if reused is not None:
                    fit=copy.deepcopy(reused[1])
                    fit.update(quality.curve_metadata(tr))
                    fit['parameter_source']='station_joint_n_beta_WSE0_width0' + ('' if method=='full' else '_'+method)
                elif method in ['observed_fixed','corrected_fixed']:
                    fit=fixed_fit(tr,n_ref,method=='corrected_fixed')
                else:
                    fit=v15.fit_method(method,tr,CFG,seed,n_ref)
                if fit is None:raise RuntimeError('fit_returned_None')
                fit['optimizer_actually_run_for_this_method']=reused is None and method not in ['observed_fixed','corrected_fixed']
                fit['equivalent_new_fold_fit_reused_from_method']=reused[0] if reused else ''
                if inputs is not None:fit_cache.append((method,inputs,fit))
                if method=='observed_fixed':
                    pred=core.predict_frame(fit,te,SCHEME,fold,0,'validation',False)
                    pred['method']=method;pred['method_label']=v15.METHOD_LABELS[method]
                    pred['validation_scheme']=SCHEME;pred['model_kind']='hydraulic'
                else:
                    pred=v15.pipeline.predict_method(method,fit,te,SCHEME,fold,0,'validation')
                pred=v15.pipeline.decorate_predictions(pred,base)
                lo=float(fit['train_wse_min']);hi=float(fit['train_wse_max']);span=hi-lo
                pred['train_wse_min']=lo;pred['train_wse_max']=hi;pred['calibration_wse_span']=span
                pred['domain']=np.where(pred.WSE<lo,'below',np.where(pred.WSE>hi,'above','within'))
                dist=np.maximum(lo-pred.WSE,np.maximum(pred.WSE-hi,0))
                pred['normalized_extrapolation_distance']=dist/span if span>0 else np.nan
                assert len(pred)==len(te) and not pred.obs_id.duplicated().any()
                assert np.isfinite(pred.Q_calculated).all() and np.isfinite(pred.Q_observed).all()
                param=v15.parameter_row(fit,sid,method,SCHEME,fold,0,len(tr),len(te))
                param.update(n_reference=n_ref,n_reference_source=n_source,
                    train_wse_unique=int(tr.wse_model.nunique()),train_Q_std=float(tr.Q.std(ddof=0)),
                    train_Q_min=float(tr.Q.min()),train_Q_max=float(tr.Q.max()),
                    train_raw_count=len(train_ids),validation_raw_count=len(test_ids),
                    training_fraction=float(len(tr)/(len(tr)+len(te))),**diag)
                outcome.update(success=True,predictions=pred,parameters=param,fit=fit)
                fits_completed+=1
            except Exception as exc:
                outcome.update(success=False,reason=str(exc),traceback=traceback.format_exc(),
                    train_n=len(tr),validation_n=len(te))
                failures+=1
            outcome['elapsed_s']=time.monotonic()-t0
            temporary=checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as f:pickle.dump(outcome,f,pickle.HIGHEST_PROTOCOL)
            temporary.replace(checkpoint)
    dump_json(station_dir/'eligibility.json',eligibility)
    return {'eligibility':eligibility,'elapsed_s':time.monotonic()-start,'fits_completed':fits_completed,'failures':failures}

def initialize():
    import scipy,sklearn,matplotlib
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
        'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__,
        'thread_limits':{n:os.environ[n] for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    dump_json(ROOT/'checks/environment.json',environment)
    metadata=core.metadata_lookup(Path(CFG.metadata_file))
    values=[v15.pipeline.finite_number(v.get('N_River')) for v in metadata.values()]
    valid=[v for v in values if np.isfinite(v) and v>0]
    global_n=float(np.clip(np.median(valid) if valid else .035,core.N_MIN,core.N_MAX))
    jobs=[]
    for path in sorted(Path(CFG.data_dir).glob('*.csv')):
        sid=core.normalize_station_id(path.stem)
        n=v15.pipeline.finite_number(metadata.get(sid,{}).get('N_River'))
        source='metadata_N_River' if np.isfinite(n) and n>0 else 'V15_global_metadata_median'
        n=float(np.clip(n if np.isfinite(n) and n>0 else global_n,core.N_MIN,core.N_MAX))
        jobs.append((str(path),n,source))
    return jobs

def aggregate():
    predictions=[];parameters=[];failures=[];assignments=[];eligibility=[]
    for station_dir in sorted((ROOT/'private/checkpoints').iterdir()):
        ep=station_dir/'eligibility.json'
        if ep.exists():eligibility.append(json.loads(ep.read_text(encoding='utf-8')))
        ap=station_dir/'observation_fold_assignment_RESTRICTED.csv'
        if ap.exists():assignments.append(pd.read_csv(ap,dtype={'station_id':str,'obs_id':str}))
        for p in sorted(station_dir.glob('fold_*.pkl')):
            with p.open('rb') as f:x=pickle.load(f)
            if x['success']:
                predictions.append(x['predictions']);parameters.append({**x['parameters'],'fit_elapsed_s':x['elapsed_s']})
            else:failures.append({k:v for k,v in x.items() if k not in ['fit','predictions']})
    if predictions:
        pred=pd.concat(predictions,ignore_index=True)
        assert not pred.duplicated(['station_id','obs_id','method']).any()
        pred.to_csv(ROOT/'private/validation_predictions_RESTRICTED.csv',index=False,encoding='utf-8-sig')
        par=pd.DataFrame(parameters)
        par.to_csv(ROOT/'private/parameter_estimates_RESTRICTED.csv',index=False,encoding='utf-8-sig')
        excluded=[c for c in par if c.startswith('train_Q_')]
        par.drop(columns=excluded).to_csv(ROOT/'tables/parameter_estimates.csv',index=False,encoding='utf-8-sig')
    if assignments:
        a=pd.concat(assignments,ignore_index=True)
        a.to_csv(ROOT/'private/split_assignments_RESTRICTED.csv',index=False,encoding='utf-8-sig')
        # Public audit has counts only, no dates/WSE or observation identifiers.
        counts=a.groupby(['station_id','qc_group','fold_id']).size().reset_index(name='validation_n')
        counts.to_csv(ROOT/'tables/fold_sample_counts.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(eligibility).to_csv(ROOT/'tables/station_eligibility.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(failures).to_csv(ROOT/'checks/failures.csv',index=False,encoding='utf-8-sig')
    report={'methods':METHODS,'successful_station_fold_method_fits':len(parameters),'failed_fits':len(failures),
        'eligible_stations':sum(x['eligible'] for x in eligibility),'stations_audited':len(eligibility),
        'prediction_rows':sum(len(x) for x in predictions),'source_files_modified':False}
    dump_json(ROOT/'checks/run_status.json',report);print(json.dumps(report),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=20)
    ap.add_argument('--methods',nargs='+',default=METHODS);ap.add_argument('--pilot-station')
    ap.add_argument('--pilot-fold',type=int);ap.add_argument('--aggregate-only',action='store_true')
    args=ap.parse_args()
    for d in ['private/checkpoints','tables','figures','checks']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    if args.aggregate_only:aggregate();return
    jobs=initialize()
    if args.pilot_station:jobs=[x for x in jobs if core.normalize_station_id(Path(x[0]).stem)==args.pilot_station]
    start=time.monotonic();done=0;fits=0;eligible=0;errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(station_job,*job,args.methods,args.pilot_fold):job[0] for job in jobs}
        for future in as_completed(futures):
            done+=1
            try:
                x=future.result();fits+=x['fits_completed'];eligible+=int(x['eligibility']['eligible'])
                print(json.dumps({'station':x['eligibility']['station_id'],'done':done,'total':len(jobs),
                    'eligible':eligible,'new_fits':fits,'station_seconds':round(x['elapsed_s'],1),
                    'elapsed_seconds':round(time.monotonic()-start,1),'failures':x.get('failures',0)}),flush=True)
            except Exception as exc:
                errors.append({'source':futures[future],'reason':str(exc),'traceback':traceback.format_exc()})
                print(json.dumps({'station_error':futures[future],'reason':str(exc)}),flush=True)
            dump_json(ROOT/'checks/progress.json',{'done':done,'total':len(jobs),'eligible':eligible,
                'new_fits':fits,'elapsed_seconds':time.monotonic()-start,'station_errors':errors})
    aggregate()
    if errors:raise RuntimeError(f'{len(errors)} station errors, see checks/progress.json')

if __name__=='__main__':main()
