"""Audited v17b -> v16 SoS matching. Reuses source products, never old Q_model."""
import numpy as np
import pandas as pd
from common import OBS, METRICS, read_csv, normalize, save, metrics, interval, sha256, keep_rule

PRODUCTS = {
    'sos_consensus_q': 'Consensus',
    'sos_metroman_q': 'MetroMan',
    'sos_momma_q': 'MOMMA',
    'sos_sic4dvar_q': 'SIC4DVar',
}
COHORTS = ['high_confidence', 'mapped_including_medium', 'high_le5km', 'high_le2km']

def match_sos(obs, args, out):
    audit = read_csv(args.station_audit, dtype_backend='numpy_nullable')
    assert audit.station_id.is_unique
    for col in ('v16_reach_id','v17b_reach_id','nearest_v17b_node_id'):
        audit[col] = normalize(audit[col])
    audit = audit[['station_id','mapping_status','mapping_confidence','v16_reach_id','v17b_reach_id','nearest_v17b_node_id']]
    d = obs.merge(audit,on='station_id',how='left',validate='many_to_one')
    # Static crosswalk is reusable only if the nearest-node/reach assignment is unchanged.
    known = d.v17b_reach_id.notna()
    assert normalize(d.loc[known,'reach_id']).equals(normalize(d.loc[known,'v17b_reach_id']))
    assert normalize(d.loc[known,'node_id']).equals(normalize(d.loc[known,'nearest_v17b_node_id']))
    mapped=d[d.mapping_status.isin(['stable_same_id_high','remapped_high','remapped_medium']) & d.v16_reach_id.notna()].copy()
    matched, files = [], []
    for reach,g in mapped.groupby('v16_reach_id'):
        path=args.sos_raw_dir/f'{reach}.csv'
        if not path.exists():
            files.append(dict(reach_id=reach,status='missing_file',model_rows=len(g)))
            continue
        digest=sha256(path)
        raw=pd.read_csv(path,dtype={'reach_id':'string','sword_version':'string'})
        if not {'reach_id','time_str'} <= set(raw):
            raise ValueError(f'Malformed SoS file: {path}')
        raw['time_sos']=pd.to_datetime(raw.time_str,utc=True,errors='coerce')
        version=normalize(raw.get('sword_version',pd.Series(pd.NA,index=raw.index,dtype='string')))
        valid=normalize(raw.reach_id).eq(reach) & (version.isna() | version.eq('16')) & raw.time_sos.notna()
        raw=raw[valid].copy()
        valid_rows_before_time_filter=len(raw)
        # Duplicate products outside the prediction window cannot participate in
        # a 30-minute nearest match. Exclude them before checking ambiguity.
        window_start=g.time.min()-pd.Timedelta(minutes=30)
        window_end=g.time.max()+pd.Timedelta(minutes=30)
        raw=raw[raw.time_sos.between(window_start,window_end)].copy()
        for col in PRODUCTS:
            raw[col]=pd.to_numeric(raw[col],errors='coerce') if col in raw else np.nan
            raw.loc[~np.isfinite(raw[col]) | raw[col].le(0) | raw[col].ge(1e10),col]=np.nan
        raw=raw[['time_sos',*PRODUCTS]].drop_duplicates()
        if raw.duplicated('time_sos').any():
            raise ValueError(f'Conflicting SoS rows at the same reach/time: {path}')
        if raw.empty:
            files.append(dict(reach_id=reach,status='no_rows_in_matching_window',model_rows=len(g),
                              valid_rows_before_time_filter=valid_rows_before_time_filter,sha256=digest))
            continue
        joined=pd.merge_asof(g.sort_values('time'),raw.sort_values('time_sos'),
                             left_on='time',right_on='time_sos',direction='nearest',tolerance=pd.Timedelta(minutes=30))
        joined['time_offset_seconds']=(joined.time-joined.time_sos).abs().dt.total_seconds()
        matched.append(joined)
        files.append(dict(reach_id=reach,status='read',model_rows=len(g),source_rows=len(raw),
                          valid_rows_before_time_filter=valid_rows_before_time_filter,
                          time_matches=int(joined.time_sos.notna().sum()),sha256=digest))
        assert sha256(path)==digest,f'SoS source changed: {path}'
    fa=pd.DataFrame(files)
    save(fa,out,'sos_source_file_audit')
    if not matched:raise ValueError('No mapped raw SoS observations available')
    wide=pd.concat(matched,ignore_index=True)
    assert not wide.duplicated(OBS).any()
    assert wide.time_offset_seconds.dropna().le(1800).all()
    save(wide,out,'sos_matched_observations')
    build=[]
    for sc,g in obs.groupby('validation_scheme'):
        w=wide[wide.validation_scheme.eq(sc)]
        build.append(dict(validation_scheme=sc,model_observations=len(g),model_stations=g.station_id.nunique(),
            mapped_observations=int(mapped.validation_scheme.eq(sc).sum()),
            matched_observations=int(w.time_sos.notna().sum()),positive_any=int(w[list(PRODUCTS)].notna().any(axis=1).sum()),
            fixed_four_positive=int(w[list(PRODUCTS)].notna().all(axis=1).sum())))
    save(pd.DataFrame(build),out,'sos_build_coverage')
    return wide

def cohort(d,name):
    if name=='mapped_including_medium':return d
    z=d[d.mapping_confidence.eq('high')]
    if name=='high_le5km':z=z[z.nearest_node_distance_km.le(5)]
    if name=='high_le2km':z=z[z.nearest_node_distance_km.le(2)]
    return z

def analyze_sos(obs,args,out):
    wide=match_sos(obs,args,out)
    records, pooled, coverage=[],[],[]
    # Primary fixed-four cohorts use >=4 pairs BEFORE screening. Observation-only
    # screens retain low-count stations in pooled scores; a separate sensitivity
    # explicitly reapplies the station count rule.
    for co in COHORTS:
        for sc,frame in cohort(wide,co).groupby('validation_scheme'):
            for support_kind in ['pairwise','fixed_four']:
                product_frames=[]
                if support_kind=='fixed_four':
                    z=frame[frame[list(PRODUCTS)].notna().all(axis=1)]
                    product_frames=[(col,z) for col in PRODUCTS]
                else:
                    product_frames=[(col,frame[frame[col].notna()]) for col in PRODUCTS]
                for col,raw in product_frames:
                    counts=raw.groupby('station_id').size()
                    base=raw[raw.station_id.isin(counts[counts.ge(args.sos_min_pairs)].index)]
                    for rule in ['baseline','slope5','slope10','slope20','error10','absolute_top1']:
                        screened=keep_rule(base,rule)
                        for station_policy in ['observation_only','reapply_min_pairs']:
                            kept=screened
                            if station_policy=='reapply_min_pairs':
                                c=kept.groupby('station_id').size()
                                kept=kept[kept.station_id.isin(c[c.ge(args.sos_min_pairs)].index)]
                            info=dict(cohort=co,validation_scheme=sc,support=support_kind,product=PRODUCTS[col],rule=rule,station_policy=station_policy)
                            coverage.append(dict(info,positive_rows_before_count=len(raw),baseline_rows=len(base),retained_rows=len(kept),
                                row_retention=len(kept)/len(base) if len(base) else np.nan,
                                baseline_stations=base.station_id.nunique(),retained_stations=kept.station_id.nunique()))
                            for model,pred in [('Local','Q_model'),(PRODUCTS[col],col)]:
                                pooled.append(dict(info,model=model,**metrics(kept.Q_observed,kept[pred],min_pairs=1)))
                            # Per-station scores have the documented >=4-pair rule in either policy.
                            if station_policy=='observation_only':
                                for sid,g in kept.groupby('station_id'):
                                    a=metrics(g.Q_observed,g.Q_model,args.sos_min_pairs)
                                    b=metrics(g.Q_observed,g[col],args.sos_min_pairs)
                                    for m in METRICS:
                                        sign=-1 if m in ['MAE','RMSE','MdAPE'] else 1
                                        records.append(dict(info,station_id=sid,metric=m,n=a['n'],local=a[m],official=b[m],
                                                            local_advantage=sign*(a[m]-b[m])))
    paired=pd.DataFrame(records,columns=['cohort','validation_scheme','support','product','rule','station_policy','station_id','metric','n','local','official','local_advantage'])
    save(paired,out,'sos_paired_station_metrics')
    save(pd.DataFrame(pooled),out,'sos_pooled_metrics')
    save(pd.DataFrame(coverage),out,'sos_retention')
    rows=[]
    for key,g in paired.dropna(subset=['local','official']).groupby(['cohort','validation_scheme','support','product','rule','metric']):
        info=dict(zip(['cohort','validation_scheme','support','product','rule','metric'],key))
        rows.append(dict(info,local_median=g.local.median(),official_median=g.official.median(),
                         local_better_fraction=g.local_advantage.gt(0).mean(),**interval(g.local_advantage,args.bootstrap)))
    save(pd.DataFrame(rows,columns=['cohort','validation_scheme','support','product','rule','metric','local_median','official_median','local_better_fraction','stations','median','ci_low','ci_high']),out,'sos_paired_summary')
    return dict(sos_matched_rows=int(wide.time_sos.notna().sum()),sos_match_keys_unique=True,sos_max_time_offset_seconds=float(wide.time_offset_seconds.max()))
