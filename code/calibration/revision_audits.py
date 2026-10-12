"""Export inspectable coverage and long-form saved curve parameters."""
from pathlib import Path
import json
import pandas as pd


def export(results):
    table=Path(results)/'tables'
    qc=pd.read_csv(table/'overpass_qc_audit.csv')
    reasons=['global_hampel_wse','global_hampel_width','global_hampel_log_slope','global_invalid_slope']
    station=qc.groupby('station_id').agg(complete_observations=('obs_id','size'),retained_observations=('global_qc_pass','sum'))
    station['retained_fraction']=station.retained_observations/station.complete_observations
    for field in reasons:station[field+'_count']=qc.groupby('station_id')[field].sum()
    station.reset_index().to_csv(table/'global_qc_station_summary.csv',index=False,encoding='utf-8-sig')
    assignment=pd.read_csv(table/'split_assignments.csv')
    support=assignment.groupby(['station_id','validation_scheme','repeat_id','fold_id','role']).agg(assigned_count=('obs_id','size'),qc_retained_count=('global_qc_pass','sum')).reset_index()
    support['qc_retained_fraction']=support.qc_retained_count/support.assigned_count
    support.to_csv(table/'split_qc_coverage.csv',index=False,encoding='utf-8-sig')
    params=pd.read_csv(table/'parameter_estimates.csv')
    rows=[]
    for row in params.to_dict('records'):
        encoded=row.get('width_curve_parameters_json')
        if not isinstance(encoded,str):continue
        curve=json.loads(encoded)
        for i in range(len(curve['curve_wse'])):
            rows.append({**{k:row[k] for k in ['station_id','method','validation_scheme','repeat_id','fold_id']},
                         'anchor_index':i,**{k:curve[k][i] for k in curve},
                         'initial_width_corrected':row['initial_width_corrected']})
    pd.DataFrame(rows).to_csv(table/'width_curve_anchor_audit.csv',index=False,encoding='utf-8-sig')
    # Record native-population metrics and exact common-support effects explicitly.
    explanation={
        'global_qc_station_summary.csv':'Whole eligible station time series; reason counts overlap.',
        'split_qc_coverage.csv':'Assigned and retained counts, including splits with failed or unavailable predictions.',
        'width_curve_anchor_audit.csv':'Calibration anchors before initial check, after isotonic fitting and after growth cap.',
        'experiment_overall_summary.csv':'Each method on its own evaluation support; no-Hampel can have more observations.',
        'ablation_paired_test_summary.csv':'Both models rescored on exactly shared held-out observation IDs before station-paired effects.',
        'parameter_estimates.csv':'Hydraulic parameters, curve anchors, QC limits and optimization starts saved per fit.',
    }
    (Path(results)/'revised_table_definitions.json').write_text(json.dumps(explanation,indent=2),encoding='utf-8')
    manifest_path=table/'table_manifest.csv'
    manifest=pd.read_csv(manifest_path)
    extra=[]
    for file in ['global_qc_station_summary.csv','split_qc_coverage.csv','width_curve_anchor_audit.csv']:
        d=pd.read_csv(table/file);extra.append({'file':file,'row_count':len(d),'column_count':len(d.columns)})
    manifest=manifest[~manifest.file.isin([x['file'] for x in extra])]
    pd.concat([manifest,pd.DataFrame(extra)],ignore_index=True).to_csv(manifest_path,index=False,encoding='utf-8-sig')
