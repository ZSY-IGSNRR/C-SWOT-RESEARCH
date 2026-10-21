"""Final CSV-only independent audit, no numerical libraries or new fitting."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import time

from _paths import work_root
ROOT = work_root()
csv.field_size_limit(100_000_000)
METHODS = ['full','beta_fixed_1','width_train_only','no_initial_width_guard',
    'no_width_growth_guard','monotonic_only','no_hampel','no_monotonic',
    'fixed_reference_n','n_only_fixed_geometry','empirical_rating_curve',
    'observed_fixed','corrected_fixed']
EQUIVALENT = {'full','width_train_only','no_initial_width_guard',
    'no_width_growth_guard','monotonic_only'}
FIXED = {'observed_fixed','corrected_fixed'}


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def truth(text):
    assert str(text).strip().lower() in ['true','false','1','0'], 'Invalid boolean token'
    return str(text).strip().lower() in ['true','1']


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9)


def main():
    start = time.monotonic()
    paths = {name: ROOT/relative for name,relative in {
        'expected':'private/expected_fold_assignments_RESTRICTED.csv',
        'actual':'private/split_assignments_RESTRICTED.csv',
        'parameters':'private/parameter_estimates_RESTRICTED.csv',
        'predictions':'private/validation_predictions_RESTRICTED.csv',
        'eligibility':'tables/station_eligibility.csv',
        'public_parameters':'tables/parameter_estimates.csv'}.items()}
    expected = {}
    for row in rows(paths['expected']):
        key=(row['station_id'],row['obs_id'])
        assert key not in expected, 'Expected assignment duplicated'
        expected[key]=(int(row['fold_id']),row['qc_group'],int(row['wse_stratum']))
    actual={}
    by_station=defaultdict(list)
    for row in rows(paths['actual']):
        key=(row['station_id'],row['obs_id'])
        assert key not in actual, 'Actual assignment duplicated'
        assignment=(int(row['fold_id']),row['qc_group'],int(row['wse_stratum']))
        assert expected[key]==assignment, 'Expected and actual assignments differ'
        assert truth(row['global_qc_pass'])==(row['qc_group']=='pass')
        assert row['repeat_id']=='0' and row['validation_scheme']=='wse_stratified_5fold'
        actual[key]={**row,'fold':int(row['fold_id']),'wse':float(row['wse_model'])}
        by_station[row['station_id']].append(actual[key])
    assert set(actual)==set(expected), 'Expected assignment coverage mismatch'
    stations=set(by_station)
    assert len(stations)==422 and len(actual)==8891
    passed={key for key,row in actual.items() if row['qc_group']=='pass'}
    assert len(passed)==7615
    eligibility=list(rows(paths['eligibility']))
    assert len(eligibility)==1143 and len({r['station_id'] for r in eligibility})==1143
    assert {r['station_id'] for r in eligibility if truth(r['eligible'])}==stations
    folds={}
    for sid,group in by_station.items():
        pass_counts=Counter(r['fold'] for r in group if r['qc_group']=='pass')
        assert set(pass_counts)==set(range(5)) and max(pass_counts.values())-min(pass_counts.values())<=1
        for fold in range(5):
            for method in METHODS:
                usable=group if method=='no_hampel' else [r for r in group if r['qc_group']=='pass']
                train=[r for r in usable if r['fold']!=fold]
                test=[r for r in usable if r['fold']==fold]
                assert len(train)>=5 and len(test)>=1
                assert not {r['obs_id'] for r in train}&{r['obs_id'] for r in test}
                folds[(sid,fold,method)]={'train_n':len(train),'test_n':len(test),
                    'train_raw_n':sum(r['fold']!=fold for r in group),
                    'test_raw_n':sum(r['fold']==fold for r in group),
                    'wse_values':sorted(set(r['wse'] for r in train))}
    params={}
    optimizer_counts=Counter();reuse_counts=Counter();method_counts=Counter()
    curve_checks=0
    for row in rows(paths['parameters']):
        key=(row['station_id'],int(row['fold_id']),row['method'])
        assert key not in params, 'Duplicate parameter fit'
        params[key]=row;method_counts[row['method']]+=1
        f=folds[key]
        assert int(row['repeat_id'])==0
        assert int(row['train_count_before_qc'])==f['train_n']
        assert int(row['validation_count'])==f['test_n']
        assert int(row['train_after_hampel_count'])==f['train_n']
        assert int(row['train_raw_count'])==f['train_raw_n']
        assert int(row['validation_raw_count'])==f['test_raw_n']
        assert close(row['train_wse_min'],f['wse_values'][0])
        assert close(row['train_wse_max'],f['wse_values'][-1])
        assert close(row['train_wse_range'],f['wse_values'][-1]-f['wse_values'][0])
        assert int(row['train_wse_unique'])==len(f['wse_values'])
        encoded=row.get('width_curve_parameters_json','')
        if encoded:
            curve=json.loads(encoded)
            anchor=curve['curve_wse']
            assert len(anchor)==len(f['wse_values'])
            assert all(close(a,b) for a,b in zip(anchor,f['wse_values']))
            assert close(anchor[0],row['train_wse_min']) and close(anchor[-1],row['train_wse_max'])
            curve_checks+=1
        ran=truth(row['optimizer_actually_run_for_this_method'])
        reused=row['equivalent_new_fold_fit_reused_from_method']
        if row['method'] in FIXED:
            assert not ran and not reused
            assert row['optimizer_selected_stage']=='fixed' and int(float(row['optimizer_nit']))==0
            assert int(float(row['optimizer_nfev']))==1
            optimizer_counts['all_four_fixed_no_optimization']+=1
        elif reused:
            assert not ran and row['method'] in EQUIVALENT and reused in EQUIVALENT
            assert reused!=row['method']
            optimizer_counts['same_new_fold_equivalent_fit_reused']+=1
            reuse_counts[f'{reused} -> {row["method"]}']+=1
        else:
            assert ran
            optimizer_counts['fresh_model_optimization']+=1
    assert set(params)==set(folds) and len(params)==27430
    assert all(method_counts[m]==2110 for m in METHODS)
    reused_fit_checks=0
    numerical=['n','beta','WSE0','width0','depth0','A0','objective_loss',
        'train_wse_min','train_wse_max','train_wse_range','train_slope_fallback',
        'selected_validation_start','validation_optimizer_start_count',
        'optimizer_nit','optimizer_nfev']
    for (sid,fold,method),row in params.items():
        reused=row['equivalent_new_fold_fit_reused_from_method']
        if not reused:continue
        origin=params[(sid,fold,reused)]
        for name in numerical:
            assert row.get(name,'')==origin.get(name,''), 'Reused fit numerical value differs'
        ca=json.loads(row['width_curve_parameters_json'])
        cb=json.loads(origin['width_curve_parameters_json'])
        assert ca['curve_wse']==cb['curve_wse'] and ca['curve_width']==cb['curve_width']
        seen={method};at=reused
        while at:
            assert at not in seen, 'Fit reuse cycle'
            seen.add(at)
            at=params[(sid,fold,at)]['equivalent_new_fold_fit_reused_from_method']
        reused_fit_checks+=1
    observed={method:set() for method in METHODS};prediction_counts=Counter();per_fit=Counter()
    for row in rows(paths['predictions']):
        method=row['method'];obskey=(row['station_id'],row['obs_id'])
        key=(row['station_id'],int(row['fold_id']),method)
        assert obskey not in observed[method], 'An observation has repeated OOF predictions'
        observed[method].add(obskey)
        assert obskey in actual and int(row['fold_id'])==actual[obskey]['fold']
        assert row['repeat_id']=='0' and row['validation_scheme']=='wse_stratified_5fold'
        assert row['sample_role']=='validation' and row['scheme']=='wse_stratified_5fold'
        if method!='no_hampel':assert obskey in passed
        assert math.isfinite(float(row['Q_observed'])) and float(row['Q_observed'])>0
        assert math.isfinite(float(row['Q_calculated']))
        p=params[key]
        assert close(row['train_wse_min'],p['train_wse_min']) and close(row['train_wse_max'],p['train_wse_max'])
        lo,hi=float(p['train_wse_min']),float(p['train_wse_max']);wse=float(row['WSE'])
        assert close(wse,actual[obskey]['wse'])
        assert row['domain']==('below' if wse<lo else 'above' if wse>hi else 'within')
        per_fit[key]+=1;prediction_counts[method]+=1
    for method in METHODS:
        assert observed[method]==(set(actual) if method=='no_hampel' else passed)
    assert all(per_fit[key]==f['test_n'] for key,f in folds.items())
    assert sum(prediction_counts.values())==100271
    manifest=json.loads((ROOT/'checks/source_manifest.json').read_text(encoding='utf-8'))
    source_hashes_checked=0
    for row in manifest['checked_sources']:
        assert sha(row['path'])==row['sha256'], f'Source file changed: {row["path"]}'
        source_hashes_checked+=1
    frozen_checked=[]
    from _paths import CALIBRATION, SOURCE
    pinned=json.loads((SOURCE/'frozen_v15/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in sorted(pinned['scientific_core_sha256'].items()):
        path=CALIBRATION/name
        assert sha(path)==expected
        frozen_checked.append({'file':path.name,'sha256':sha(path)})
    from _paths import load_configuration
    reference=json.loads((SOURCE/'frozen_v15/configuration/calibration_v15.json').read_text(encoding='utf-8'))
    active=load_configuration()
    assert all(active[key]==value for key,value in reference.items() if key not in {'data_dir','metadata_file','output_dir','basin_shapefile','workers'})
    forbidden={'Q','Q_observed','Q_calculated','obs_id','time','time_dt','train_Q_min','train_Q_max','train_Q_std'}
    with paths['public_parameters'].open(encoding='utf-8-sig',newline='') as handle:
        fields=set(next(csv.reader(handle)))
    assert not fields&forbidden, 'Raw or identifying observation fields in public parameter table'
    status=json.loads((ROOT/'checks/run_status.json').read_text(encoding='utf-8'))
    assert status['failed_fits']==0 and status['successful_station_fold_method_fits']==len(params)
    report={'status':'PASS','independent_audit':'Python standard-library CSV/JSON/hashlib; no model fitting or source/checkpoint edits',
        'stations_audited':len(eligibility),'eligible_stations':len(stations),
        'expected_actual_assignments_equal':True,'raw_eligible_observations':len(actual),
        'full_QC_pass_observations':len(passed),'full_each_QC_pass_observation_predicted_exactly_once':True,
        'all_13_methods_each_station_has_5_fits':True,'station_fold_fits_per_method':dict(method_counts),
        'total_station_fold_method_fits':len(params),'prediction_rows':sum(prediction_counts.values()),
        'prediction_counts_by_method':dict(prediction_counts),
        'training_complement_check':'Every fit count, raw count, unique WSE anchors and WSE bounds reconciled against the other four assigned folds. Runner static audit confirms training IDs use fold_id != held-out fold; serialized parameter table does not store training observation IDs.',
        'all_training_sets_at_least_5':True,'all_validation_sets_nonempty':True,
        'geometry_curve_anchor_sets_checked':curve_checks,'geometry_curve_covers_exact_training_WSE_range':True,
        'optimizer_fit_counts':dict(optimizer_counts),'equivalent_fit_reuse_edges':dict(reuse_counts),
        'reused_fits_numeric_parameters_and_geometry_verified':reused_fit_checks,
        'reuse_is_same_station_same_new_fold_no_cycles':True,
        'all_recorded_source_hashes_rechecked':source_hashes_checked,
        'frozen_code_matches_original':frozen_checked,'frozen_config_matches_original':True,
        'public_parameter_table_excludes_raw_Q_and_observation_keys':True,
        'report_contains_no_observation_level_Q_Z_or_prediction_rows':True,
        'source_files_modified':False,'run_checkpoints_touched':False,
        'input_file_hashes':{name:sha(path) for name,path in paths.items()},
        'elapsed_seconds':time.monotonic()-start}
    output=ROOT/'checks/final_independent_audit.json'
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'PASS','fits':len(params),'gauges':len(stations),
        'full_OOF':len(passed),'prediction_rows':sum(prediction_counts.values()),
        'optimization':dict(optimizer_counts),'report':str(output),'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':
    main()
