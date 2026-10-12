"""Resume only stages 5-7 after a SoS-stage failure; keep completed tables."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import traceback
import pandas as pd
from common import read_csv, sha256
from analyze_postprocess import verify_baseline
from sos_analysis import analyze_sos
from plot_postprocess import plot_all

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('postprocess_results'))
    opts=p.parse_args();root=opts.output_dir
    if (root/'verification_report.json').exists():
        raise FileExistsError('Completed verification already exists; refusing to resume')
    failure=root/'FAILED.txt'
    if not failure.exists() or 'sos_analysis.py' not in failure.read_text(encoding='utf-8'):
        raise ValueError('This recovery entry requires a recorded SoS-stage failure')
    cfg=json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    for path,digest in cfg['sources'].items():
        assert sha256(path)==digest,f'Original source changed: {path}'
    args=SimpleNamespace(**cfg)
    for key in ['results_dir','output_dir','metadata','station_audit','sos_raw_dir','boundary_file']:
        setattr(args,key,Path(getattr(args,key)))
    assert root.resolve()==args.output_dir.resolve()
    tables=root/'tables'
    for name in ['retention','scale_spatial_performance_summary','occurrence_rates',
                 'adjusted_associations','association_model_support','empirical_scale_comparison',
                 'empirical_paired_repeat_metrics','observation_diagnostics','baseline_all_station_repeat_metrics']:
        if not (tables/f'{name}.csv').is_file():raise FileNotFoundError(name)
    original_tables={str(f):sha256(f) for f in tables.glob('*.csv') if not f.name.startswith('sos_')}
    report={'recovered_at_utc':datetime.now(timezone.utc).isoformat(),
        'reason':'Exclude SoS rows outside the actual prediction time span plus 30-minute tolerance before duplicate checks',
        'original_run_log_preserved':True,'completed_stages_reused':[1,2,3,4],
        'code_sha256':{str(f):sha256(f) for f in Path(__file__).parent.glob('*.py')}}
    try:
        print('[Recovery 1/3] Rebuilding SoS comparisons',flush=True)
        obs=read_csv(tables/'observation_diagnostics.csv');obs['time']=pd.to_datetime(obs.time,utc=True,errors='raise')
        sos=analyze_sos(obs,args,tables)
        print('[Recovery 2/3] Rendering all figures',flush=True)
        figures=plot_all(root,args.boundary_file)
        print('[Recovery 3/3] Verifying baseline and unchanged sources',flush=True)
        units=verify_baseline(read_csv(tables/'baseline_all_station_repeat_metrics.csv'),args)
        for path,digest in {**cfg['sources'],**original_tables}.items():
            assert sha256(path)==digest,f'Source or completed table changed: {path}'
        report.update(passed=True,smoke_test=cfg['smoke_test'],baseline_units_reconciled=units,
            independent_observations=len(obs),figure_count=len(figures),original_sources_unchanged=True,
            completed_tables_unchanged=True,main_model_refitted=False,**sos)
        (root/'recovery_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (root/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        failure.rename(root/'INITIAL_FAILURE.txt')
        print(json.dumps(report,indent=2),flush=True)
    except Exception:
        (root/'RECOVERY_FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':main()
