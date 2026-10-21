"""Record the current authorized inputs and immutable scientific code hashes.

This is an adapted provenance helper, not a model calculation. It does not
copy scientific files or assume an earlier local experiment directory.
"""
from pathlib import Path
import hashlib,json
from _paths import work_root,CALIBRATION,SOURCE,configuration_path,load_configuration
ROOT=work_root()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    cfg=load_configuration()
    pinned=json.loads((SOURCE/'frozen_v15/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    paths=[CALIBRATION/name for name,expected in pinned['scientific_core_sha256'].items()]
    for p in paths:assert sha(p)==pinned['scientific_core_sha256'][p.name]
    paths += [configuration_path(),Path(cfg['metadata_file']),Path(cfg['basin_shapefile'])]
    data=sorted(Path(cfg['data_dir']).glob('*.csv'));paths+=data
    checked=[{'path':str(p),'sha256':sha(p),'group':'station_data_files' if p in data else 'source_files'} for p in paths]
    checks=ROOT/'checks';checks.mkdir(parents=True,exist_ok=True)
    report={'checked_sources':checked,'station_data_file_count':len(data),'all_scientific_core_hashes_match_pinned_V15':True,'source_files_modified':False,
        'validation_scheme':'standard WSE-stratified five-fold CV','no_temporal_validation':True,
        'privacy':'Source records and predictions remain in the external work root; do not add private/ to this repository.'}
    (checks/'source_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'checked_files':len(checked),'station_data_files':len(data),'all_core_hashes_match':True}))
if __name__=='__main__':main()
