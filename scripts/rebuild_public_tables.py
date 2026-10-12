#!/usr/bin/env python3
"""Export released aggregate CSV tables as readable Markdown without gauge inputs."""
from pathlib import Path
import argparse
import csv
import json

def cell(value):
    return str(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Separate writable output directory; leave released files unchanged')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    release_root = Path(__file__).resolve().parents[1]
    if output == data or output.is_relative_to(data) or output.is_relative_to(release_root):
        parser.error('Use an output directory outside the unpacked release to preserve its frozen files and checksums.')
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for p in sorted(data.rglob('*.csv')):
        if p.name in ('data_dictionary.csv', 'paper_traceability.csv'):
            continue
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            rows = list(reader)
        if not columns:
            continue
        rel = p.relative_to(data)
        target = output / rel.with_suffix('.md')
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ['# ' + p.stem.replace('_', ' '), '', 'Source aggregate: `data/' + rel.as_posix() + '`.',
                 '', 'This is a formatted export of released aggregates, not a recalculation from individual observations.',
                 '', '| ' + ' | '.join(cell(c) for c in columns) + ' |',
                 '| ' + ' | '.join('---' for _ in columns) + ' |']
        lines += ['| ' + ' | '.join(cell(row.get(c, '')) for c in columns) + ' |' for row in rows]
        target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        outputs.append({'source': rel.as_posix(), 'output': target.relative_to(output).as_posix(), 'rows': len(rows)})
    if not outputs:
        parser.error('No aggregate CSV files found.')
    report = {'method': 'Direct formatting of public aggregate CSVs; no model fitting or metric recomputation',
              'outputs': outputs}
    (output / 'reconstruction_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Exported', len(outputs), 'public aggregate tables to', output)

if __name__ == '__main__':
    main()
