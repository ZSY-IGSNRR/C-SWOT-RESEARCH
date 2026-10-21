#!/usr/bin/env python3
"""Format externally supplied aggregate CSV tables without fitting a model."""
from pathlib import Path
import argparse
import csv
import json

from check_release import FORBIDDEN_COLUMNS


def cell(value):
    return str(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data',
                        help='External aggregate CSV directory; current repository does not bundle data/')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Separate writable output directory; leave research inputs unchanged')
    args = parser.parse_args(argv)
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    release_root = Path(__file__).resolve().parents[1]
    if not data.is_dir():
        parser.error('This code-and-figures repository does not bundle aggregate tables. '
                     'Supply --data-dir pointing to an external aggregate CSV directory.')
    if output == data or output.is_relative_to(data) or output.is_relative_to(release_root):
        parser.error('Use an output directory outside the repository and input data directory.')
    sources = []
    for p in sorted(data.rglob('*.csv')):
        if p.name in ('data_dictionary.csv', 'paper_traceability.csv'):
            continue
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            bad = {c.strip().lower() for c in columns}.intersection(FORBIDDEN_COLUMNS)
            if bad:
                parser.error('Expected aggregates, found observation/station columns in ' + str(p) + ': ' + ', '.join(sorted(bad)))
            rows = list(reader)
        if columns:
            sources.append((p, columns, rows))
    if not sources:
        parser.error('No aggregate CSV files found in --data-dir.')
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for p, columns, rows in sources:
        rel = p.relative_to(data)
        target = output / rel.with_suffix('.md')
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ['# ' + p.stem.replace('_', ' '), '', 'Source aggregate: `' + rel.as_posix() + '`.',
                 '', 'Formatted export of supplied aggregates; no recalculation from individual observations.',
                 '', '| ' + ' | '.join(cell(c) for c in columns) + ' |',
                 '| ' + ' | '.join('---' for _ in columns) + ' |']
        lines += ['| ' + ' | '.join(cell(row.get(c, '')) for c in columns) + ' |' for row in rows]
        target.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        outputs.append({'source': rel.as_posix(), 'output': target.relative_to(output).as_posix(), 'rows': len(rows)})
    report = {'method': 'Formatting of externally supplied aggregate CSVs; no model fitting or metric recomputation',
              'outputs': outputs}
    (output / 'reconstruction_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Exported', len(outputs), 'aggregate tables to', output)
    return 0


if __name__ == '__main__':
    main()
