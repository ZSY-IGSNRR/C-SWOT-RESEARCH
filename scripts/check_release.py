#!/usr/bin/env python3
"""Verify a public-only release without reading or downloading gauge inputs."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET

ALLOWED_SUFFIXES = {'.py', '.ps1', '.cmd', '.json', '.csv', '.md', '.txt', '.cff',
                    '.bib', '.png', '.svg', '.pdf', '.drawio', '.yml', '.yaml'}
FORBIDDEN_COLUMNS = {'station_id', 'gauge_id', 'site_id', 'node_id', 'reach_id',
                     'observation_id', 'obs_id', 'q_obs', 'q_observed', 'observed_q',
                     'q_true', 'q_measured', 'q_calculated', 'q_pred', 'predicted_q',
                     'timestamp', 'datetime', 'latitude', 'longitude', 'lon', 'lat'}
FORBIDDEN_BASENAMES = {'data.pkl', 'response_audit.json', 'line_map.json',
                       'word_line_map.json', 'observation_diagnostics.csv',
                       'manuscript.docx'}
SECRET_NAME = re.compile(r'(password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)', re.I)
DRIVE_PATH = re.compile(r'^[A-Za-z]:[\\/]')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json-output', type=Path, help='Optional report outside the immutable release')
    args = parser.parse_args()
    root = args.root.resolve()
    errors = []
    checked = []
    required = ['README.md', 'LICENSE', 'LICENSE_CODE.txt', 'LICENSE_DATA.txt',
                'CITATION.cff', 'citation.bib', 'SHA256SUMS.txt',
                'data/data_dictionary.csv', 'data/paper_traceability.csv',
                'figures/figure_manifest.json', 'metadata/zenodo_metadata.json']
    for name in required:
        if not (root / name).is_file():
            errors.append('Missing required file: ' + name)
    checksum = root / 'SHA256SUMS.txt'
    if checksum.is_file():
        for line in checksum.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                expected, name = line.split('  ', 1)
            except ValueError:
                errors.append('Invalid checksum entry')
                continue
            if not re.fullmatch(r'[0-9a-f]{64}', expected):
                errors.append('Invalid hash format for: ' + name)
                continue
            candidate = (root / name).resolve()
            if not candidate.is_relative_to(root):
                errors.append('Unsafe checksum path: ' + name)
            elif not candidate.is_file():
                errors.append('Missing checksummed file: ' + name)
            elif sha256(candidate) != expected:
                errors.append('Checksum mismatch: ' + name)
            else:
                checked.append(name)
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and p.name != 'SHA256SUMS.txt'
                  and '__pycache__' not in p.parts}
        if actual != set(checked):
            errors.append('Checksum coverage differs from release files: ' +
                          ', '.join(sorted(actual.symmetric_difference(checked))))

    code_files = csv_files = vector_files = 0
    for p in root.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        license_text = ('third_party_licenses' in p.parts and
                        (p.name.upper().startswith('LICENSE') or p.name == 'COPYING'))
        if p.suffix.lower() not in ALLOWED_SUFFIXES and not license_text and p.name != 'LICENSE':
            errors.append('Unapproved file type: ' + rel)
        if p.name.lower() in FORBIDDEN_BASENAMES:
            errors.append('Excluded file class present: ' + rel)
        if p.suffix.lower() == '.csv':
            csv_files += 1
            with p.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
                bad = fields.intersection(FORBIDDEN_COLUMNS)
                if bad:
                    errors.append('Disallowed observation/station columns in ' + rel + ': ' + ', '.join(sorted(bad)))
                # Parse every row to catch malformed or ragged export files.
                for i, row in enumerate(reader, 2):
                    if None in row or any(value is None for value in row.values()):
                        errors.append('Malformed CSV row in ' + rel + ' at line ' + str(i))
        if p.suffix.lower() == '.json':
            try:
                json.loads(p.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError):
                errors.append('Invalid JSON: ' + rel)
        if p.suffix.lower() == '.py':
            code_files += 1
            try:
                text = p.read_text(encoding='utf-8-sig')
                tree = ast.parse(text, filename=rel)
                compile(tree, rel, 'exec')
            except (SyntaxError, UnicodeError):
                errors.append('Python syntax/encoding failure: ' + rel)
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        names = [x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)]
                        if any(SECRET_NAME.fullmatch(n) for n in names) and value.value.strip() and not value.value.startswith(('YOUR_', 'REPLACE_', '<')):
                            errors.append('Potential credential literal: ' + rel + ':' + str(node.lineno))
        if p.suffix.lower() == '.svg':
            vector_files += 1
            if 'Figure_3' in p.name:
                errors.append('Figure 3 native vector must be excluded: ' + rel)
            try:
                xml = ET.fromstring(p.read_bytes())
                for node in xml.iter():
                    if node.tag.rsplit('}', 1)[-1].lower() == 'script':
                        errors.append('Active SVG script present: ' + rel)
                    for k, v in node.attrib.items():
                        if k.rsplit('}', 1)[-1].lower() in ('href', 'src'):
                            if v.startswith(('file:', 'javascript:')) or DRIVE_PATH.match(v):
                                errors.append('Private/active vector link: ' + rel)
            except ET.ParseError:
                errors.append('Invalid SVG XML: ' + rel)
    fig_pngs = sorted(p.name for p in (root / 'figures').glob('Figure_*.png'))
    expected_figures = sorted(['Figure_' + str(n) + '.png' for n in range(1, 7)] +
                             ['Figure_S1.png', 'Figure_S2.png'])
    if fig_pngs != expected_figures:
        errors.append('Final figure PNG set differs from Figure 1–6/S1–S2')
    report = {'status': 'PASS' if not errors else 'FAIL', 'checksummed_files': len(checked),
              'python_files_syntax_checked': code_files, 'csv_files_checked': csv_files,
              'svg_files_checked': vector_files, 'errors': errors,
              'scope': 'Integrity, file/schema exclusions and syntax only; no restricted-input numerical rerun or rights certification.'}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1

if __name__ == '__main__':
    sys.exit(main())
