#!/usr/bin/env python3
"""Verify repository/release integrity without reading external gauge inputs."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

ALLOWED_SUFFIXES = {'.py', '.ps1', '.cmd', '.json', '.csv', '.md', '.txt', '.cff',
                    '.bib', '.png', '.jpg', '.jpeg', '.svg', '.pdf', '.drawio', '.yml', '.yaml'}
ALLOWED_BASENAMES = {'LICENSE', '.gitignore', '.gitattributes'}
EXCLUDED_DIRECTORIES = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache',
                        '.mypy_cache', '.ruff_cache', '.tox', '.nox', '.ipynb_checkpoints'}
FORBIDDEN_DIRECTORIES = {'private', 'private_cache', 'checkpoints', 'outputs'}
FORBIDDEN_COLUMNS = {'station_id', 'gauge_id', 'site_id', 'node_id', 'reach_id',
                     'observation_id', 'obs_id', 'q_obs', 'q_observed', 'observed_q',
                     'q_true', 'q_measured', 'q_calculated', 'q_pred', 'predicted_q',
                     'timestamp', 'datetime', 'latitude', 'longitude', 'lon', 'lat'}
FORBIDDEN_BASENAMES = {'data.pkl', 'response_audit.json', 'line_map.json',
                       'word_line_map.json', 'observation_diagnostics.csv', 'manuscript.docx'}
SECRET_NAME = re.compile(r'(password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)', re.I)
DRIVE_PATH = re.compile(r'^[A-Za-z]:[\\/]')
BASE_REQUIRED = ['README.md', 'LICENSE', 'LICENSE_CODE.txt', 'LICENSE_DATA.txt',
                 'CITATION.cff', 'citation.bib', 'figures/figure_manifest.json',
                 'metadata/zenodo_metadata.json']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def iter_release_files(root):
    """Ignore Git/environment internals, never private data or result folders."""
    paths = []
    for parent, dirs, names in os.walk(root, followlinks=False):
        kept = []
        for name in sorted(dirs):
            p = Path(parent) / name
            if name in EXCLUDED_DIRECTORIES:
                continue
            if p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()):
                paths.append(p)
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            if name == '.git' or name.endswith(('.pyc', '.pyo')):
                continue
            paths.append(Path(parent) / name)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def relative_file(root, name):
    if not isinstance(name, str) or DRIVE_PATH.match(name) or name.startswith(('/', '\\')):
        raise ValueError('Expected a safe repository-relative path: ' + str(name))
    p = root / name
    if not p.resolve().is_relative_to(root):
        raise ValueError('Path escapes the repository: ' + name)
    return p


def validate(root, verify_checksums=True):
    root = Path(root).resolve()
    errors, checked = [], []
    profile_path = root / 'metadata/release_profile.json'
    profile = None
    if profile_path.is_file():
        try:
            profile = json.loads(profile_path.read_text(encoding='utf-8-sig'))
            if not isinstance(profile, dict):
                raise ValueError('Profile must be an object')
        except (ValueError, UnicodeError) as exc:
            errors.append('Invalid release profile: ' + str(exc))
    required = list(BASE_REQUIRED)
    if profile is None:
        required += ['data/data_dictionary.csv', 'data/paper_traceability.csv']
    else:
        supplied = profile.get('required_files', [])
        if not isinstance(supplied, list):
            errors.append('release_profile.required_files must be a list')
        else:
            required += supplied
    if verify_checksums:
        required += ['SHA256SUMS.txt', 'FILE_MANIFEST.csv']
    for name in sorted(set(required)):
        if not verify_checksums and name in {'SHA256SUMS.txt', 'FILE_MANIFEST.csv'}:
            continue
        try:
            if not relative_file(root, name).is_file():
                errors.append('Missing required file: ' + name)
        except ValueError as exc:
            errors.append(str(exc))

    paths = iter_release_files(root)
    regular = []
    for p in paths:
        rel = p.relative_to(root).as_posix()
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            errors.append('Symlink/junction or non-regular release file: ' + rel)
            continue
        regular.append(p)
        if any(part.lower() in FORBIDDEN_DIRECTORIES for part in p.relative_to(root).parts[:-1]):
            errors.append('Excluded private/output directory present: ' + rel)
        if profile and profile.get('data_included') is False and p.relative_to(root).parts[0] == 'data':
            errors.append('This code-and-figures profile excludes data/: ' + rel)

    checksum = root / 'SHA256SUMS.txt'
    if verify_checksums and checksum.is_file():
        expected_names = set()
        for line in checksum.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            try:
                expected, name = line.split('  ', 1)
                if not re.fullmatch(r'[0-9a-f]{64}', expected):
                    raise ValueError('Invalid hash format for: ' + name)
                candidate = relative_file(root, name)
                if name in expected_names:
                    raise ValueError('Duplicate checksum entry: ' + name)
                expected_names.add(name)
                if not candidate.is_file() or candidate.is_symlink():
                    errors.append('Missing/unsafe checksummed file: ' + name)
                elif sha256(candidate) != expected:
                    errors.append('Checksum mismatch: ' + name)
                else:
                    checked.append(name)
            except ValueError as exc:
                errors.append('Invalid checksum entry: ' + str(exc))
        actual = {p.relative_to(root).as_posix() for p in regular if p.name != 'SHA256SUMS.txt'}
        if actual != expected_names:
            errors.append('Checksum coverage differs from release files: ' +
                          ', '.join(sorted(actual.symmetric_difference(expected_names))))

    core_hashes = profile.get('immutable_calibration_sha256', {}) if profile else {}
    core_checked = 0
    if not isinstance(core_hashes, dict):
        errors.append('release_profile.immutable_calibration_sha256 must be an object')
        core_hashes = {}
    for name, expected in core_hashes.items():
        try:
            candidate = relative_file(root, name)
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                errors.append('Invalid immutable calibration hash: ' + name)
            elif not candidate.is_file() or sha256(candidate) != expected:
                errors.append('Immutable calibration source differs: ' + name)
            else:
                core_checked += 1
        except ValueError as exc:
            errors.append(str(exc))

    code_files = csv_files = vector_files = 0
    excluded_vectors = set(profile.get('excluded_vectors', [])) if profile else {'figures/Figure_3.svg'}
    for p in regular:
        rel = p.relative_to(root).as_posix()
        license_text = ('third_party_licenses' in p.parts and
                        (p.name.upper().startswith('LICENSE') or p.name == 'COPYING'))
        if p.suffix.lower() not in ALLOWED_SUFFIXES and not license_text and p.name not in ALLOWED_BASENAMES:
            errors.append('Unapproved file type: ' + rel)
        if p.name.lower() in FORBIDDEN_BASENAMES:
            errors.append('Excluded file class present: ' + rel)
        if p.suffix.lower() == '.csv':
            csv_files += 1
            try:
                with p.open(encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    fields = {c.strip().lower() for c in (reader.fieldnames or [])}
                    bad = fields.intersection(FORBIDDEN_COLUMNS)
                    if bad:
                        errors.append('Disallowed observation/station columns in ' + rel + ': ' + ', '.join(sorted(bad)))
                    for i, row in enumerate(reader, 2):
                        if None in row or any(value is None for value in row.values()):
                            errors.append('Malformed CSV row in ' + rel + ' at line ' + str(i))
            except (csv.Error, UnicodeError) as exc:
                errors.append('Invalid CSV: ' + rel + ': ' + str(exc))
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
            if rel in excluded_vectors:
                errors.append('Profile excludes this observation-level vector: ' + rel)
            try:
                xml = ET.fromstring(p.read_bytes())
                for node in xml.iter():
                    if node.tag.rsplit('}', 1)[-1].lower() == 'script':
                        errors.append('Active SVG script present: ' + rel)
                    for k, v in node.attrib.items():
                        local = k.rsplit('}', 1)[-1].lower()
                        if local.startswith('on'):
                            errors.append('Active SVG event handler present: ' + rel)
                        if local in ('href', 'src') and (v.startswith(('file:', 'javascript:')) or DRIVE_PATH.match(v)):
                            errors.append('Private/active vector link: ' + rel)
            except ET.ParseError:
                errors.append('Invalid SVG XML: ' + rel)
    fig_pngs = sorted(p.name for p in (root / 'figures').glob('Figure_*.png'))
    expected_figures = profile.get('current_figures', []) if profile else (
        ['Figure_' + str(n) + '.png' for n in range(1, 7)] + ['Figure_S1.png', 'Figure_S2.png'])
    if not isinstance(expected_figures, list) or fig_pngs != sorted(expected_figures):
        errors.append('Final figure PNG set differs from the release profile')
    return {'status': 'PASS' if not errors else 'FAIL', 'checksummed_files': len(checked),
            'python_files_syntax_checked': code_files, 'csv_files_checked': csv_files,
            'svg_files_checked': vector_files, 'immutable_calibration_files_checked': core_checked,
            'profile': profile.get('profile') if profile else 'legacy_aggregate_release',
            'errors': errors,
            'scope': 'Integrity, source hashes, file/schema exclusions and syntax; no restricted-input numerical rerun.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json-output', type=Path, help='Optional report outside this repository/release')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.json_output and args.json_output.resolve().is_relative_to(root):
        parser.error('Write the validation report outside the repository to preserve checksum coverage.')
    report = validate(root)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
