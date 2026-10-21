#!/usr/bin/env python3
"""Refresh metadata checksums or explicitly prepare a validated new code version."""
from pathlib import Path
import argparse
import csv
import json
import re

from check_release import iter_release_files, relative_file, sha256, validate

METADATA_EDITS = {'README.md', 'ZENODO_UPLOAD.md', 'CITATION.cff', 'citation.bib',
                  'metadata/zenodo_metadata.json', 'metadata/release_validation.json'}


def read_old_checksums(root, parser):
    old = {}
    path = root / 'SHA256SUMS.txt'
    if not path.is_file():
        parser.error('No existing checksum list. Use --new-version with an explicit release profile.')
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        try:
            h, name = line.split('  ', 1)
            relative_file(root, name)
            if not re.fullmatch(r'[0-9a-f]{64}', h) or name in old:
                raise ValueError('Invalid or duplicate checksum')
        except ValueError:
            parser.error('Malformed or unsafe original checksum list.')
        old[name] = h
    return old


def write_manifests(root):
    payload = [p for p in iter_release_files(root) if p.name not in {'FILE_MANIFEST.csv', 'SHA256SUMS.txt'}]
    inventory = root / 'FILE_MANIFEST.csv'
    with inventory.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'bytes', 'sha256', 'category'], lineterminator='\n')
        w.writeheader()
        for p in payload:
            name = p.relative_to(root).as_posix()
            w.writerow({'file': name, 'bytes': p.stat().st_size, 'sha256': sha256(p),
                        'category': name.split('/')[0] if '/' in name else 'metadata'})
    paths = [p for p in iter_release_files(root) if p.name != 'SHA256SUMS.txt']
    (root / 'SHA256SUMS.txt').write_text(''.join(sha256(p) + '  ' + p.relative_to(root).as_posix() + '\n' for p in paths),
                                        encoding='utf-8', newline='\n')
    return len(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--new-version', help='Explicitly rebuild research/code manifests; must match release_profile.version')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.new_version:
        if not re.fullmatch(r'\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?', args.new_version):
            parser.error('--new-version must be an explicit semantic version.')
        try:
            profile = json.loads((root / 'metadata/release_profile.json').read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            parser.error('A valid metadata/release_profile.json is required for a new code version.')
        if profile.get('version') != args.new_version:
            parser.error('--new-version must exactly match metadata/release_profile.json version.')
    else:
        old = read_old_checksums(root, parser)
        paths = [p for p in iter_release_files(root) if p.name != 'SHA256SUMS.txt']
        current = {p.relative_to(root).as_posix() for p in paths}
        if current != set(old):
            parser.error('Added/removed files detected. Prepare a new version with --new-version; do not silently refresh this version.')
        changed = [name for name, h in old.items() if sha256(root / name) != h]
        prohibited = set(changed) - METADATA_EDITS
        if prohibited:
            parser.error('Research/code/license files changed: ' + ', '.join(sorted(prohibited)) + '. Prepare a new version with --new-version.')
    report = validate(root, verify_checksums=False)
    if report['status'] != 'PASS':
        parser.error('Pre-manifest structural validation failed:\n  ' + '\n  '.join(report['errors']))
    count = write_manifests(root)
    print('Rebuilt checksum inventory for version ' + args.new_version if args.new_version else 'Refreshed metadata-only checksum inventory.')
    print('Checksummed files:', count)
    print('Run python scripts/check_release.py before packaging. This command performs no upload.')
    return 0


if __name__ == '__main__':
    main()
