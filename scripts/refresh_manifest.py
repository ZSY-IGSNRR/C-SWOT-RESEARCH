#!/usr/bin/env python3
"""Refresh checksums after DOI/citation metadata edits; refuse changed research files."""
from pathlib import Path
import argparse
import csv
import hashlib
import re

METADATA_EDITS = {'README.md', 'ZENODO_UPLOAD.md', 'CITATION.cff', 'citation.bib',
                  'metadata/zenodo_metadata.json', 'metadata/release_validation.json'}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    old = {}
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        h, name = line.split('  ', 1)
        if not re.fullmatch(r'[0-9a-f]{64}', h) or not (root / name).resolve().is_relative_to(root):
            parser.error('Malformed or unsafe original checksum list.')
        old[name] = h
    paths = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.name != 'SHA256SUMS.txt')
    current = {p.relative_to(root).as_posix() for p in paths}
    if current != set(old):
        parser.error('Added/removed files detected. Prepare and validate a new release; do not silently refresh this version.')
    changed = [name for name, h in old.items() if sha(root / name) != h]
    prohibited = set(changed) - METADATA_EDITS
    if prohibited:
        parser.error('Research/code/license files changed: ' + ', '.join(sorted(prohibited)) + '. Prepare a new release.')
    inventory = root / 'FILE_MANIFEST.csv'
    with inventory.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row['file'] in ['FILE_MANIFEST.csv', 'SHA256SUMS.txt']:
            continue
        p = root / row['file']
        row['bytes'] = str(p.stat().st_size)
        row['sha256'] = sha(p)
    with inventory.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'bytes', 'sha256', 'category'])
        w.writeheader()
        w.writerows(rows)
    (root / 'SHA256SUMS.txt').write_text(''.join(sha(p) + '  ' + p.relative_to(root).as_posix() + '\n' for p in paths), encoding='utf-8')
    print('Refreshed checksum inventory for metadata edits:', ', '.join(changed) if changed else 'none')
    print('Run python scripts/check_release.py before repackaging. This command performs no upload.')

if __name__ == '__main__':
    main()
