"""Regression tests for code-only repositories and protected manifest refreshes."""
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_release
import refresh_manifest


class ReleaseToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in check_release.BASE_REQUIRED:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{}\n' if p.suffix == '.json' else 'Fixture\n', encoding='utf-8', newline='\n')
        self.profile = {'version': '1.1.0', 'profile': 'code_and_figures', 'data_included': False,
                        'required_files': [], 'current_figures': [], 'immutable_calibration_sha256': {}}
        self.save_profile()

    def save_profile(self):
        (self.root / 'metadata/release_profile.json').write_text(json.dumps(self.profile) + '\n', encoding='utf-8', newline='\n')

    def refresh(self, new=False):
        args = ['--root', str(self.root)] + (['--new-version', '1.1.0'] if new else [])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return refresh_manifest.main(args)

    def test_code_profile_without_data_and_git_files_excluded(self):
        (self.root / '.git/objects').mkdir(parents=True)
        (self.root / '.git/objects/internal').write_bytes(b'not a research file')
        (self.root / '.venv').mkdir()
        (self.root / '.venv/secret.bin').write_bytes(b'cache')
        (self.root / '.gitignore').write_text('private/\n', encoding='utf-8', newline='\n')
        self.refresh(new=True)
        report = check_release.validate(self.root)
        self.assertEqual(report['status'], 'PASS', report['errors'])
        names = (self.root / 'SHA256SUMS.txt').read_text(encoding='utf-8')
        self.assertIn('  .gitignore\n', names)
        self.assertNotIn('.git/objects', names)
        self.assertNotIn('.venv/', names)

    def test_uppercase_observation_columns_cannot_enter_release(self):
        (self.root / 'metadata/derived.csv').write_text('Fold_ID,Q_Observed\n0,12\n', encoding='utf-8', newline='\n')
        report = check_release.validate(self.root, verify_checksums=False)
        self.assertTrue(any('Disallowed observation/station columns' in e for e in report['errors']))
        with self.assertRaises(SystemExit):
            self.refresh(new=True)

    def test_private_results_are_checked_not_silently_ignored(self):
        (self.root / 'private').mkdir()
        (self.root / 'private/result.json').write_text('{}\n', encoding='utf-8', newline='\n')
        report = check_release.validate(self.root, verify_checksums=False)
        self.assertTrue(any('Excluded private/output directory' in e for e in report['errors']))

    def test_new_data_directory_does_not_restore_deleted_release_data(self):
        (self.root / 'data').mkdir()
        (self.root / 'data/aggregate.csv').write_text('fold,NSE\n0,0.4\n', encoding='utf-8', newline='\n')
        report = check_release.validate(self.root, verify_checksums=False)
        self.assertTrue(any('profile excludes data/' in e for e in report['errors']))

    def test_added_research_file_requires_explicit_new_version(self):
        self.refresh(new=True)
        (self.root / 'model.py').write_text('RESULT = 1\n', encoding='utf-8', newline='\n')
        with self.assertRaises(SystemExit):
            self.refresh()
        self.refresh(new=True)
        self.assertEqual(check_release.validate(self.root)['status'], 'PASS')

    def test_removed_research_file_requires_explicit_new_version(self):
        (self.root / 'model.py').write_text('RESULT = 1\n', encoding='utf-8', newline='\n')
        self.refresh(new=True)
        (self.root / 'model.py').unlink()
        with self.assertRaises(SystemExit):
            self.refresh()

    def test_changed_research_file_is_not_metadata_only(self):
        p = self.root / 'model.py'
        p.write_text('RESULT = 1\n', encoding='utf-8', newline='\n')
        self.refresh(new=True)
        p.write_text('RESULT = 2\n', encoding='utf-8', newline='\n')
        with self.assertRaises(SystemExit):
            self.refresh()

    def test_metadata_only_refresh_and_version_match(self):
        self.refresh(new=True)
        (self.root / 'README.md').write_text('Updated metadata\n', encoding='utf-8', newline='\n')
        self.refresh()
        self.assertEqual(check_release.validate(self.root)['status'], 'PASS')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            refresh_manifest.main(['--root', str(self.root), '--new-version', '2.0.0'])

    def test_core_hash_and_current_figure_grain(self):
        p = self.root / 'model.py'
        p.write_text('RESULT = 1\n', encoding='utf-8', newline='\n')
        self.profile['immutable_calibration_sha256'] = {'model.py': check_release.sha256(p)}
        self.save_profile()
        (self.root / 'figures/Figure_3.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n', encoding='utf-8', newline='\n')
        self.assertEqual(check_release.validate(self.root, verify_checksums=False)['status'], 'PASS')
        p.write_text('RESULT = 2\n', encoding='utf-8', newline='\n')
        report = check_release.validate(self.root, verify_checksums=False)
        self.assertTrue(any('Immutable calibration source differs' in e for e in report['errors']))

    def test_current_figures_ignore_legacy_subdirectory_and_allow_photo(self):
        self.profile['current_figures'] = ['Figure_1.png']
        self.save_profile()
        (self.root / 'figures/Figure_1.png').write_bytes(b'PNG fixture')
        legacy = self.root / 'figures/legacy_v1_0_0'
        legacy.mkdir()
        (legacy / 'Figure_S1.png').write_bytes(b'Historical PNG fixture')
        (self.root / 'figures/field_photo.jpg').write_bytes(b'JPEG fixture')
        report = check_release.validate(self.root, verify_checksums=False)
        self.assertEqual(report['status'], 'PASS', report['errors'])


if __name__ == '__main__':
    unittest.main()
