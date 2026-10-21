"""Portable entry-point forwarding must preserve help, arguments and exit codes."""
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import importlib.util
import io
import sys
import unittest

ENTRY = Path(__file__).resolve().parents[2] / 'code/run.py'
spec = importlib.util.spec_from_file_location('swot_entrypoint_under_test', ENTRY)
entry = importlib.util.module_from_spec(spec)
before_import = set(sys.modules)
spec.loader.exec_module(entry)
new_imports = set(sys.modules) - before_import


class CliForwardingTests(unittest.TestCase):
    def test_loading_entrypoint_needs_no_scientific_library(self):
        self.assertFalse({'numpy', 'pandas', 'scipy', 'matplotlib'}.intersection(new_imports))

    def assert_forwarded(self, command, child, extra, returncode=0):
        with patch.object(sys, 'argv', [str(ENTRY), command, *extra]), \
             patch.object(entry.subprocess, 'call', return_value=returncode) as call:
            actual = entry.main()
        self.assertEqual(actual, returncode)
        call.assert_called_once()
        self.assertEqual(call.call_args.args[0], [sys.executable, str(ENTRY.parent / child), *extra])
        env = call.call_args.kwargs['env']
        self.assertEqual(env['PYTHONIOENCODING'], 'utf-8')
        self.assertEqual(env['PYTHONUTF8'], '1')

    def test_fivefold_help_reaches_fivefold_parser(self):
        self.assert_forwarded('fivefold', 'fivefold/cli.py', ['--help'])

    def test_figure_help_reaches_figure_parser(self):
        self.assert_forwarded('fivefold-figures', 'fivefold/build_figures.py', ['--help'])

    def test_child_error_status_and_arguments_are_preserved(self):
        self.assert_forwarded('fivefold', 'fivefold/cli.py',
                              ['run', '--root', '/external path/run', '--config', '/external path/config.json'],
                              returncode=23)

    def test_legacy_parser_help_still_works_without_a_child_process(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', [str(ENTRY), 'calibration', '--help']), \
             patch.object(entry.subprocess, 'call') as call, redirect_stdout(output), \
             self.assertRaises(SystemExit) as caught:
            entry.main()
        self.assertEqual(caught.exception.code, 0)
        call.assert_not_called()
        self.assertIn('--data-dir', output.getvalue())
        self.assertIn('--metadata-file', output.getvalue())


if __name__ == '__main__':
    unittest.main()
