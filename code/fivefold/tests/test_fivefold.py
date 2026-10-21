"""Synthetic fold invariants and portable-CLI boundaries; no gauge data or fitting."""
from pathlib import Path
import hashlib
import ast
import __future__
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1]
REPOSITORY = SOURCE.parents[1]
CLI = SOURCE / "cli.py"


class PortableCli(unittest.TestCase):
    def call(self, *args):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        return subprocess.run([sys.executable, "-S", "-B", str(CLI), *map(str, args)],
                              text=True, capture_output=True, env=env, encoding="utf-8")

    def test_help_works_without_scientific_packages(self):
        result = self.call("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("audit-assignments", result.stdout)
        for command in ["run", "summarize", "compare-products", "audit-assignments", "plot"]:
            with self.subTest(command=command):
                result = self.call(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_private_output_inside_repository_is_rejected(self):
        output = REPOSITORY / "not-created-by-fivefold-test"
        self.assertFalse(output.exists())
        result = self.call("check-summary", "--root", output)
        self.assertEqual(result.returncode, 2)
        self.assertIn("outside the repository", result.stderr)
        self.assertFalse(output.exists())

    def test_missing_inputs_fail_before_scientific_imports_or_writes(self):
        with tempfile.TemporaryDirectory(prefix="cswot-fivefold-cli-") as temporary:
            base = Path(temporary)
            config = base / "settings.json"
            config.write_text(json.dumps({"data_dir": str(base / "missing-observations"),
                                          "metadata_file": str(base / "missing-metadata.csv"),
                                          "basin_shapefile": str(base / "missing-boundary.shp")}), encoding="utf-8")
            output = base / "not-created"
            result = self.call("run", "--root", output, "--config", config)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Missing external inputs", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse(output.exists())

    def test_scientific_settings_are_pinned(self):
        with tempfile.TemporaryDirectory(prefix="cswot-fivefold-settings-") as temporary:
            config = Path(temporary) / "changed.json"
            config.write_text('{"seed": 99}', encoding="utf-8")
            output = Path(temporary) / "not-created"
            result = self.call("run", "--root", output, "--config", config)
            self.assertEqual(result.returncode, 2)
            self.assertIn("pinned V15 scientific settings", result.stderr)
            self.assertFalse(output.exists())

    def test_mismatched_existing_work_root_is_rejected_without_writes(self):
        with tempfile.TemporaryDirectory(prefix="cswot-fivefold-resume-") as temporary:
            base = Path(temporary)
            data = base / "inputs"
            data.mkdir()
            (data / "synthetic.csv").write_text("synthetic input placeholder\n", encoding="utf-8")
            metadata = base / "metadata.csv"
            metadata.write_text("synthetic metadata placeholder\n", encoding="utf-8")
            boundary = base / "boundary.shp"
            boundary.write_bytes(b"synthetic boundary placeholder")
            config = base / "settings.json"
            config.write_text(json.dumps({"data_dir": str(data), "metadata_file": str(metadata),
                                          "basin_shapefile": str(boundary)}), encoding="utf-8")
            output = base / "existing-run"
            stored = output / "frozen_v15/configuration/calibration_v15.json"
            stored.parent.mkdir(parents=True)
            previous = json.loads((SOURCE / "frozen_v15/configuration/calibration_v15.json").read_text(encoding="utf-8"))
            previous["seed"] = 99
            stored.write_text(json.dumps(previous), encoding="utf-8")
            before = {str(p.relative_to(output)): p.read_bytes() if p.is_file() else None for p in output.rglob("*")}
            result = self.call("run", "--root", output, "--config", config)
            self.assertEqual(result.returncode, 2)
            self.assertIn("different configuration", result.stderr)
            after = {str(p.relative_to(output)): p.read_bytes() if p.is_file() else None for p in output.rglob("*")}
            self.assertEqual(before, after)


class ScientificCoreIdentity(unittest.TestCase):
    def test_seven_core_files_unchanged(self):
        manifest = json.loads((SOURCE / "frozen_v15/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["scientific_core_sha256"]), 7)
        for name, expected in manifest["scientific_core_sha256"].items():
            with self.subTest(file=name):
                actual = hashlib.sha256((SOURCE.parent / "calibration" / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


class FoldInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:
            raise unittest.SkipTest("Install requirements_fivefold.txt for numerical checks: " + str(exc))
        cls.np, cls.pd = np, pd
        cls.temporary = tempfile.TemporaryDirectory(prefix="cswot-fivefold-synthetic-")
        # Execute the actual pure allocation functions, without importing the
        # unrelated hydraulic optimizer and its SciPy dependency.
        core_path = SOURCE.parent / "calibration/swot_n_wse0_width0_nonnegative_cv.py"
        core_tree = ast.parse(core_path.read_text(encoding="utf-8-sig"))
        seed_node = next(n for n in core_tree.body if isinstance(n, ast.FunctionDef) and n.name == "stable_seed")
        seed_default = next(ast.literal_eval(n.value) for n in core_tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "RANDOM_SEED" for t in n.targets))
        ns = {"hashlib": hashlib, "RANDOM_SEED": seed_default, "np": np, "pd": pd,
              "CFG": SimpleNamespace(seed=20260723), "SCHEME": "wse_stratified_5fold"}
        exec(compile(ast.Module(body=[seed_node], type_ignores=[]), str(core_path), "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        ns["v15"] = SimpleNamespace(pipeline=SimpleNamespace(stable_seed=ns["stable_seed"]))
        path = SOURCE / "workflow/run_fivefold.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"assign_group", "split_station"}]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        cls.workflow = SimpleNamespace(split_station=ns["split_station"])

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "temporary"):
            return
        cls.temporary.cleanup()

    def frame(self, passed, failed=5):
        n = passed + failed
        return self.pd.DataFrame({"obs_id": [f"synthetic-{i:04d}" for i in range(n)],
                                 "wse_model": self.np.arange(n) // 2,
                                 "global_qc_pass": [True] * passed + [False] * failed})

    def test_fivefold_heldout_coverage_balance_and_training_complements(self):
        for count in [7, 8, 13, 22, 50, 97]:
            with self.subTest(qc_pass=count):
                source = self.frame(count)
                assigned = self.workflow.split_station(source, "synthetic-gauge")
                self.assertEqual(len(assigned), len(source))
                self.assertTrue(assigned.obs_id.is_unique)
                self.assertEqual(set(assigned.obs_id), set(source.obs_id))
                primary = assigned.loc[assigned.qc_group.eq("pass")]
                counts = primary.fold_id.value_counts().reindex(range(5), fill_value=0)
                self.assertGreaterEqual(counts.min(), 1)
                self.assertLessEqual(counts.max() - counts.min(), 1)
                self.assertGreaterEqual(count - counts.max(), 5)
                for _, stratum in assigned.groupby(["qc_group", "wse_stratum"]):
                    sizes = stratum.fold_id.value_counts().reindex(range(5), fill_value=0)
                    self.assertLessEqual(sizes.max() - sizes.min(), 1)
                primary_keys = set(primary.obs_id)
                for fold in range(5):
                    train = set(primary.loc[primary.fold_id.ne(fold), "obs_id"])
                    heldout = set(primary.loc[primary.fold_id.eq(fold), "obs_id"])
                    self.assertFalse(train & heldout)
                    self.assertEqual(train | heldout, primary_keys)

    def test_allocation_is_reproducible_and_qc_failures_do_not_change_primary_folds(self):
        base = self.frame(23)
        first = self.workflow.split_station(base, "synthetic-gauge")
        second = self.workflow.split_station(base.copy(), "synthetic-gauge")
        self.pd.testing.assert_frame_equal(first, second)
        clean = self.workflow.split_station(base.loc[base.global_qc_pass].copy(), "synthetic-gauge")
        self.pd.testing.assert_frame_equal(first.loc[first.qc_group.eq("pass")].reset_index(drop=True),
                                          clean.loc[clean.qc_group.eq("pass")].reset_index(drop=True),
                                          check_dtype=False)

    def test_fewer_than_seven_primary_observations_is_ineligible(self):
        self.assertIsNone(self.workflow.split_station(self.frame(6), "synthetic-gauge"))


if __name__ == "__main__":
    unittest.main()
