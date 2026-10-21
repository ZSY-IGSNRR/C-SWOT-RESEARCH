# Standard WSE-stratified fivefold workflow

This is the current calculation workflow for the 21 October 2026 manuscript revision. It replaces the earlier ten repeated WSE allocations as the main evaluation. It uses the same seven scientific model/QC modules in `code/calibration/`; their pinned hashes are in `frozen_v15/SOURCE_MANIFEST.json`. No scientific core, objective weight, parameter bound, optimizer setting or seed was changed for portability.

## Design and scope

QC-pass observations are ranked by WSE into five similarly sized strata. Within each stratum they are shuffled reproducibly and assigned cyclically to five validation folds with a carried offset. Each observation is validated once, using the other four folds for calibration, approximately 80%/20%. At least seven QC-pass records are required so every fold has at least five calibration and one validation record. QC-failed records receive separate assignments for the no-global-Hampel sensitivity. Width curves use calibration data only.

The saved manuscript cohort has 422 eligible gauges, 7,615 primary OOF observations and 2,110 full-model fits. These describe that completed dataset, not mandatory counts for the general `run` and `summarize` commands. The `audit-results`, `check-equivalence`, `drivers`, `build-readout`, `build-station-readout`, `build-station-inline` and `finalize` helpers retain checks or narrative for the complete manuscript cohort. They are intended for verifying/recreating that cohort and can reject, or be unsuitable for, a different dataset. Pilot runs are incomplete sensitivity/smoke outputs, not manuscript reproduction.

National NSE/RMSE are computed separately in each validation fold, then summarized as median and IQR/2 across folds. Single national OOF scores and equal-gauge OOF distributions are reported separately. Dates are mixed within folds; this is validation of held-out water-level states, with retrospective complete-record QC, rather than forward temporal validation. The machine-readable design is `fivefold_design.json`.

## Install and configure

Use Python 3.12 and `code/environment/requirements_fivefold.txt` for the recorded numerical environment. Dependencies are not vendored. Read the repository environment notes for plotting and upstream workflows. Help and missing-input checks use the standard library and work without NumPy/SciPy.

Copy `code/configuration/fivefold.example.json` to a local file and set `data_dir`, `metadata_file` and `basin_shapefile` to authorized inputs. Relative paths resolve against the terminal's working directory. The scientific settings are pinned to the original V15 values; changing them is rejected by this manuscript entry point. Workers can be changed without changing scientific settings. `output_dir` in the example is overridden by the required external `--root`.

The example retains legacy `repeat_count: 10` and temporal flags for provenance of the unchanged scientific model configuration. They do not organize the current evaluation: the new runner implements exactly five disjoint folds, no full-data reconstruction and no temporal experiment. Changing the old repeat count to five is not a substitute for this runner.

All generated data, checkpoints, summaries, caches and readouts are written outside the repository. Every command requires an explicit `--root`. The runner hashes its authorized inputs and records an effective configuration in that work root. Existing configurations are checked before any additional outputs are created; use a new root if settings or input paths differ. Restricted gauge observations, predictions, station-level diagnostic tables and checkpoints are not distributed here.

## Run from the repository root

The main CLI forwards everything after `fivefold` to `code/fivefold/cli.py`:

```text
python code/run.py fivefold --help
python code/run.py fivefold run --root ../private_fivefold --config ../local_fivefold.json --workers 20
python code/run.py fivefold summarize --root ../private_fivefold
python code/run.py fivefold compare-products --root ../private_fivefold --station-audit ../inputs/station_spatial_audit.csv --sos-raw-dir ../inputs/sos_v16_mapped_raw
python code/run.py fivefold error-diagnostics --root ../private_fivefold
python code/run.py fivefold station-analysis --root ../private_fivefold
python code/run.py fivefold plot --root ../private_fivefold
```

`run` can resume its own existing checkpoints after the effective configuration matches. `--methods`, `--pilot-station` and `--pilot-fold` limit an explicitly requested incomplete run. `--aggregate-only` collects existing checkpoints without fitting. No full model calculation was rerun while integrating this repository update.

Old-ten-split comparisons are opt-in. No command silently reads an earlier local drive or sibling experiment:

```text
python code/run.py fivefold summarize --root ../private_fivefold --legacy-predictions ../old_results/tables/validation_predictions.csv --legacy-controls-root ../old_controls
python code/run.py fivefold compare-products --root ../private_fivefold --station-audit ../inputs/station_spatial_audit.csv --sos-raw-dir ../inputs/sos_v16_mapped_raw --legacy-native-dir ../old_native_products
python code/run.py fivefold audit-assignments --root ../private_fivefold --config ../local_fivefold.json
```

Optional `--legacy-qc-summary` and `--legacy-qc-flags` add comparison with the prior QC audit; supply both or neither. `summarize --legacy-local-root` can locate an explicitly supplied old local summary's QA source path. `--legacy-controls-root` is used only with an explicitly requested old-prediction comparison. Official SoS records and the audited station/SWORD crosswalk are always explicit inputs.

## Other commands and source tracking

`check-summary` runs small in-memory metric-definition checks. `inspect` independently recalculates primary scores from existing predictions. `drivers` analyzes gauge-level skill associations without refitting. `audit-results` verifies the complete saved manuscript cohort. `freeze-sources` records input hashes and checks the seven unchanged core files. Readout helpers write notes and bounded chart assets into the external root; they preserve the saved-cohort interpretation and are not generic reporting templates. `review-documents --documents ...` is an optional local editorial helper; manuscript/reviewer files are not included in this code repository.

The `workflow/` directory retains all 19 source workflow/helper scripts from the completed fivefold experiment. `frozen_v15/SOURCE_MANIFEST.json` lists original and published hashes and the path adaptations. Computation functions are retained; adaptations supply external roots/configuration, make historical inputs optional, remove app-specific output paths, and verify model hashes against the repository. `with_dependencies.py` is a compatibility helper using the active Python environment, not a bundled environment installer. Use the documented CLI for input checking rather than running these working scripts directly.

This workflow's `plot` command creates diagnostic panels from local score tables. The separate manuscript-figure entry point described in the root code README generates the final seven paper figures, including the supplied national boundary and inset. Older `statistics-figures` and `method-figure` commands document the earlier figure design.

## Checks

```text
python -B code/fivefold/tests/test_fivefold.py -v
python -B code/run.py fivefold check-summary --root ../synthetic_summary_check
```

The tests use synthetic data to verify one validation assignment per observation, balanced WSE strata/folds, training/validation complements, primary-fold independence from QC-failed records, reproducibility, input/output boundaries and unchanged scientific core hashes. Pure fold functions and the original seed function are executed from their actual ASTs, avoiding an unrelated optimizer import; no fitting, provider observations or serialized model objects are involved. The CLI tests include standard-library-only help/missing-input checks and rejection of mismatched resume configurations before any output writes.
