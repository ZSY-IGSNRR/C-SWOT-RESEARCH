# C-SWOT research code

Version 1.1.0, updated 21 October 2026. This repository follows the current v15 manuscript, *Estimating Streamflow Across Major Chinese Rivers Using SWOT Satellite Data and Sparse Ground Observations*. The current analysis uses WSE-stratified fivefold cross-validation and seven main figures. The earlier ten-assignment and temporal workflows remain available as legacy source.

The seven hydraulic model/QC modules in `code/calibration` are unchanged. New code organizes fivefold assignments, reuses the original calibration settings, summarizes independent out-of-fold predictions, compares unconstrained SoS products, and examines station skill and error influence. The current evaluation covers 422 eligible gauges and 7,615 unique observations, with 2,110 full-model calibration fits. These are manuscript alignment facts; the underlying observations are not included here.

## Contents

| Path | Purpose |
|---|---|
| `code/fivefold/` | Current fivefold orchestration, summaries, product comparisons, diagnostics, station analysis, and Figures 1–7 |
| `code/calibration/` | Unchanged original hydraulic model and QC implementation |
| `code/configuration/fivefold.example.json` | Editable example with explicit external input paths and unchanged scientific settings |
| `code/upstream/` | Satellite download, station/SWORD audit, quality control and node/reach fusion |
| `code/postprocessing/`, `code/diagnostics/`, `code/figures/` | Earlier workflow code retained for reproducing the original analysis |
| `figures/` | Current final figures, captions and image provenance; old figures are in `legacy_v1_0_0` |
| `scripts/` | Standard-library repository checks, file inventories and optional export of externally supplied aggregates |
| `metadata/` | Source products, method references, manuscript alignment and validation records |

The remote repository removed its previous `data/` directory. This update preserves that state. It contains no gauge time series, station-coordinate/attribute tables, observation-level measured or predicted discharge, model checkpoints, private caches, manuscript DOCX files or reviewer correspondence. The separate local research archive contains those working materials under their existing access conditions.

## Start here

```text
python scripts/check_release.py
python code/run.py --help
python code/run.py fivefold --help
python code/run.py fivefold run --help
python code/run.py fivefold-figures --help
```

Help and repository checks use the standard library and do not read gauge data. For numerical work, install `code/environment/requirements_fivefold.txt` in a Python 3.12 environment. Copy the example configuration outside the repository and set its authorized input paths before running:

```text
python code/run.py fivefold run --root ../swot_local_run --config ../local_config.json
python code/run.py fivefold summarize --root ../swot_local_run
python code/run.py fivefold error-diagnostics --root ../swot_local_run
```

Product comparison additionally needs the audited SoS inputs. Read `code/fivefold/README.md` for every command, optional old-design comparisons, diagnostics and drawing arguments. Keep generated results outside the repository. Original configuration and core-model hashes remain documented; source adaptations change path handling and execution entry points rather than the scientific equations, objective, random seeds or optimizer settings.

## Interpretation

Within five rank-based WSE strata, observations are assigned to five folds. Each retained observation is validated once, using parameters and a width relation fitted on the other four folds. Each gauge requires at least seven QC-retained records. Approximately 80% are used for calibration and 20% for validation; actual fractions reflect integer counts. Calibration WSE defines the below, within and above domains.

Fold summaries report the median and half the interquartile range across five scores. This describes split variation, not a confidence interval. Network out-of-fold scores concatenate mutually exclusive validation predictions; gauge-level scores use each gauge's out-of-fold observations. These aggregations answer different questions and must be distinguished.

The WSE design tests held-out water-level states across the sampled record. Dates are mixed; this is not a chronological forecast experiment. Centered complete-record satellite QC and datum treatment are retrospective. Error screening uses measured validation discharge and is a retrospective influence diagnostic, not an operational filter. The older `calibration`, `postprocess`, `statistics-figures` and `error-diagnostics` top-level commands describe the legacy ten-assignment workflow; use the `fivefold` subcommands for current results.

## Figures and source products

The current seven figure PNGs match the 21 October manuscript exactly. SVG/PDF companions and captions follow the current numbering. Figure 1 credits Tim Merrick / USGS for the field photograph; conceptual satellite/river artwork and effective cross sections are illustrative. Asset notes and hashes are in `figures/`.

The satellite inputs are SWOT RiverSP Version D, SoS Version 3, SWORD v16/v17b and Hydrocron 1.6.4. Original identifiers and DOIs remain in `metadata/source_products.csv`. No upstream satellite archive is redistributed. The longer satellite archive window does not imply an equally long discharge-validation period.

## Verification and citation

`scripts/check_release.py` checks hashes, schemas, syntax, figure inventory and unchanged core files. Tests cover fivefold allocation and the portable interfaces; `metadata/release_validation.json` records what was actually run. This update did not rerun the national optimization. Full numerical reproduction requires externally supplied observations and compatible processed inputs.

Use `CITATION.cff` or `citation.bib` for this code version and cite the manuscript and relevant source products. No DOI is fabricated. The Zenodo metadata is a preparation template and no deposit is performed by any repository check. Study-authored code is MIT licensed; author-created documentation and figure elements are CC BY 4.0, with attributed third-party material retaining its own terms.
