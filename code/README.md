# Source code for manuscript v15

This directory contains the station calibration, validation, postprocessing, and figure-generation code used for the v15 manuscript. The original calibration sources are retained byte for byte. A portable entry point supplies explicit paths, checks external inputs, and separates output directories. Gauge records and observation-level tables containing measured discharge are excluded from this release. The public aggregates and finished figures support inspection of the reported results; this code release does not provide every input needed for a full numerical reproduction.

## Directory contents

- `calibration/`: original model, global QC, WSE-stratified validation, expanding-window forward temporal validation, component tests, and checks. The archive's model sources match the current calculation sources. Old plotting utilities are excluded; the `figures/` entry points generate the v15 figures.
- `upstream/`: station/SWORD inventory, Hydrocron download, satellite QC, node–reach matching, gauge fusion, roughness assignment, datum alignment, network audit, and spatially audited SoS download/comparison tools.
- `postprocessing/`: screening diagnostics, scale/regional summaries, empirical and matched SoS comparisons. `finish_postprocess.py` is an original recovery utility for a recorded SoS-stage failure, rather than an extra step required after a successful run.
- `diagnostics/`: extreme-residual and controlled validation-width analyses and Figure 5.
- `figures/`: private cache preparation, WSE-domain/distance aggregation, Figures 2–4 and 6, supplementary Figures S1–S2, and the final method Figure 1.
- `configuration/`: saved scientific settings with paths relocated to editable examples. Settings retain ten WSE assignments, up to three temporal blocks, three optimization starts, and 2,000 bootstrap resamples.
- `environment/`: observed dependency versions, environment notes, and third-party license notices.
- `SOURCE_MANIFEST.json`: source provenance, original/published hashes, and the changes made for portability and v15 figure numbering.

## Running the code

Start in the root of the unpacked release. Use `python code/run.py --help` and `python code/run.py COMMAND --help` to inspect the supported public entry points. Relative paths in upstream configuration are relative to the working directory. Every calculation requiring gauge data checks that the necessary external files or directories exist before invoking scientific code. Supply authorized local data and use new output directories.

The immutable calibration files retain their historical defaults for provenance. The public entry point always overrides those paths explicitly. Run the model through `code/run.py`, rather than relying on those historical defaults. The original model prints a legacy plotting suggestion at completion; use the v15 figure commands below in its place.

Examples, with locally supplied input paths:

```text
python code/run.py calibration --data-dir inputs/station_overpasses_datum_adjusted --metadata-file inputs/Find_N.csv --boundary inputs/boundaries/Class2_hydrounits.shp --output outputs/results
python code/run.py postprocess --results-dir outputs/results --metadata inputs/station_metadata.csv --station-audit inputs/station_spatial_audit.csv --sos-raw-dir inputs/sos_v16_mapped_raw --boundary inputs/boundaries/Class2_hydrounits.shp --output outputs/postprocess_results
python code/run.py figure-cache --data-root outputs --output private_cache/base
python code/run.py wse-domain --data-root outputs --base-cache private_cache/base/cache/data.pkl --output private_cache/domain
python code/run.py statistics-figures --domain-cache private_cache/domain/data.pkl --boundary inputs/boundaries/Class2_hydrounits.shp --output private_figures/statistics
python code/run.py error-diagnostics --data-root outputs --output private_figures/errors
python code/run.py method-figure --output private_figures/method
```

Figure 1 uses `figures/assets/Fig1a_background.png` in the release root by default, and `--background` can supply another file. This is a generated conceptual illustration, with appearance informed by an official satellite illustration; it is not an official NASA/CNES image. Its description and source are in the asset notes. The method entry point writes `Figure_1.png` and `Figure_1.svg` to its output directory. The statistical entry point adds manuscript-named copies `Figure_2`, `Figure_3`, `Figure_4`, and `Figure_6` in `main/`, and `Figure_S1` and `Figure_S2` in `supplement/`, as PNG/SVG; the PDF copies are in `pdf/`. The extreme-error entry point adds `Figure_5.png`, `Figure_5.svg`, and `Figure_5.pdf` in `width_analysis/figures/`. Original statistical/diagnostic renderer filenames are retained alongside these copies for provenance.

For upstream preparation, edit `code/configuration/upstream.example.json`, save a local configuration, and run:

```text
python code/run.py upstream --config local_config.json --steps inventory,download,qc,attach,fuse,roughness,shift,audit
```

The download step accesses remote services when invoked. No scientific data downloads or dependency installations were performed while packaging this release. The upstream source retains an optional separately stored SoS Version 2.0 layer from the archive workflow. That optional layer is distinct from the Version 3 product benchmark in the manuscript. The benchmark requires the corresponding externally supplied SoS records and audited crosswalk; changing product versions requires a separate audit.

## External inputs and scientific scope

The model needs fused station-overpass records with SWOT WSE, width, positive slope, dates, flags, and matched daily gauge discharge, plus station metadata. Upstream preparation needs station locations, a SWORD node index, authorized gauge records, and roughness inputs for the steps using them. Postprocessing and statistical figures need the saved result tables and spatial/product metadata. Hydrological boundary shapefiles and their companion files are external inputs. The run configurations identify the computational settings, and the manuscript identifies the public satellite products by DOI.

Private figure caches contain observation-level material and are created locally when the figure chain is run. They are not included in this release and should not be added to a public archive. Fixed cohort assertions in the figure and diagnostic scripts refer to the saved manuscript evaluation; other data versions require an explicit cohort audit.

WSE-stratified assignments test held-out hydraulic states, while forward temporal blocks test later observations using expanding earlier calibration records. Centered complete-record QC and datum alignment make both evaluations retrospective. The width comparison changes the terminal trapezoid width used for area evaluation and the Manning width term while holding calibration anchors and fitted parameters fixed. Residual screening uses measured validation discharge and is a sensitivity diagnostic, with no model refitting or claim of operational error removal. These boundaries remain unchanged in this public code.

## Validation and licensing

Packaging checks cover source syntax, public CLI help, external-input checks, and hashes of the immutable calibration files. Full calibration, postprocessing, and statistical figure regeneration were not rerun for this release. Consult `environment/README_EN.md` for the difference between observed package versions and historical-run provenance.

The release-root MIT license applies to author-created code. Dependencies are not vendored and retain their own licenses; notices and metadata are in `environment/`. The release-root data/documentation/figure license does not replace those dependency licenses or third-party asset terms.
