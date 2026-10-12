# SWOT China: public release accompanying manuscript v15

**Release version:** 1.0.0  
**Prepared:** 12 October 2026  
**Status:** locally prepared deposit package; it has not been uploaded or published. No release DOI has been assigned.

This release accompanies *Estimating Streamflow Across Major Chinese Rivers Using SWOT Satellite Data and Sparse Ground Observations*, clean manuscript **SZ-SWOT-15_language_polished_clean.docx**. The manuscript file is identified for version alignment and is not distributed in this package. This release freezes the public artifacts used for v15; the data release version is independent of the manuscript revision number.

## Scope and scientific context

The study evaluates a local adaptation of the Manning-based SWOT discharge framework for Chinese rivers with limited access to gauge records and consistent channel surveys. Adjustable effective geometry and a calibration WSE–width relation are evaluated against simpler hydraulic configurations, an empirical WSE–discharge curve, and existing unconstrained SoS products. WSE-stratified tests assess held-out hydraulic states; expanding-window forward temporal tests assess later observations. Complete-record satellite QC and datum treatment make these tests retrospective. Fitted parameters describe an effective discharge relation.

This package contains study-authored source code, network/group-level summary statistics, field definitions, provenance and paper traceability, and the final six main and two supplementary figures. Figures, table names, and statistic labels follow v15. Data and code from earlier local archives were selected and documented for this release; the archives themselves are not redistributed as nested ZIP files.

## Contents

| Location | Contents |
|---|---|
| `data/` | Public aggregate CSV/JSON results, English data dictionary, and figure/table/section traceability |
| `figures/` | Figure 1–6 and Figure S1–S2, current captions, and an image provenance manifest |
| `code/` | Selected model, QC, validation, postprocessing and figure-source routines with environment and input instructions |
| `scripts/` | Standard-library verification, aggregate table reconstruction, and metadata-only checksum refresh |
| `metadata/` | Source identifiers, release description and file provenance |
| `LICENSE*` | Software and data/documentation licensing and scope |
| `CITATION.cff`, `citation.bib` | Citation information without a fabricated DOI |
| `SHA256SUMS.txt` | Checksums for every release file other than the checksum file itself |
| `ZENODO_UPLOAD.md` | Local-to-Zenodo deposit instructions; it performs no upload |

## Public and restricted material

Ground discharge time series are restricted by the originating providers. Observation-level evaluation tables that contain measured discharge, matched observed/predicted discharge pairs, and associated caches are excluded. This release also excludes station-coordinate or station-attribute tables, private runtime environments, downloads, credentials, manuscript DOCX files, reviewer correspondence and document-preparation intermediates.

The statistical files supplied here contain pooled, network or group summaries. A summary can contain a median flow, error or fitted parameter at the network/group level; these values are distinct from individual gauge records. Each file's grain and exclusions are documented in `data/`. Published figures are retained as the figures used in v15. Figure 3 includes the manuscript's rendered observed-versus-estimated scatter panel; its native vector source and point-level numerical table are excluded. A rendered figure is not a substitute for a reusable observation-level dataset.

The public-input workflow reproduces released tables from the supplied aggregates and verifies the release. Re-running local calibration, validation, matching or observation-level diagnostics requires the excluded ground data and compatible processed inputs. The source code is provided to document those operations for researchers who already have permission to obtain the inputs. This package does not grant access to, or a redistribution license for, restricted provider data. Provider access requests must follow the originating providers' actual procedures; unrestricted access is not promised.

## Quick start: no restricted input required

Use Python 3.12 (tested public verification environment: Python 3.12.14 on Windows). Unzip the release, open a terminal in the release's top-level directory and run:

```text
python scripts/check_release.py
python scripts/rebuild_public_tables.py --output-dir ../SWOT_v15_reproduced
```

These commands use the Python standard library and do not download data, contact Zenodo, or require access to a gauge. The first checks file integrity, naming and public-data schema exclusions. The second rebuilds readable table/summary exports from the public aggregate files. It does not refit the model or recalculate metrics from excluded observations. Write outputs outside the unpacked release directory; changing the release files or adding files inside it invalidates its original checksum coverage.

For the scientific source workflows, read `code/README.md` and the environment specifications under `code/` before running them. Their dependency and input requirements are wider than the public quick-start commands. Historical environment requirements describe the available source environment; successful checks of public aggregates do not establish that the complete numerical model was rerun. No new calibration or validation run was performed while assembling this release.

## Reading the results

- **Full-record reconstruction** measures fit to calibration observations and is a calibration diagnostic.
- **WSE-stratified validation** uses ten assignments across five WSE quantile strata. Approximately 70% of model-complete observations enter calibration within each stratum before the precomputed QC flags are applied. The retained split can therefore differ from 70/30.
- **Forward temporal validation** uses earlier observations for initial calibration and predicts up to three later blocks, expanding/refitting the calibration record between blocks while keeping parameters and the width curve fixed within each block.
- **Within/below/above range** refers to validation WSE relative to the WSE limits retained in that particular calibration fit.
- **Pooled statistics** combine observations. WSE pooled results in Table 1 are medians of per-assignment summaries. Station summaries first aggregate repeated assignments at each station and then take the median across eligible stations. Temporal station metrics combine later blocks at each station. Eligible cohorts differ among rows.
- **Factor error** summarizes multiplicative error. Boundary distance is normalized by the positive calibration WSE span. Bootstrap intervals and station counts must be read together; the far-below-range tail has weak support.
- **Figure 5** uses a separate cohort of 6,152 distinct observations at 270 stations and median prediction per observation. These pooled statistics differ in aggregation from Table 1. Residual screening uses measured validation discharge and is a retrospective error-influence diagnostic. It is not an operational screening rule for locations where discharge is unknown.
- **Width diagnostic** changes validation width while retaining WSE, slope, calibration anchors and the four fitted parameters. Width changes both terminal-trapezoid area evaluation and the Manning width term; independent reference measurements would be required to partition intrinsic width and WSE observation errors.
- **Figure 6** compares matched observations with unconstrained SoS products. The local model uses additional gauge constraints. The comparison assesses the combined benefit of local information and implementation, not a controlled reproduction of every original Durand algorithm.
- **Empirical baseline** uses the same calibration discharge information and remains stronger overall, including after the reported residual screening. A large pooled improvement does not establish improved performance at every station.

Machine-readable source aggregates are retained where available. Values transcribed from the manuscript are explicitly marked as reported/rounded values; they are not presented as unrounded observation-level calculations. Blank uncertainty fields mean unavailable/not reported, not zero uncertainty. Overlapping QC flags must not be summed as mutually exclusive removals.

## Source products and temporal scope

The manuscript uses SWOT RiverSP Version D, SoS Version 3, SWORD v16 and v17b, and Hydrocron 1.6.4. Product identifiers and their original DOIs are in `metadata/source_products.csv`. These upstream products are cited at their original repositories and are not copied into this release. The longer satellite archive extends from April 2023 to August 2026; complete modeled SWOT–gauge pairs cover 6 April 2023 to 7 March 2024 (337 days). The long archive window must not be interpreted as the duration of discharge validation.

## Citation and licenses

Please cite this release using `CITATION.cff` or `citation.bib`, its version **1.0.0**, and the eventual DOI for this specific release after deposit. Also cite the manuscript and relevant source products. No manuscript publication DOI or release DOI is invented here. `ZENODO_UPLOAD.md` explains where to add the assigned DOI after deposit.

Study-authored software under `code/` and `scripts/` is licensed under **MIT** (`LICENSE_CODE.txt`). Original public aggregate results, documentation and figure elements are licensed under **CC BY 4.0** (`LICENSE_DATA.txt`). Third-party packages and any attributed third-party figure elements retain their own terms; the root `LICENSE` explains this boundary. Upstream source datasets and the excluded gauge observations are not relicensed by this release.

## Verification and known limitations

`metadata/release_validation.json` records the checks actually performed during preparation. These checks cover manuscript alignment, released figure bytes, aggregate values, excluded file/schema classes, code syntax, public quick-start execution and ZIP integrity. They do not certify permission to access restricted observations and do not claim a rerun of the full numerical analysis. The source/release manifests preserve hashes and relative provenance without disclosing private filesystem paths.

The package is ready for local inspection and deposit as a public, aggregate-level research artifact. Once a repository DOI is assigned, update the citation files and regenerate file checksums before uploading a replacement file set or a new version. The manuscript's Data/Code Availability statements and the corresponding reviewer replies should then cite the actual accessible record.
