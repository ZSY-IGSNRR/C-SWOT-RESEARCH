# Changes made when preparing the v15 public source

## Current fivefold update on 21 October 2026

Version 1.1.0 adds the current fivefold orchestration and analysis under
`fivefold/`, with explicit external work roots and configuration. Scientific
model/QC modules are reused from `calibration/` without modification. The
current plot renderer replaces personal installation paths and DOCX reads
with explicit inputs; its numerical and layout functions retain the original
completed fivefold workflow. New source hashes and portability changes are
listed in `fivefold/frozen_v15/SOURCE_MANIFEST.json` and the code-level manifest.
Git attributes preserve the calibration files' original line endings so their
archive hashes remain identical after checkout on different platforms. This
can appear as an end-of-line-only diff against older Git-normalized blobs.

The repository follows the upstream removal of the old aggregate data
directory. Model outputs, observations, station metadata and document-review
files are external. Current figures are numbered 1–7; the earlier set is
preserved under `figures/legacy_v1_0_0`. Help, input boundaries, fold invariants,
saved-prediction summaries and selected figures are checked separately; the
national optimization is not rerun by this update.

## Earlier source preparation for version 1.0.0

The ten files in `calibration/` retain their exact original bytes. Their source hashes match both the model-code archive and the calculation source. Three older diagnostic plotting files and the Windows launchers were excluded because the v15 figures have separate sources and portable entry points.

Other files retain their numerical logic but are not described as byte-identical. Historical absolute default paths in upstream and postprocessing code were made relative. Saved configuration paths were relocated, with the computational values retained. The public `run.py` supplies explicit paths and checks that required external inputs exist before scientific code is invoked. Helper-module locations are resolved within this code directory.

The final WSE-domain and statistical figure sources came from the focused figure workflow. Unused legacy concept/gallery functions were removed. The functions generating Figures 2, 3, and 4 retain their original syntax trees; the retained SoS, width, and parameter rendering functions are taken from the original helper module. Output naming was updated so the extreme-error/width diagnostic is Figure 5, the SoS comparison is Figure 6, and the parameter figure is supplementary Figure S2. Figure 1 uses the final method-diagram source with configurable background and output paths. Diagnostic output to the console was reduced where it was unnecessary for running the figure preparation.

Source and published-file hashes, archive-relative source names, and file-specific modifications are listed in `SOURCE_MANIFEST.json`. No gauge records, observation-level result tables, pickle caches, credentials, manuscript/reply automation, or downloaded input products were copied. Source compilation, help commands, missing-input handling, and source identity were checked. Full numerical recalibration, postprocessing, and statistical figure regeneration were not performed as part of preparing this release.
