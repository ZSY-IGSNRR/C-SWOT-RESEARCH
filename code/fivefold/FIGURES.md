# Fivefold manuscript figure entry point

`build_figures.py --help` uses only the standard library and does not read data.
The script accepts explicit external inputs and writes to a new empty local
directory. It does not refit the model or alter any input files. Scientific plot
functions and numerical summaries retain the completed manuscript workflow.

Examples (replace the external paths with locally authorized inputs):

```text
python code/fivefold/build_figures.py --source-figure local/source_Figure_1.png --output local/figure1 --figures 1
python code/fivefold/build_figures.py --root local/fivefold_run --output local/figure4 --figures 4
```

For the complete set, provide `--root`, `--evidence`, `--coordinates`,
`--qc-summary`, `--basin`, `--source-figure` and `--output`. `--error-evidence`
names the Figure 5 screening CSV directory, defaulting to `errors/` beside
`--evidence`. `--figures` accepts one or more integers from 1 to 7.

Figure 1 requires the final source_Figure_1.png (not a manuscript DOCX), retaining
its first two panels and drawing the fivefold third panel. Figure 2 needs station
coordinates, QC counts and basin boundaries. Figure 3 needs the local WSE–Z gauge
evidence and calibrated width parameters. Figure 4 uses network/fold aggregates.
Figure 5 needs saved screening evidence. Figure 6 reads local matched-product
records; Figure 7 reads local gauge metrics. Restricted inputs and gauge-level
outputs, including Figure_7_paired_gauge_derived_scores.csv, stay local and must
not be copied into public release data. Aggregate plot-data exports are distinct
from those gauge-level records.

The active Python environment needs numpy, pandas, matplotlib and Pillow. Maps
also need pyshp, shapely and pyproj. There are no per-computer DLL or package-path
fallbacks. The map asset reproduces the author-supplied national boundary and
uses a separate South China Sea inset; its provenance and hashes are in assets/.
External basin .shp/.shx/.dbf/.prj files and station-coordinate tables are not
distributed by this source module. The original map coverage and inset
occlusion assertions remain enabled.
