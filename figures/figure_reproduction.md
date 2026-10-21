# Reproducing the current seven figures

The files Figure_1 through Figure_7 match the current manuscript; their PNG bytes were checked against its embedded figures. SVG and PDF copies retain the same scientific panels. Current vector panels show reported aggregate or derived results, maps, and conceptual elements; no point-level observed/predicted discharge table is distributed.

Run `python code/run.py fivefold-figures --help`. The renderer takes a completed external fivefold result root and explicit paths for evidence summaries, station coordinates, QC summaries, basin boundaries and the Figure 1 source image. `--figures` selects individual figures. For Figure 5, provide the screening diagnostic directory with `--error-evidence`. Outputs must go to a separate directory. Required inputs are documented in code/fivefold/README.md.

The national boundary geometry is retained with the renderer and its source hashes. The original source was the user-provided Chinese boundary archive. Maps display the complete supplied extent with the southern offshore extent in an inset.

The historical figures under legacy_v1_0_0 belong to the original ten-assignment manuscript and are retained for provenance. They must not be used as the current paper figures. Data tables, gauge coordinates and observed/predicted discharge pairs are external inputs, not repository files.
