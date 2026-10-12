# Final figure provenance and reproduction

The eight PNG files were extracted directly from the clean v15 manuscript without re-encoding. Their SHA-256 values match the manuscript's embedded image bytes. Captions were extracted from the adjacent final caption paragraphs. The manuscript and reviewer responses are not included here.

The actual final captions determine numbering. Figure 5 corresponds to the historical `F06_extreme_errors_width` export, and Figure 6 to `F05_official_comparison`.

Seven confirmed SVG companions and six confirmed PDF companions are included with unchanged bytes. Their source export routines, labels, and metadata were inspected. PDF companions were rendered and compared with the authoritative PNGs. All eight final PNGs were visually checked. The companion formats preserve the final scientific content; raster and vector rendering can differ in font rasterization and antialiasing. Figure 1 has a matching SVG but no proven matching native PDF or drawio file. Older concept diagrams are excluded.

Figure 3 is supplied only as the authoritative PNG. Its native SVG and PDF store individual observed–estimated discharge scatter coordinates and are excluded from the public package. No observation-level Q table, gauge time series, timestamp list, or hidden tooltip dataset is included. The other supplied vectors have no attached observation-level Q dataset, interactive tooltip records, or embedded data table. Their editable geometry is the visible graphic, including public SWOT WSE/width anchors in Figure S1 and derived parameter points in Figure S2.

## Numerical redraw from public summaries

| Figure | Panels supported by public numerical summaries | Panels preserved as the final graphic |
|---|---|---|
| 1 | Conceptual workflow can be rebuilt using the supplied plotting source and generated background; no measured Q is needed. | None. |
| 2 | b, c, d: data counts, paired model differences, and sample-size summaries. | a: station map inputs are absent from the public tables. |
| 3 | d: statistics for each WSE assignment. | a: station map; b: discharge scatter; c: exact station-NSE cumulative distributions. |
| 4 | a–d: WSE-domain and extrapolation-distance summaries. | None. |
| 5 | a–d: anonymous normalized cumulative-error curve, screening statistics, and width-control summaries. | None. |
| 6 | b–d: product-specific support and bootstrap summaries. | a: station membership map. |
| S1 | None: the selected WSE/width anchor-series table is not supplied. | a–d. |
| S2 | a, b, d, e, f: histograms, bound fractions, CV box statistics, and resistance sensitivity. | c: individual fitted-parameter scatter. |

The manifest identifies each supporting file in `data/`. The source-recipe names and hashes record the historical generating scripts; they are provenance identifiers, rather than promises that the original private workspace or archive is included. Use the portable plotting entries supplied in the public code package for supported redraws. Numerical redraw means using already computed summaries. It does not mean recalculating the original calibration, validation predictions, paired comparisons, or bootstrap intervals. Those operations require the restricted gauge inputs and original evaluation records. The final image and captions remain available for panels whose underlying station or observation tables are excluded.

For Figure 5a, the public curve contains only normalized rank and cumulative squared-error percentages. It contains no discharge value, absolute residual, station identity, observation identity, or time. Residual screening remains a retrospective diagnostic; reproducing the curve does not supply a screening rule for unknown discharge.

## Figure 1 background

`assets/Fig1a_background.png` is the exact generated background used by the final Figure 1 source. It is an AI-generated conceptual composite, not a measured scene or an official NASA/CNES image. The satellite appearance references NASA PIA25623, credited to CNES / Mira Productions. The original third-party cutout is absent. See `assets/asset_notes.md` and the package license for provenance and rights limitations.
