# Figure 1 assets

The current Figure 1 combines conceptual satellite/river artwork, a real ADCP field photograph, a schematic hydraulic cross-section, and a fivefold validation diagram. The river artwork is an AI-generated illustration, not a surveyed study reach or an official NASA/CNES image. Its spacecraft appearance was informed by the previously referenced CNES / Mira Productions illustration; those source rights are unchanged.

`USGS_Alvin_ADCP.jpg` is the public-domain photograph by Tim Merrick / USGS, from https://www.usgs.gov/media/images/measuring-boise-river-streamflow-adcp-0 . It illustrates measurement in the Boise River and is not a Chinese study gauge.

`source_Figure_1.png` preserves the source composition used by the current renderer. Only its upper two rows are retained; the renderer constructs the current fivefold bottom row. Its original bottom row is historical and must not be substituted for final `figures/Figure_1.png`.

`Fig1a_background.png` is the older generated conceptual background retained for the legacy method-figure entry point. The final current workflow is `code/run.py fivefold-figures --figures 1`, using `source_Figure_1.png`.

The project license applies to author-created elements to the extent that the authors hold rights. Attributed third-party elements retain their original terms. The image hashes and current manuscript correspondence are recorded in figure_manifest.json.
