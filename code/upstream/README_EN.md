# Upstream data preparation

This source prepares the station/SWORD inventory, requests SWOT observations through Hydrocron, applies satellite QC, attaches reach variables to nodes, combines same-day gauge observations, assigns roughness, and estimates a constant datum shift. Numerical logic was retained from the upstream archive; only historical default paths were made relative. Use the public `code/run.py upstream` entry point with an explicit configuration.

The station locations, gauge records, roughness NetCDF, SWORD index, downloaded products, and authentication material are not distributed. Download steps require external network access when a reader invokes them. The example config contains paths and settings only. API authentication is supplied through the existing environment/header mechanism, rather than a credential file in this archive.

The `vc_legacy` node matching mode selects the nearest node in degree space and available IDs ±10, with at most three nodes. Node/reach variables use the configured 30-minute match. The legacy positive-slope fallback has no maximum lag and retains its source timestamp; gauge matching uses the same UTC date. Centered satellite QC and datum alignment use complete records, so this workflow supplies retrospective estimates. An operational application requires an additional evaluation using only information available at prediction time.

The archive includes version/network audit and SoS comparison tools. An optional SoS Version 2.0 layer remains separate from Version D node/reach data and from the manuscript's Version 3 benchmark. Verify the collection/version and crosswalk before changing or extending that benchmark.
