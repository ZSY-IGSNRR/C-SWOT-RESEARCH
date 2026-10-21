# Research workflows for manuscript v15

The current workflow is in `fivefold/`. Start with `python code/run.py fivefold --help` and its README. All inputs and numerical outputs are external paths. `code/configuration/fivefold.example.json` preserves the scientific settings and provides editable input locations. The current national experiment is WSE-stratified fivefold CV, with one validation prediction per QC-retained observation.

The seven core modules in `calibration/` match the frozen model used by the latest experiment byte for byte. The new workflow changes assignment and orchestration, not the original Manning equations, width correction, loss or optimizer. Per-gauge results and measured discharge remain local.

`python code/run.py fivefold-figures --help` draws the current Figures 1–7. Explicit inputs replace personal installation paths and the previous dependency on a manuscript DOCX. The Figure 1 source is `figures/assets/source_Figure_1.png`.

The earlier top-level commands remain available for the original ten-assignment/temporal analysis: `calibration`, `postprocess`, `figure-cache`, `wse-domain`, `statistics-figures`, `method-figure`, `error-diagnostics`, and `upstream`. They are not current fivefold commands. Their historical settings are retained in calibration_v15.json/postprocessing_v15.json. Upstream download commands access remote services only when explicitly invoked.

The current fivefold requirements are in `environment/requirements_fivefold.txt`. Earlier environment files describe the older workflows. Full calibration and empirical/product comparisons require authorized fused observations, metadata and audited product inputs; none is distributed by this code repository.

Source provenance is recorded in SOURCE_MANIFEST.json and the fivefold source manifest. Public verification does not rerun the national model. Use the root README and metadata/release_validation.json for the actual verification scope.
