# Observed environments and verification

The current fivefold analysis uses Python 3.12.14, NumPy 2.3.3, pandas 2.3.3,
SciPy 1.16.2, scikit-learn 1.7.2 and Matplotlib 3.10.7. Install the current
`requirements_fivefold.txt` for this workflow; the environment descriptions
below refer to the older release. Geographic drawing also needs pyshp,
Shapely and pyproj, and Figure 1 uses Pillow. No binary environment is vendored.

The local fivefold experiment's saved environment and computation checks
establish these versions. Updating the portable repository does not repeat the
national optimization. Current interface and regression checks are listed in
`metadata/release_validation.json`.

The original source archive records Python 3.13 for calculation. `requirements_main.txt` records versions observed in the existing local Python 3.13 installation during release preparation on 12 October 2026. The scientific source was not recalibrated, and this list is not represented as a package lock captured at the original calculation date. Direct dependencies required by calibration, postprocessing, upstream preparation, and plotting are included.

`requirements_figures.txt` records metadata from the dedicated plotting dependency directory used in preparation of the v15 figures. It provides an alternative plotting environment. Its numpy, pandas, matplotlib, and geographic package versions differ from the local main environment; install the two requirement sets in separate environments. The release does not distribute these libraries or claim a platform-independent binary lock.

The primary bundled Python used for packaging is Python 3.12.14 on Windows, with numpy 2.3.5 and pandas 3.0.1. It does not contain several scientific dependencies, including scipy, scikit-learn, and matplotlib. Source compilation, public entry-point help, and missing-input checks passed in this environment without importing the scientific modules. Native CLI help for calibration, postprocessing, and upstream preparation also passed in the existing local Python 3.13 environment. No network package installation was part of these checks, and no full numerical model rerun is claimed.

Readers can install the listed packages in an appropriate local environment, for example with `python -m pip install -r code/environment/requirements_main.txt`, after choosing a platform compatible with those versions. The example command is guidance for users; it was not executed during packaging. Arial was used in the original figures. A different available font can change text layout and should be checked visually when figures are rebuilt.

Third-party packages retain their individual licensing terms. `third_party_dependencies.json` records their observed metadata, and `third_party_licenses/` contains available installed license texts. These notices do not apply the author-code MIT license to numpy, scipy, matplotlib, geographic libraries, or other dependencies. An empty/unspecified license metadata field means that the copied notice should be consulted; it is not a claim that no license applies.
