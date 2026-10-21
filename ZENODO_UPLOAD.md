# Optional software deposit

This file is preparation guidance, not an upload script. The current repository profile is code and figures, version 1.1.0. It does not restore the deleted aggregate data directory or include restricted observations.

Run `python scripts/check_release.py` before preparing a deposit. Use metadata/zenodo_metadata.json, the current author list, source identifiers and license scopes as a template; verify the actual record and identifiers when depositing. Choose the software resource type for this code version. Retain existing third-party attribution and external-data restrictions.

Add a DOI only after the repository service actually assigns it. For citation metadata edits, run the protected refresh_manifest.py and recheck the release. Changes to scientific code or file membership require an explicit new version and a new validation record. Do not overwrite a deposited version with a different analysis while retaining its identifier.

Creating a record, uploading files or publishing it is outside this local repository update.
