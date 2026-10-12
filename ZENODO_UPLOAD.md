# Zenodo deposit instructions for this local release

This document describes a future deposit. It is not an upload script.

1. Unzip the package and run `python scripts/check_release.py`. Read README.md,
   the code environment notes, and the data/figure traceability files.
2. Sign in to Zenodo and create a new upload. Use the title, author order,
   description, keywords and original product DOIs in metadata/zenodo_metadata.json.
   Use release version 1.0.0. Resource type Dataset is appropriate for this mixed
   public aggregate-data/software package. All included files can be public;
   excluded provider data must not be added as a completeness shortcut.
3. Add the ZIP and, if useful for record preview, the English README as a
   separate file. Record CC BY 4.0 for original aggregates/documentation and MIT
   for study-authored code; describe the scopes in the record. These scopes do
   not relicense upstream provider datasets or third-party components.
4. Reserve a DOI. Add the assigned identifier to CITATION.cff (`doi:`),
   citation.bib (`doi = {...}`), metadata/zenodo_metadata.json, and the record
   description. Run `python scripts/refresh_manifest.py` and then
   `python scripts/check_release.py`, and recreate the ZIP after those edits. The
   refresh utility permits citation/record metadata changes only; changes to
   research data, figures, code or licenses require a newly validated release. A
   reserved DOI is not registered until publication of the record.
5. Preview and check the complete file list before publishing. After publication,
   check the landing page and file download without being signed in.
6. Update the manuscript's data/code availability statements, references and
   corresponding reviewer replies using the actual accessible release DOI.
   Retain the restriction statement for the excluded gauge/evaluation inputs.
7. For substantive later file updates create a new Zenodo version and cite the
   specific version used for the paper. Manuscript revision v15 and data release
   version 1.0.0 are different identifiers.

Official documentation consulted:
- https://help.zenodo.org/docs/deposit/create-new-upload/
- https://help.zenodo.org/docs/deposit/manage-versions/
- https://help.zenodo.org/docs/deposit/describe-records/licenses/
- https://www.agu.org/publications/authors/journals/data-software-for-authors

Author names are taken from the v15 manuscript. Add ORCIDs or a later paper DOI
only when their real identifiers are available; no placeholders are used here.
