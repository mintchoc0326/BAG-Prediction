# Local data (not distributed)

Place authorized source datasets and reference resources in `final/ADNI/`, `final/HABS/`, `final/ADRC/`, and `final/ADDecode/`. Configure exact connectome, atlas, and participant metadata paths in the current scripts before running them. DWI harmonization uses the configured plain connectomes; fMRI uses each configured fMRI directory.

Raw data, participant metadata, atlas resources, derived arrays, historical data, and symlinks remain local and are excluded from Git. Dataset access and redistribution permissions must be obtained separately.
