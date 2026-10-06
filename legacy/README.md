# Original scripts — immutable archive

`train_ocean_eddy_aug_level.py` is the user-designated **stable training baseline**.
It is copied byte-for-byte, not rewritten or superseded by the new package.

`other_versions/` contains all other supplied script versions: earlier training,
BN fix, original inference and postprocessed inference, binary validation and
CLAHE preprocessing. Their presence does not promote any of them to stable.

`SOURCE_MANIFEST.json` records SHA-256 and byte count for every archived source.
Tests verify the manifest. `.gitattributes` prevents Git line-ending conversion
from changing these bytes, including on Windows.

The archived scripts retain their original dependencies, assumptions and known
limitations; they are not silently corrected. In particular, the training masks
are binary, and the legacy SAM-like option is not actual Meta SAM fine-tuning.
Use the new package for explicit multiclass training, and read
`docs/MIGRATION.md` before changing an existing workflow.
