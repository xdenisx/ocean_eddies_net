# Release review required

This is a source-code release candidate assembled from scripts supplied in the
conversation plus a new modular multiclass implementation. The original stable
script is archived unchanged and has not been promoted/replaced.

No distribution license or institutional copyright ownership has been assigned
by this package build. Before publishing publicly, the person responsible for
release should determine the correct author/contributor list, ownership and
license and add the approved LICENSE/NOTICE text. Do not infer permissions from
this file. Do not add an institutional claim or blanket MIT license without the
appropriate authorization.

Third-party packages, model implementations and downloaded pretrained weights
have their own licenses and conditions. They are dependencies, not bundled source
or weights in this archive. Check the terms for the versions and checkpoints you
actually distribute or use. Raster imagery and annotation licensing are separate.

The package contains no training imagery, user checkpoints, access tokens, author
email list, DOI, or measured performance claims. Synthetic smoke-test data can be
generated locally. Checkpoint args can contain local data paths; review metadata
before releasing trained checkpoints separately.

GitHub Actions configuration is supplied but not executed on GitHub by this build.
CUDA training and accuracy on real imagery remain to be validated in the target
environment. See docs/TEST_REPORT.md for the actual completed local tests.
