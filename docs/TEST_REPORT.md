# Local test report — 0.1.0rc1

## Completed checks

**80 passed, 6 skipped** (`python -m pytest -q`, local CPU environment).
The raw test summary is in `pytest-results.txt`; versions/import results are in
`test-environment.json`.

Passed checks include:

- SHA-256 equality for the stable baseline and all 12 supplied archived scripts.
- CLI parsing for architecture names, heavy augmentation, early-stop monitor,
  min_delta, freeze_batchnorm, and all six command help pages.
- Arbitrary mask-code mapping, binary 255 foreground, ignored values and invalid
  schema rejection.
- Finite multiclass losses and gradients for 2, 3 and 5 classes, all seven losses,
  ignored-pixel zero gradients and all-ignored batches.
- Exact confusion-matrix counts, macro/foreground metrics and undefined-class policy.
- Synchronized augmentation, validity, categorical labels and deterministic seeds
  for 1 and 4 channels, three augmentation levels and three normalization modes.
- Full tiling coverage, positive blend windows, masked smoothing normalization,
  per-class cleanup and protection of image borders/NoData/other classes.
- A real two-epoch CPU training run with the native tiny model, three-class
  checkpoint save/load, tiled flip-TTA inference, smoothing, component cleanup,
  output georeferencing, C-band probabilities and multiclass evaluation.
- A binary CPU training run and output masks where valid 255 is distinct from
  NoData 65535.
- Exact native legacy TransUNet weight-conversion/logit round-trip.
- Multiband validity, missing-channel errors, valid zero pixels after preprocessing,
  and strict/explicit-ignore handling of missing prediction coverage.
- The requested finite-image cv2 + skimage preprocessing chain, within uint8
  rounding tolerance, plus empty/constant/invalid support handling.
- Editable installation via the declared setuptools backend and execution of the
  installed `eddy-doctor --architecture tiny_unet --num_classes 3 --forward`
  entry point, which returned shape [1,3,64,64].
- Installed console-entrypoint smoke workflow on four generated three-class
  GeoTIFFs: eddy-train, eddy-infer, eddy-validate and eddy-preprocess all completed.
  This was a synthetic execution check, not a held-out accuracy experiment.
- Wheel and source-distribution builds completed with setuptools.build_meta.
- Notebook JSON/schema validation. Training cells were not run on real imagery.

## What was not verified here

The optional integration checks for SMP (3), Hugging Face transformers (2) and
timm (1) were **skipped**, not passed. Those packages were unavailable, and network
access from the runtime prevented installing them. Their adapters follow the
inspected APIs but need target-environment forward/backward checks. Use:

```bash
python -m pip install -e ".[all,dev]"
python -m pytest tests/test_optional_models.py -q
eddy-doctor --architecture deeplabv3plus_resnet101 --num_classes 3 --forward
```

CUDA/AMP execution, downloaded pretrained weights, every architecture variant,
Windows multiprocessing, full-scale memory use, external legacy checkpoints,
GitHub Actions execution, and scientific accuracy on the user's imagery were not
tested. Unit tests do not establish ocean-eddy recognition quality or prove that
augmentation/postprocessing improves performance.

The editable installation check reused the runtime's available core dependencies
in a separate test environment; it was not a clean network installation of all
extras. The dependency ranges in pyproject.toml are supported targets, not an
exhaustive compatibility matrix. The supplied CI matrix has not been executed on
GitHub by this build.

The user-designated stable script remains unchanged; only the new modular line
is labeled 0.1.0rc1.
