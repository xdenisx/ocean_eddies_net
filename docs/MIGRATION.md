# Migration from the supplied scripts

## Stable means unchanged

The user-designated stable `train_ocean_eddy_aug_level.py` is an exact binary copy
under `legacy/`. Its SHA-256 is
`195a8b386638997629e16b7a1c12cf2fb0f0a1c318e198dcae13c1b3e8280d5d`.
The BN-fix variant and other supplied versions are archived separately and have
not replaced it. The new `0.1.0rc1` package is a separate development line.

The following are deliberate changes **only in the modular package**, not claims
about what the stable script previously implemented.

| Area | Supplied stable behavior | New package behavior |
|---|---|---|
| Labels | `mask == 255`, everything else background; two classes enforced | Explicit code/name legend, C classes, configurable unknown/ignore policy |
| Invalid support | Empty-tile filtering, but invalid/padded pixels can enter background loss | Invalid and padded pixels ignored by all losses/metrics |
| Selected bands | Missing channels padded with zeros | Missing requested channels raise an error |
| Split | Random tile split after filtering | Scene split before tiling; explicit leakage-prone tile fallback |
| Foreground filter | Applied before train/validation split | Applied to training only; validation background retained |
| Metrics | Average ratios over batches | Aggregate confusion counts, then calculate ratios |
| Augmentation | Albumentations transforms, including older keyword forms | NumPy/OpenCV equivalent-strength presets with tested class/validity geometry |
| Label dropout | Original CoarseDropout requests mask fill 0 | No augmentation erases semantic labels |
| Image borders | Reflection padding in affine augmentation | Constant padding, with synthetic/invalid support ignored |
| HF initialization | Configuration loaded, model constructed from config | Explicit actual pretrained weights when requested |
| Model backend | Can depend on available imports/fallbacks | Explicit backend, stored with checkpoint; no silent fallback |
| Inference weight loading | Existing scripts use non-strict loading | Strict state-dict loading |
| BatchNorm option | Stable script has no freeze option | Optional running-stat freeze applied after every `train()` |
| Early stopping | Validation-IoU patience | IoU/Dice/loss monitor, min_delta, independent best-checkpoint updates |
| Checkpoint | Weights, optimizer, args and basic stats | Adds full model/schema/config, scheduler/scaler, RNG and history |
| Inference | Single foreground probability | C probability bands, mutually exclusive labels, class-aware cleanup |
| Output NoData | Potential collision with valid 0/255 | uint16 code mask, NoData 65535 and internal GDAL mask |
| Preprocessing | Original full-array Histogram/CLAHE chain | Same chain with explicit valid-pixel histogram and NoData handling |

The new normalization uses common image support (all selected bands finite and
GDAL-valid; optionally all-zero pixels excluded). It normalizes each selected
band over this support and resets invalid pixels to zero. This is an intentional
change from the original per-band nonzero normalization. Do not assume new and
legacy runs produce identical learning trajectories or metric values.

## Old weights

`eddy-convert-legacy` reads only a tensor/dictionary checkpoint with
`weights_only=True`, identifies supported legacy architectures, verifies exact
parameter keys/shapes and saves package metadata. It keeps the original binary
mask value and original per-band normalization formula. The new raster reader's
validity masking still applies, so scenes with inconsistent metadata may differ
from historical inference. Pixel-perfect compatibility on such scenes is not
asserted.

Native TransUNet conversion is covered by an exact-logit synthetic test. External
SMP/timm/HF checkpoint compatibility must be checked in the environment used to
train those weights; dependencies and their defaults can change. For legacy HF,
the converter retrieves the named config once, preserves the original auxiliary
head structure when present, and embeds it. Missing/unexpected weights stop
conversion; they are not accepted as warnings.

Keep the old checkpoint. A converted file is an inference checkpoint, not an
exact training resume checkpoint. Changing `[0,255]` into `[0,100,255]` requires
training a new segmentation head/model with the new labeled data. The converter
does not claim to infer missing class labels.

## Naming and scientific claims

The native `transunet` architecture is the supplied custom transformer-bottleneck
U-Net with an extra refinement layer. The `sam_vit_unet` option wraps a larger
version of that same model. Neither alias claims official TransUNet reproduction
or Meta SAM/MedSAM fine-tuning. HRNet uses the supplied custom multi-scale fusion
head, not an asserted HRNet+OCR benchmark implementation.

The package is infrastructure, not a new eddy-recognition method or a measured
accuracy improvement. Compare held-out results using the same scenes, class
legend, preprocessing and postprocessing. Do not rank architectures on the basis
of their names or complexity alone.
