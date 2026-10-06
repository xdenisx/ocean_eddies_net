# Ocean Eddy Segmentation

GeoTIFF training, tiled inference, class-aware postprocessing, pixel validation,
and HistEq/CLAHE preprocessing in one Python package.

**Version: 0.1.0rc1 — release candidate, not a replacement for the stable script.**
The original `train_ocean_eddy_aug_level.py` is preserved byte-for-byte under
`legacy/`. All other supplied scripts are archived under `legacy/other_versions/`.
See [Russian instructions](README_RU.md), [migration notes](docs/MIGRATION.md),
and [test status](docs/TEST_REPORT.md).

This is **semantic segmentation**: one mutually exclusive class per pixel,
including background. It is not whole-image classification, multi-label
segmentation, or instance separation of touching eddies.

## Install

Create an isolated environment. Install a PyTorch/torchvision build suitable for
your CPU or CUDA environment first; do not replace a working research environment
in-place. The dependency ranges are declared in `pyproject.toml`; they are not a
claim that every version combination has been tested.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[all]"
eddy-doctor
```

The `all` extra installs the external model backends. A smaller installation for
DeepLabV3+ / U-Net++ / SegFormer is `python -m pip install -e ".[smp]"`.
For just the native smoke-test models and data tools, use `python -m pip install -e .`.
For development tests add `python -m pip install -e ".[dev]"`.

In Jupyter, use `%pip install -e "/absolute/path/to/repository[all]"`, restart the
kernel, and check `sys.executable`. `%pip` targets the active kernel environment.
Use `num_workers: 0` initially on Windows and in notebooks.

## Dataset

```text
data/
  train/
    images/scene_001.tif
    masks/scene_001.tif
  test/
    images/scene_101.tif
    masks/scene_101.tif
```

Image and mask rasters must have the same dimensions, CRS and affine transform.
Pairing uses **relative paths and filename stems**; `.tif` and `.tiff` both work.
Masks must be one-band integer codes, not RGB colors. Input images may have one
or more channels. The first `in_channels` bands are read in the same order at
training/inference; missing bands are an error, not silently zero-padded.

Default binary codes: **0 = background, 255 = eddy**.
For multiple classes, supply an explicit ordered legend. Example:

```yaml
in_channels: 1
class_values: [0, 100, 255]
class_names: [background, eddy_type_1, eddy_type_2]
ignore_values: [65535]
unknown_mask_policy: error
```

These map internally to indices `0, 1, 2`. The first entry is background; `255`
is an ordinary class when listed, **not** an implicit ignore value. Output masks
restore the original codes as `uint16`, with **65535 = NoData** and an internal
GDAL validity mask. Valid code 0 and valid code 255 are not lost as NoData.
Unknown codes fail by default. `unknown_mask_policy: background` is available for
intentional legacy binary behavior; `ignore` supports unlabeled regions.
See [the data contract](docs/DATA_CONTRACT.md).

## Train

```bash
eddy-train --config configs/binary_deeplab.yaml
eddy-train --config configs/multiclass_deeplab.yaml
```

Edit the paths and annotation legend before running. CLI flags override YAML:

```bash
eddy-train --config configs/multiclass_deeplab.yaml \
  --output_dir runs/multiclass_trial \
  --architecture deeplabv3plus_resnet101 \
  --augment_level heavy --batch_size 1 --freeze_batchnorm
```

Equivalent: `python -m ocean_eddy train ...` or `python scripts/train.py ...`.
Both `--augment_level` and `--augment-level` are accepted. `--num_classes`, when
provided, checks the legend length; it does not invent missing class mappings.

Tiles default to 512×512. Fully invalid tiles are always skipped, even with
`min_valid_fraction: 0`. Valid image support respects GeoTIFF masks, finite values
in all selected bands, and optionally the all-zero-pixel rule. Invalid/padded
pixels are excluded from **every** loss and metric rather than labeled background.

Splitting is by scene before tiling. At least two usable scenes are required.
For related acquisitions/overlapping scenes, make an explicit `split_file` JSON
with `train` and `val` lists of relative scene IDs, grouped geographically and/or
by acquisition. Scene splitting alone cannot guarantee independence between
nearby/overlapping acquisitions. A tile split exists only as an explicit fallback
and warns about leakage. Validation retains background tiles and never applies
`min_positive_fraction`; final reporting should use an independent held-out set.

Augmentation presets: `light`, `medium`, `heavy`; disable with `--no_augment`.
Image, class mask and validity share the geometric transform. Masks use nearest
neighbor interpolation. Constant synthetic borders are ignored, not reflected
into additional labels. Radiometric transforms are image-only; signed z-score
images do not receive gamma or [0,1] clipping, and raw `normalize: none` images
receive geometry/blur only. No dropout erases class labels. Reflections can
reverse handedness, so `allow_flips: false` is the multiclass example default.
See [augmentation details](docs/AUGMENTATION.md).

Losses: `ce`, `dice`, `dice_ce`, `focal`, `dice_focal`, `tversky`,
`focal_tversky`, all implemented for C classes. `class_weights` is ordered exactly
like `class_values`. CE/focal include all classes; Dice/Tversky exclude background
by default. `focal_alpha` is binary-only; use `class_weights` for multiclass.
The Tversky false-positive/false-negative penalties and focal exponent are explicit.
See [metric/loss definitions](docs/METRICS_AND_LOSSES.md).

Early stopping supports `--patience`, `--early_stop_monitor val_iou|val_dice|val_loss`
and `--min_delta`. With multiple classes, `val_iou` and `val_dice` are means over
defined foreground-class metrics. Best checkpoints track every strict improvement;
`min_delta` controls only the patience reset. Training writes:

```text
best_checkpoint.pt   last_checkpoint.pt
config.json          class_schema.json
split.json           environment.json
history.json         training_log.csv
```

Batch-size-1 models with pooled BatchNorm need `--freeze_batchnorm`. Running
statistics are frozen after each `model.train()`; affine parameters remain
trainable. Training drops the final incomplete batch by default; validation never
drops a batch. Gradient accumulation does not increase BatchNorm's actual batch.
An exact interrupted run can be resumed using the same config plus
`--resume runs/my_run/last_checkpoint.pt`. Do not change the planned epoch count,
model or data configuration during resume; the cosine schedule belongs to that run.

## Architectures

| CLI name | Implementation / installation |
|---|---|
| `deeplabv3plus_resnet50`, `deeplabv3plus_resnet101` | SMP DeepLabV3+; `[smp]` |
| `unetpp_effb4`, `unetpp_effb5` | SMP U-Net++ / timm EfficientNet; `[smp]` |
| `segformer`, `segformer_b0`, `segformer_b2`, `segformer_b4`, `segformer_b5` | SMP by default, or explicit `--backend hf`; `segformer` aliases B2 |
| `upernet_swin_t`, `upernet_swin_s` | Hugging Face UPerNet + Swin; `[hf]` |
| `hrnet_w18`, `hrnet_w32` | timm HRNet with the supplied custom fusion head; `[timm]` |
| `transunet` | Supplied **custom TransUNet-like** model; not an official reproduction |
| `sam_vit_unet` | Legacy alias for a larger custom TransUNet-like model; **not Meta SAM / MedSAM**, no SAM weights |
| `tiny_unet` | Small native model for CPU smoke tests only |

The model registry is shared by training and inference. The chosen backend is
saved and does not silently fall back to a different model. In the new package,
HF `pretrained: true` loads actual weights, not just the config. For arbitrary
channels, SegFormer adapts its first projection and UPerNet uses the supplied
learned 1×1 input adapter. No ImageNet mean/std preprocessing is silently applied;
the configured per-tile normalization is used consistently. Pretrained features
are only an initialization, not evidence of improved eddy performance.

Missing dependencies produce an installation hint. To check your environment:

```bash
eddy-doctor --architecture deeplabv3plus_resnet101 --num_classes 3 --forward
```

Inference reconstructs from embedded metadata, loads weights **strictly**, and
does not download pretrained weights. Only load checkpoints you trust; the loader
uses `torch.load(..., weights_only=True)` and never silently switches to unsafe
pickle loading.

## Infer

```bash
eddy-infer --input data/test/images \
  --checkpoint runs/deeplab_multiclass/best_checkpoint.pt \
  --output_dir predictions/multiclass \
  --tile_size 512 --stride 256 --batch_size 2 \
  --blend_mode distance --smooth_sigma 1.0 \
  --min_object_size 200 --min_hole_size 100 \
  --save_prob --save_raw_prob
```

The class legend, channel count, normalization and zero/NoData policy come from
the checkpoint. All-empty windows are skipped. Weighted overlap averaging supports
`uniform`, `distance`, `hann`, and `gaussian`; weights are strictly positive at
image edges. Optional `--tta` averages predictions from four flip views. **Do not
use flip TTA when a reflection changes class meaning.**

For binary models only, use `--threshold 0.35` to change the operating point.
Choose thresholds and postprocessing parameters on validation data, not final test
data. Multiclass inference uses `argmax`; it does not apply separate overlapping
binary thresholds. Smoothing is performed on probabilities, with valid-support
normalization and renormalization across classes. Small objects are removed
separately per class; hole filling never overwrites another foreground class or
NoData. Sizes and Gaussian sigma are in **pixels**, not physical units.
Per-class minimum sizes can be provided as original-code JSON:

```bash
--class_min_sizes '{"100":200,"255":100}'
```

Outputs: `scene_pred.tif`, optional `scene_pred_prob.tif` and
`scene_pred_raw_prob.tif`. Probability files contain **one band per class,
including background**, in legend order. `_prob` is after smoothing but before
object cleanup; `_raw_prob` is the blended map before smoothing. Labels can
therefore differ from a simple argmax of the saved smoothed probabilities after
object cleanup. Probability outputs use float32 and NaN NoData.

Accumulation is disk-backed: about `4*(C+1)*height*width` scratch bytes, plus output
files. Processing still needs RAM for tile batches, one or more whole-image 2D
arrays, and component labeling; it is not unlimited-memory streaming. Set
`--work_dir` to a large local scratch directory. Output must be outside input.
Existing outputs fail unless `--overwrite` is explicit.

## Validate

```bash
eddy-validate --pred_dir predictions/multiclass \
  --mask_dir data/test/masks --images_dir data/test/images \
  --checkpoint runs/deeplab_multiclass/best_checkpoint.pt \
  --output_dir metrics/multiclass
```

Validation checks alignment, aggregates confusion counts over valid pixels, and
reports per-image, per-class, macro, foreground-macro and overall statistics.
Outputs include `metrics_summary.json`, `metrics_per_image.csv`,
`metrics_per_image_class.csv`, `metrics_per_class.csv`, and `confusion_matrix.csv`.
IoU, Dice/F1, precision, recall, specificity, pixel accuracy and TP/FP/FN/TN are
included. Undefined class ratios are JSON `null` / empty CSV, not fabricated
perfect scores. Matrix rows are reference classes and columns predicted classes.
Missing predictions fail by default. Explicitly ignored coverage and missing
reference scenes are reported, not silently removed. These are **pixel metrics**;
object detection/instance metrics are not implemented in this release.

## Preprocess

```bash
eddy-preprocess --input_dir data/train/images \
  --output_dir data/train/images_hist_clahe --clip_limit 0.03
```

Chain: cv2 minmax [0,1] → masked global histogram equalization →
`skimage.exposure.equalize_adapthist` → cv2 minmax uint8 [0,255].
Use `--skip_equalize_hist` for CLAHE-only and `--kernel_size 128` for an explicit
context size. File stems and georeferencing are retained. Each band is processed
independently; masks are **never** histogram-equalized. Since CLAHE has no mask
parameter, invalid locations are temporarily filled from nearest valid neighbors;
this is an approximation, not masked CLAHE. Whole images are read into memory.

Output zero can be a valid dark pixel. An internal GDAL mask distinguishes it from
missing data; set **`zero_is_nodata: false`** for training on these outputs. The
setting is stored for inference and inherited by validation. Apply the same chosen
preprocessing to training, validation and test inputs. CLAHE changes radiometry
and may amplify artifacts; its usefulness must be measured, not assumed.

## Existing binary checkpoints

The original scripts remain usable in `legacy/`. For the new interface:

```bash
eddy-convert-legacy --input old_run/best_checkpoint.pt --output converted_binary.pt
eddy-infer --input data/test/images --checkpoint converted_binary.pt \
  --output_dir predictions/legacy_converted --threshold 0.35
```

Conversion verifies state-dict compatibility and saves class/normalization
metadata. It does **not** train new classes, permit multiclass training resume, or
relax strict weight loading. HF legacy conversion may need one configuration
fetch because old checkpoints did not store it; converted inference is offline.
Version differences in third-party model implementations can still prevent
conversion. The exact normalization formula is retained, while new IO validity
checks remain active. See [migration details](docs/MIGRATION.md).

## Tests and GitHub publication

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/make_demo_data.py
eddy-train --config configs/smoke_cpu.yaml
```

The synthetic workflow verifies execution and file contracts, not model quality.
External model tests skip when optional backends are absent; a skip is not a pass.
Actual local results and limitations are in [TEST_REPORT](docs/TEST_REPORT.md).
A CPU CI workflow is supplied; it has not been run on GitHub by this package build.

Before public release, review [NOTICE_RELEASE.md](NOTICE_RELEASE.md), choose the
appropriate license/attribution, and confirm permission to distribute all code.
No license, institution ownership, author list, DOI or research performance claim
has been invented. Data and weights are excluded by `.gitignore`.

```bash
git init
git add .
git commit -m "Package ocean eddy segmentation with multiclass support"
git branch -M main
# Create an empty GitHub repository, then insert its actual URL:
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

This archive does not publish or modify a remote GitHub repository.
