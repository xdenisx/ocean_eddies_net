# Changelog

## 0.1.0rc1

- Preserve the exact user-designated stable training script and all supplied
  source versions in an immutable legacy archive with SHA-256 manifest.
- Add installable `ocean_eddy` modules, unified CLI, YAML examples and diagnostics.
- Generalize training, losses, inference and validation to arbitrary mutually
  exclusive pixel classes with explicit original-code mapping.
- Add ignored-pixel handling, scene-first split, strict alignment checks and
  training-only foreground filtering.
- Share explicit model backends across training/inference; preserve custom native
  model parameter names; correct documentation of the SAM-like alias.
- Add tested image/mask/validity augmentation presets and BatchNorm-stat freezing.
- Implement configurable early stopping and self-describing resumable checkpoints.
- Add multiclass center-weighted inference, disk-backed accumulation, valid-support
  probability smoothing and per-class component cleanup.
- Add per-class and pooled confusion-matrix validation with explicit missing coverage.
- Package the requested cv2 normalization + skimage HistEq/CLAHE preprocessing,
  preserving valid zero pixels through an internal GDAL mask.
- Add strict legacy binary checkpoint conversion, CPU synthetic tests and CI workflow.

The release-candidate label is intentional. These changes are not applied to the
stable archive and have not been benchmarked on the user's ocean imagery.
