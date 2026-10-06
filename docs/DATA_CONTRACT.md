# Data and checkpoint contract

## Raster inputs

GeoTIFF/TIFF, one or more numeric image bands; categorical masks are single-band.
Image/mask pairs must have identical height, width, CRS and affine transform.
The code checks alignment but does not resample labels automatically. Use nearest
neighbor when deliberately resampling a categorical mask outside this package.

Default image validity requires all selected bands to be finite and GDAL-valid.
With `zero_is_nodata: true`, an all-zero multichannel pixel is additionally
invalid. A zero in one band can still be valid when other bands contain data.
With false, zeros are accepted whenever the GDAL validity mask accepts them.
Invalid pixels are ignored in loss/metrics; their input tensor values are zero.
No silent missing-channel replication or zero-padding is performed.

## Class codes

`class_values` is ordered. Index 0 is background even when its original code is
not zero. Values must be unique integers in 0..65534; names must be unique and
have the same length. Output code 65535 is reserved for NoData. `ignore_values`
is disjoint from class_values. Internal ignored target value is -100.

Examples:

- Binary: `[0,255]` -> internal `[0,1]`.
- Three classes: `[0,1,2]` -> internal `[0,1,2]`.
- Three arbitrary codes: `[0,100,255]` -> internal `[0,1,2]`.

Unknown codes error by default. Unlabeled pixels are not automatically background.
Set explicit ignore_values or unknown_mask_policy=ignore when appropriate. For
intentional legacy binary mapping only, unknown_mask_policy=background is available.

Source mask nodata conflicts are reported during indexing. When a GDAL mask is
specifically derived from the declared nodata value, explicit class_values take
priority over that declaration. This keeps a foreground code of 255 usable even
in a mislabeled file declaring nodata=255. An explicit internal/external GDAL
mask still applies. NaN, configured ignored codes and invalid image support stay
ignored regardless of metadata. Prefer correcting contradictory mask metadata
rather than relying on this compatibility rule.

## Training geometry and sampling

Windows cover the full image extent, including a last shifted window at an edge.
Thus edge windows may overlap even when stride==tile_size; tile splitting can leak.
Pixels outside small images are padded and ignored. All-empty image windows and
windows with no supervised labels are excluded. min_valid_fraction is the valid
fraction of the real, unpadded window. min_positive_fraction is the fraction of
supervised pixels in any non-background class; it is a training-only filter.
It does not balance individual foreground classes or guarantee higher recall.

An explicit scene split JSON can list subsets and leave other scenes reserved for
test. The same JSON can be reused for comparisons. Scene IDs are relative paths
without the TIFF extension, preserving subfolder components and dots in names.

## Predictions

Class masks are uint16 with original codes, NoData 65535, and internal GDAL masks.
Probabilities are float32, NaN outside predicted image support. There are C bands,
including background; descriptions include class name and original code.
Tags store schema, normalization policy, tile/stride, blending and cleanup settings.

Files restore source dimensions, transform and CRS. Original palette, physical
scales, statistics and JPEG settings are not blindly copied onto class/probability
rasters. Object area thresholds are pixel counts; projected/geodesic area is not
computed in this release.

## Checkpoint

format_version=1 includes model_spec (architecture, explicit backend, channels,
classes and HF config when needed), class_schema, preprocessing and state_dict.
Training checkpoints additionally store full config, optimizer/scheduler/scaler,
epoch/history/best metric/early-stopping state, loader generator and PyTorch RNG.
Files use safe dictionary/tensor contents. Loading is strict, with weights_only=True.
Only load trusted checkpoints: that option is not a substitute for provenance.

Resume is for an interrupted run with the same configuration and data. The
checkpoint does not hash the source raster bytes, so you must not modify the
underlying images/masks between resume attempts. Changing epoch count changes the
planned cosine schedule and is deliberately rejected. Bitwise equivalence across
different hardware, library versions or nondeterministic CUDA operations is not
promised.

## Validation scope

Use original images to define reference support, particularly at NoData borders.
Image channel/zero policies are inherited from checkpoint or prediction tags unless
explicitly overridden. Missing predictions over valid annotated support fail by
default. `--missing_prediction_policy ignore` excludes them but records counts.
Missing scene files fail unless `--allow_missing` is explicit; missing IDs are
reported. Validation does not treat missing coverage as successful background.
