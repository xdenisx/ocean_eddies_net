# Augmentation implementation

The new package uses NumPy/OpenCV transforms rather than depending on evolving
Albumentations keyword forms. The unchanged legacy source retains Albumentations.

| Parameter | light | medium | heavy |
|---|---:|---:|---:|
| Additional affine probability | 0.25 | 0.55 | 0.70 |
| Affine rotation limit, degrees | 15 | 45 | 180 |
| Shift fraction per axis | 0.03 | 0.05 | 0.08 |
| Scale deviation | 0.05 | 0.10 | 0.15 |
| Brightness/contrast probability | 0.25 | 0.40 | 0.60 |
| Brightness/contrast limit | 0.10 | 0.18 | 0.25 |
| Gamma probability | 0.15 | 0.30 | 0.40 |
| Gamma exponent range | 0.90–1.10 | 0.80–1.25 | 0.70–1.40 |
| Noise probability | 0.10 | 0.25 | 0.40 |
| Noise maximum standard deviation ([0,1] images) | 0.015 | 0.035 | 0.060 |
| Gaussian blur probability | 0.05 | 0.15 | 0.30 |

All levels sample 0/90/180/270-degree rotation uniformly. Optional horizontal and
vertical reflections each have probability 0.5: increasing flip probability to
0.7 would not create additional orientations. Heavy blur selects 3, 5 or 7 pixels;
other presets select 3 or 5. There is one combined additional affine operation,
not repeated resampling through separate rotate and shift/scale transforms.

Images use bilinear affine interpolation; categorical targets use nearest neighbor.
Valid support is warped too; interpolated image pixels touching invalid support
are ignored. Borders use constant filling rather than creating reflected labels.
Noise/brightness/gamma are image-only. There is no elastic/perspective transform,
mask-erasing coarse dropout, or forced change of label identities.

`percentile` and `minmax` images use [0,1] radiometric transforms. `zscore` permits
signed values, applies contrast/brightness/noise in standardized units, and skips
gamma and [0,1] clipping. `none` applies only geometry/blur because the physical
intensity range is unknown. Gaussian blur is normalized over valid support.

`allow_flips: false` / `--no_flips` disables both reflections. For classes based
on chirality or another orientation-dependent meaning, keep this disabled unless
a correct corresponding label transformation is defined. Rotations still apply.
The package does not infer cyclonic/anticyclonic labels from optical appearance.
`--tta` is also reflection-based; leave it off for these classes.

Augmentations are seeded from run seed, epoch and tile index; validation receives
none. These presets are implementation choices, not a claim of physical validity
for every optical band or every class legend. Compare on held-out data.
