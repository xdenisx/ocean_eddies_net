# Metrics and losses

## Counts and metrics

For class c, evaluated one versus the remaining classes:

```text
IoU         = TP / (TP + FP + FN)
Dice / F1   = 2 TP / (2 TP + FP + FN)
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)
Specificity = TN / (TN + FP)
Pixel accuracy = sum(diagonal(confusion)) / sum(confusion)
```

Confusion rows are reference classes; columns are predictions. Counts are summed
across batches/scenes before pooled ratios are computed. NoData/ignored reference
pixels do not contribute. Class counts include background. Macro metrics are
unweighted means over defined class ratios; foreground macro excludes index 0.
Undefined denominators produce null / empty CSV, excluded from means. A class
absent in both reference and prediction has undefined IoU, not 1. An absent class
with false-positive predictions has IoU 0 and contributes to macro IoU.

`val_iou` and `val_dice` monitor foreground macro metrics. With binary labels,
these reduce to the eddy class. Pixel accuracy can be high for predominantly
background images; it is not used to choose the best model by default. Object
precision/recall, eddy counts and centroid/boundary distances are not implemented.

## Losses

All models output raw logits with shape N,C,H,W; target indices are N,H,W.
Softmax yields mutually exclusive per-pixel probabilities. Internal target -100
is ignored. Reductions are performed in float32, including mixed-precision runs.
An entirely ignored batch returns differentiable zero loss; a fully ignored epoch
raises an error instead of reporting fictitious metrics.

- CE: weighted cross entropy over valid target indices, divided by the sum of
  target weights.
- Focal: `-w_t * alpha_t * (1-p_t)^gamma * log(p_t)`, averaged over valid pixels.
  `p_t` comes from unweighted log-softmax, never from weighted CE.
  Binary alpha_t is `[1-focal_alpha, focal_alpha]`; multiclass alpha_t is 1,
  and class_weights defines the class weighting.
- Soft Dice: `1 - (2 TP + 1)/(2 TP + FP + FN + 1)` per class.
- Tversky: `1 - (TP + 1)/(TP + alpha*FP + beta*FN + 1)` per class.
- Focal Tversky: the per-class Tversky loss raised to focal_tversky_gamma,
  then averaged. The exponent is explicit because conventions vary.
- dice_ce / dice_focal: equal 0.5/0.5 combinations of the named components.

Soft counts are aggregated over the current batch and spatial dimensions. By
default Dice/Tversky means omit background, while CE/focal include background.
`include_background_loss: true` adds background to overlap-based losses.
Per-class overlap terms are averaged with class_weights. Classes absent in a
batch are not silently dropped; false-positive probability contributes to their
soft overlap loss. A smoothing constant of 1 is used.

Increasing beta relative to alpha changes the penalty for missed foreground in
Tversky; it does not guarantee a measured recall gain. Likewise, stronger
augmentation, a lower binary threshold or larger postprocessing filters must be
chosen using validation results rather than assumed to improve accuracy.

Epoch loss is a valid-pixel-weighted mean of batch loss scalars, not an exact
whole-dataset recomputation of nonlinear Dice/Tversky. IoU/Dice metrics, unlike
loss, are computed from pooled counts and do not depend on averaging batch ratios.
