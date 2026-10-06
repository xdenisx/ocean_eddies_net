#!/usr/bin/env python3
"""
Validate ocean eddy segmentation predictions against GeoTIFF masks.

Assumptions by default:
    255 = ocean eddy / foreground
    everything else = background

The script pairs files from prediction and mask folders by basename, with optional
suffix removal such as *_pred.tif -> *.tif.

Outputs:
    - metrics_per_image.csv
    - metrics_summary.json
    - optional confusion_total.npy

Metrics:
    - IoU / Jaccard
    - Dice / F1
    - Precision
    - Recall
    - Specificity
    - Accuracy
    - Balanced accuracy
    - False positive rate
    - False negative rate
    - Pixel counts: TP, FP, FN, TN

Example:
python validate_ocean_eddy_segmentation.py \
    --pred_dir predictions \
    --mask_dir masks \
    --output_dir validation_metrics \
    --pred_suffix _pred \
    --mask_value 255
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling


VALID_EXTENSIONS = {".tif", ".tiff"}


def list_geotiffs(folder: str) -> List[Path]:
    folder_path = Path(folder)
    files = []
    for ext in VALID_EXTENSIONS:
        files.extend(folder_path.rglob(f"*{ext}"))
        files.extend(folder_path.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def normalized_stem(path: Path, suffix_to_remove: str = "") -> str:
    stem = path.stem
    if suffix_to_remove and stem.endswith(suffix_to_remove):
        stem = stem[: -len(suffix_to_remove)]
    return stem


def build_file_map(folder: str, suffix_to_remove: str = "") -> Dict[str, Path]:
    out = {}
    for path in list_geotiffs(folder):
        key = normalized_stem(path, suffix_to_remove=suffix_to_remove)
        if key in out:
            print(f"WARNING: duplicate key {key}; keeping first: {out[key]}")
        else:
            out[key] = path
    return out


def read_mask(path: Path, mask_value: int = 255) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        valid = arr != nodata
    else:
        valid = np.ones(arr.shape, dtype=bool)

    mask = (arr == mask_value) & valid
    return mask.astype(bool), profile


def read_probability_or_mask(path: Path, threshold: float, mask_value: int = 255) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        valid = arr != nodata
    else:
        valid = np.ones(arr.shape, dtype=bool)

    arr = arr.astype("float32")

    # If prediction looks like probability [0, 1], threshold it.
    finite = np.isfinite(arr)
    if np.any(finite):
        mn = float(np.nanmin(arr[finite]))
        mx = float(np.nanmax(arr[finite]))
    else:
        mn, mx = 0.0, 0.0

    if mn >= 0.0 and mx <= 1.0:
        pred = arr >= threshold
    else:
        pred = arr == mask_value

    pred = pred & valid
    return pred.astype(bool), profile


def reproject_prediction_to_mask_grid(
    pred_path: Path,
    mask_profile: dict,
    threshold: float,
    mask_value: int = 255,
) -> np.ndarray:
    with rasterio.open(pred_path) as src:
        pred_arr = src.read(1).astype("float32")
        src_nodata = src.nodata
        src_transform = src.transform
        src_crs = src.crs

        dst = np.zeros(
            (mask_profile["height"], mask_profile["width"]),
            dtype="float32",
        )

        reproject(
            source=pred_arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=mask_profile["transform"],
            dst_crs=mask_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    finite = np.isfinite(dst)
    if np.any(finite):
        mn = float(np.nanmin(dst[finite]))
        mx = float(np.nanmax(dst[finite]))
    else:
        mn, mx = 0.0, 0.0

    if mn >= 0.0 and mx <= 1.0:
        return dst >= threshold
    return dst == mask_value


def compute_counts(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None) -> Dict[str, int]:
    if valid is None:
        valid = np.ones(gt.shape, dtype=bool)

    pred = pred.astype(bool) & valid
    gt = gt.astype(bool) & valid

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt & valid))
    fn = int(np.sum(~pred & gt & valid))
    tn = int(np.sum(~pred & ~gt & valid))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    if np.isfinite(recall) and np.isfinite(specificity):
        balanced_accuracy = 0.5 * (recall + specificity)
    else:
        balanced_accuracy = float("nan")

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def profiles_match(pred_profile: dict, mask_profile: dict) -> bool:
    keys = ["height", "width", "crs", "transform"]
    return all(pred_profile.get(k) == mask_profile.get(k) for k in keys)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate quality metrics for ocean eddy segmentation predictions."
    )
    parser.add_argument("--pred_dir", required=True, help="Folder with predicted GeoTIFF masks/probabilities.")
    parser.add_argument("--mask_dir", required=True, help="Folder with reference mask GeoTIFFs.")
    parser.add_argument("--output_dir", required=True, help="Output folder for CSV/JSON metrics.")
    parser.add_argument("--pred_suffix", default="_pred", help="Suffix to remove from prediction basename before matching.")
    parser.add_argument("--mask_suffix", default="", help="Suffix to remove from mask basename before matching.")
    parser.add_argument("--mask_value", type=int, default=255, help="Foreground value in masks/predictions.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold if prediction files contain probabilities [0, 1].")
    parser.add_argument("--allow_reproject", action="store_true", help="Reproject prediction to reference mask grid if grids differ.")
    parser.add_argument("--save_confusion", action="store_true", help="Save total confusion matrix as .npy.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_map = build_file_map(args.pred_dir, suffix_to_remove=args.pred_suffix)
    mask_map = build_file_map(args.mask_dir, suffix_to_remove=args.mask_suffix)

    common_keys = sorted(set(pred_map.keys()) & set(mask_map.keys()))
    missing_pred = sorted(set(mask_map.keys()) - set(pred_map.keys()))
    missing_mask = sorted(set(pred_map.keys()) - set(mask_map.keys()))

    if not common_keys:
        raise RuntimeError(
            "No matching prediction/mask pairs found. Check --pred_suffix and --mask_suffix."
        )

    if missing_pred:
        print(f"WARNING: {len(missing_pred)} masks have no prediction.")
    if missing_mask:
        print(f"WARNING: {len(missing_mask)} predictions have no mask.")

    rows = []
    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for key in common_keys:
        pred_path = pred_map[key]
        mask_path = mask_map[key]

        gt, mask_profile = read_mask(mask_path, mask_value=args.mask_value)
        pred, pred_profile = read_probability_or_mask(
            pred_path,
            threshold=args.threshold,
            mask_value=args.mask_value,
        )

        if pred.shape != gt.shape or not profiles_match(pred_profile, mask_profile):
            if args.allow_reproject:
                pred = reproject_prediction_to_mask_grid(
                    pred_path,
                    mask_profile=mask_profile,
                    threshold=args.threshold,
                    mask_value=args.mask_value,
                )
            else:
                raise RuntimeError(
                    f"Grid mismatch for {key}. Use --allow_reproject to reproject prediction to mask grid.\n"
                    f"Prediction: {pred_path}\nMask: {mask_path}"
                )

        valid = np.ones(gt.shape, dtype=bool)
        counts = compute_counts(pred, gt, valid=valid)
        metric_values = metrics_from_counts(counts)

        for k in total_counts:
            total_counts[k] += counts[k]

        rows.append(
            {
                "name": key,
                "prediction": str(pred_path),
                "mask": str(mask_path),
                **counts,
                **metric_values,
                "gt_positive_pixels": int(np.sum(gt)),
                "pred_positive_pixels": int(np.sum(pred)),
                "total_pixels": int(gt.size),
            }
        )

        print(
            f"{key}: IoU={metric_values['iou']:.4f}, "
            f"Dice={metric_values['dice']:.4f}, "
            f"Precision={metric_values['precision']:.4f}, "
            f"Recall={metric_values['recall']:.4f}"
        )

    df = pd.DataFrame(rows)
    per_image_csv = output_dir / "metrics_per_image.csv"
    df.to_csv(per_image_csv, index=False)

    total_metrics = metrics_from_counts(total_counts)
    macro_metrics = {
        f"macro_{col}": float(np.nanmean(df[col].values))
        for col in [
            "iou",
            "dice",
            "precision",
            "recall",
            "specificity",
            "accuracy",
            "balanced_accuracy",
            "false_positive_rate",
            "false_negative_rate",
        ]
    }

    summary = {
        "num_pairs": len(common_keys),
        "num_missing_predictions": len(missing_pred),
        "num_missing_masks": len(missing_mask),
        "mask_value": args.mask_value,
        "threshold": args.threshold,
        "total_counts": total_counts,
        "micro_metrics_from_total_counts": total_metrics,
        "macro_metrics_mean_over_images": macro_metrics,
        "missing_predictions": missing_pred,
        "missing_masks": missing_mask,
    }

    summary_json = output_dir / "metrics_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.save_confusion:
        confusion = np.array(
            [[total_counts["tn"], total_counts["fp"]], [total_counts["fn"], total_counts["tp"]]],
            dtype=np.int64,
        )
        np.save(output_dir / "confusion_total.npy", confusion)

    print("\nSummary:")
    print(f"Pairs evaluated: {len(common_keys)}")
    print(f"Micro IoU:  {total_metrics['iou']:.4f}")
    print(f"Micro Dice: {total_metrics['dice']:.4f}")
    print(f"Micro Precision: {total_metrics['precision']:.4f}")
    print(f"Micro Recall:    {total_metrics['recall']:.4f}")
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
