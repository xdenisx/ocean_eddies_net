#!/usr/bin/env python3
"""
Batch GeoTIFF preprocessing with global histogram equalization + CLAHE.

For each GeoTIFF:
  1) read each band
  2) normalize valid pixels to [0, 1] with cv2.normalize
  3) apply skimage.exposure.equalize_hist
  4) apply skimage.exposure.equalize_adapthist
  5) normalize result to [0, 255] uint8 with cv2.normalize
  6) save georeferenced uint8 GeoTIFF

The script preserves:
  - CRS
  - transform
  - dimensions
  - number of bands
  - folder structure, if --recursive is used

NaN/nodata pixels are kept as 0 in the output by default.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from skimage import exposure


def normalize_valid_to_01(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Normalize only valid pixels to [0, 1] using cv2.normalize.
    Invalid pixels are filled with 0.
    """
    out = np.zeros(img.shape, dtype=np.float32)

    if not np.any(valid):
        return out

    vals = img[valid].astype(np.float32)

    if vals.size == 0:
        return out

    if np.nanmax(vals) <= np.nanmin(vals):
        out[valid] = 0.0
        return out

    vals_norm = cv2.normalize(
        vals,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    out[valid] = vals_norm.reshape(-1).astype(np.float32)
    return out


def normalize_valid_to_uint8(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Normalize only valid pixels to [0, 255] uint8 using cv2.normalize.
    Invalid pixels are set to 0.
    """
    out = np.zeros(img.shape, dtype=np.uint8)

    if not np.any(valid):
        return out

    vals = img[valid].astype(np.float32)

    if vals.size == 0:
        return out

    if np.nanmax(vals) <= np.nanmin(vals):
        out[valid] = 0
        return out

    vals_u8 = cv2.normalize(
        vals,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    out[valid] = vals_u8.reshape(-1).astype(np.uint8)
    return out


def process_band(
    band: np.ndarray,
    nodata=None,
    clip_limit: float = 0.03,
    kernel_size=None,
    equalize_hist_first: bool = True,
) -> np.ndarray:
    """
    Apply:
        img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_norm_hist = exposure.equalize_hist(img_norm)
        img_norm_hist = exposure.equalize_adapthist(img_norm_hist, clip_limit=0.03)
        data = cv2.normalize(img_norm_hist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    but with valid-pixel masking for GeoTIFF nodata/NaN support.
    """
    img = band.astype(np.float32)

    valid = np.isfinite(img)
    if nodata is not None and np.isfinite(nodata):
        valid &= img != nodata

    # If a band is fully empty, return zeros.
    if not np.any(valid):
        return np.zeros(img.shape, dtype=np.uint8)

    img_norm = normalize_valid_to_01(img, valid)

    # skimage equalization operates on full array.
    # Invalid pixels are 0, then restored to 0 in final output.
    if equalize_hist_first:
        img_norm_hist = exposure.equalize_hist(img_norm)
    else:
        img_norm_hist = img_norm

    if kernel_size is None or kernel_size <= 0:
        img_norm_hist = exposure.equalize_adapthist(
            img_norm_hist,
            clip_limit=clip_limit,
        )
    else:
        img_norm_hist = exposure.equalize_adapthist(
            img_norm_hist,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )

    img_norm_hist = img_norm_hist.astype(np.float32)
    data = normalize_valid_to_uint8(img_norm_hist, valid)

    return data


def process_geotiff(
    input_path: Path,
    output_path: Path,
    clip_limit: float,
    kernel_size,
    equalize_hist_first: bool,
    output_nodata: int,
):
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read()
        nodata = src.nodata
        colorinterp = src.colorinterp
        descriptions = src.descriptions
        tags = src.tags()

    out = np.zeros(data.shape, dtype=np.uint8)

    for band_idx in range(data.shape[0]):
        out[band_idx] = process_band(
            data[band_idx],
            nodata=nodata,
            clip_limit=clip_limit,
            kernel_size=kernel_size,
            equalize_hist_first=equalize_hist_first,
        )

    profile.update(
        dtype=rasterio.uint8,
        nodata=output_nodata,
        compress="lzw",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out)

        try:
            dst.colorinterp = colorinterp
        except Exception:
            pass

        for i, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)

        if tags:
            dst.update_tags(**tags)


def collect_files(input_dir: Path, recursive: bool):
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]

    files = []
    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))

    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply cv2 min-max normalization, skimage equalize_hist, "
            "skimage equalize_adapthist, and uint8 normalization to GeoTIFFs."
        )
    )

    parser.add_argument("--input_dir", required=True, help="Input folder with GeoTIFF files.")
    parser.add_argument("--output_dir", required=True, help="Output folder.")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders recursively.")

    parser.add_argument(
        "--clip_limit",
        type=float,
        default=0.03,
        help="CLAHE clip limit for skimage.exposure.equalize_adapthist.",
    )

    parser.add_argument(
        "--kernel_size",
        type=int,
        default=0,
        help=(
            "CLAHE kernel size. Use 0 to let skimage choose automatically. "
            "Try 128 or 256 for large ocean images."
        ),
    )

    parser.add_argument(
        "--skip_equalize_hist",
        action="store_true",
        help="Skip global exposure.equalize_hist before CLAHE.",
    )

    parser.add_argument(
        "--suffix",
        default="_hist_clahe",
        help="Suffix added before file extension. Use empty string to keep original filenames.",
    )

    parser.add_argument(
        "--output_nodata",
        type=int,
        default=0,
        help="Output nodata value for uint8 GeoTIFF.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = collect_files(input_dir, args.recursive)

    print(f"Found {len(files)} GeoTIFF files.")

    kernel_size = None if args.kernel_size <= 0 else args.kernel_size

    for idx, input_path in enumerate(files, start=1):
        rel = input_path.relative_to(input_dir)

        if args.suffix:
            out_name = f"{input_path.stem}{args.suffix}{input_path.suffix}"
            output_path = output_dir / rel.parent / out_name
        else:
            output_path = output_dir / rel

        print(f"[{idx}/{len(files)}] {input_path} -> {output_path}")

        try:
            process_geotiff(
                input_path=input_path,
                output_path=output_path,
                clip_limit=args.clip_limit,
                kernel_size=kernel_size,
                equalize_hist_first=not args.skip_equalize_hist,
                output_nodata=args.output_nodata,
            )
        except Exception as exc:
            print(f"WARNING: failed to process {input_path}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
