#!/usr/bin/env python3
"""
Apply CLAHE / adaptive histogram equalization to GeoTIFF images using
skimage.exposure.equalize_adapthist.

Features
--------
- Processes all .tif / .tiff files in a folder.
- Optional recursive search through subfolders.
- Preserves GeoTIFF georeferencing, CRS, transform, width/height, band count.
- Applies CLAHE independently to each band.
- Handles NaN and nodata values.
- Supports percentile clipping before CLAHE.
- Saves outputs to another directory, optionally preserving folder structure.

Example
-------
python clahe_skimage_geotiff_folder.py \
    --input_dir input_images \
    --output_dir output_clahe \
    --recursive \
    --clip_limit 0.01 \
    --kernel_size 128 \
    --p_low 2 \
    --p_high 98
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from skimage import exposure
import warnings


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def find_geotiffs(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))
    return sorted(set(files))


def parse_kernel_size(value: str) -> Optional[Tuple[int, int]]:
    """
    Parse kernel size from CLI.

    Accepted:
    - "none" -> None, skimage chooses automatically
    - "128" -> (128, 128)
    - "128,256" -> (128, 256)
    """
    value = str(value).strip().lower()
    if value in {"none", "auto", "0"}:
        return None

    if "," in value:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError("kernel_size must be one integer or two comma-separated integers")
        ky, kx = int(parts[0]), int(parts[1])
    else:
        ky = kx = int(value)

    if ky <= 0 or kx <= 0:
        raise ValueError("kernel_size must be positive")

    return ky, kx


def normalize_valid_pixels(
    band: np.ndarray,
    valid: np.ndarray,
    p_low: float,
    p_high: float,
) -> tuple[np.ndarray, float, float]:
    """
    Percentile-normalize valid pixels to [0, 1].
    Invalid pixels are set to 0 in the normalized array.
    """
    vals = band[valid].astype(np.float32)

    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid percentile range for this band")

    norm = np.zeros(band.shape, dtype=np.float32)
    clipped = np.clip(band.astype(np.float32), lo, hi)
    norm[valid] = (clipped[valid] - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)

    return norm, lo, hi


def apply_clahe_skimage_band(
    band: np.ndarray,
    nodata: Optional[float],
    clip_limit: float,
    kernel_size: Optional[Tuple[int, int]],
    p_low: float,
    p_high: float,
    output_scale: str,
) -> np.ndarray:
    """
    Apply skimage CLAHE to one band.

    output_scale:
    - "original": map enhanced [0,1] back to percentile value range.
    - "unit": keep enhanced values in [0,1].
    """
    band_float = band.astype(np.float32, copy=False)
    valid = np.isfinite(band_float)

    if nodata is not None and np.isfinite(nodata):
        valid &= band_float != nodata

    if int(valid.sum()) == 0:
        return np.full(band.shape, nodata if nodata is not None else np.nan, dtype=np.float32)

    try:
        norm, lo, hi = normalize_valid_pixels(
            band_float,
            valid,
            p_low=p_low,
            p_high=p_high,
        )
    except ValueError:
        out = band_float.astype(np.float32).copy()
        out[~valid] = nodata if nodata is not None else np.nan
        return out

    enhanced = exposure.equalize_adapthist(
        norm,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    ).astype(np.float32)

    out = np.empty(band.shape, dtype=np.float32)

    if output_scale == "unit":
        out[:] = np.nan
        out[valid] = enhanced[valid]
    elif output_scale == "original":
        out[:] = np.nan
        out[valid] = enhanced[valid] * (hi - lo) + lo
    else:
        raise ValueError(f"Unknown output_scale: {output_scale}")

    if nodata is not None and np.isfinite(nodata):
        out[~valid] = nodata
    else:
        out[~valid] = np.nan

    return out.astype(np.float32)


def process_one_geotiff(
    input_path: Path,
    output_path: Path,
    clip_limit: float,
    kernel_size: Optional[Tuple[int, int]],
    p_low: float,
    p_high: float,
    output_scale: str,
    suffix: str,
    compress: str,
) -> None:
    with rasterio.open(input_path) as src:
        data = src.read()
        profile = src.profile.copy()
        nodata = src.nodata
        colorinterp = src.colorinterp
        descriptions = src.descriptions
        tags = src.tags()
        band_tags = [src.tags(i + 1) for i in range(src.count)]

    out = np.empty(data.shape, dtype=np.float32)

    for b in range(data.shape[0]):
        out[b] = apply_clahe_skimage_band(
            data[b],
            nodata=nodata,
            clip_limit=clip_limit,
            kernel_size=kernel_size,
            p_low=p_low,
            p_high=p_high,
            output_scale=output_scale,
        )

    profile.update(
        dtype="float32",
        count=out.shape[0],
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    if nodata is not None and np.isfinite(nodata):
        profile.update(nodata=float(nodata))
    else:
        profile.update(nodata=np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out)
        try:
            dst.colorinterp = colorinterp
        except Exception:
            pass
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                if desc is not None:
                    dst.set_band_description(i, desc)
        if tags:
            dst.update_tags(**tags)
        for i, bt in enumerate(band_tags, start=1):
            if bt:
                dst.update_tags(i, **bt)


def build_output_path(
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
    preserve_structure: bool,
    suffix: str,
) -> Path:
    if preserve_structure:
        rel = input_file.relative_to(input_dir)
        parent = output_dir / rel.parent
    else:
        parent = output_dir

    return parent / f"{input_file.stem}{suffix}{input_file.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply skimage CLAHE to GeoTIFF images in a folder."
    )

    parser.add_argument("--input_dir", required=True, help="Input folder with GeoTIFF images.")
    parser.add_argument("--output_dir", required=True, help="Output folder for CLAHE GeoTIFFs.")

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folder recursively."
    )
    parser.add_argument(
        "--no_preserve_structure",
        action="store_true",
        help="Do not preserve subfolder structure in output_dir."
    )

    parser.add_argument(
        "--suffix",
        default="_clahe",
        help="Suffix added to output filenames before extension. Default: _clahe"
    )

    parser.add_argument(
        "--clip_limit",
        type=float,
        default=0.01,
        help="CLAHE clipping limit for skimage equalize_adapthist. Default: 0.01"
    )
    parser.add_argument(
        "--kernel_size",
        default="128",
        help="CLAHE kernel size: integer, 'height,width', or 'none'. Default: 128"
    )
    parser.add_argument(
        "--p_low",
        type=float,
        default=2.0,
        help="Lower percentile for normalization before CLAHE. Default: 2"
    )
    parser.add_argument(
        "--p_high",
        type=float,
        default=98.0,
        help="Upper percentile for normalization before CLAHE. Default: 98"
    )
    parser.add_argument(
        "--output_scale",
        choices=["original", "unit"],
        default="original",
        help="Save enhanced data in original percentile range or [0,1]. Default: original"
    )
    parser.add_argument(
        "--compress",
        default="lzw",
        help="GeoTIFF compression. Default: lzw"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not (0 <= args.p_low < args.p_high <= 100):
        raise ValueError("Require 0 <= p_low < p_high <= 100")

    kernel_size = parse_kernel_size(args.kernel_size)
    preserve_structure = not args.no_preserve_structure

    files = find_geotiffs(input_dir, recursive=args.recursive)
    print(f"Found {len(files)} GeoTIFF file(s).")

    if len(files) == 0:
        return

    for idx, input_file in enumerate(files, start=1):
        output_file = build_output_path(
            input_file=input_file,
            input_dir=input_dir,
            output_dir=output_dir,
            preserve_structure=preserve_structure,
            suffix=args.suffix,
        )

        if output_file.exists() and not args.overwrite:
            print(f"[{idx}/{len(files)}] Skip existing: {output_file}")
            continue

        print(f"[{idx}/{len(files)}] {input_file} -> {output_file}")

        try:
            process_one_geotiff(
                input_path=input_file,
                output_path=output_file,
                clip_limit=args.clip_limit,
                kernel_size=kernel_size,
                p_low=args.p_low,
                p_high=args.p_high,
                output_scale=args.output_scale,
                suffix=args.suffix,
                compress=args.compress,
            )
        except Exception as exc:
            print(f"  ERROR processing {input_file}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
