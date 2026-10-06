"""Shared raster IO, normalization, tiling and georeferencing checks."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.windows import Window
from .schema import ClassSchema, IGNORE_INDEX

EXTENSIONS = {".tif", ".tiff"}


def list_rasters(path: str | Path, recursive: bool = True) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Not a TIFF: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in EXTENSIONS)


def raster_map(folder: str | Path, suffix: str = "") -> dict[str, Path]:
    root = Path(folder)
    result = {}
    for p in list_rasters(root):
        rel = p.relative_to(root)
        stem = rel.stem
        if suffix:
            if not stem.endswith(suffix):
                continue
            stem = stem[:-len(suffix)]
        # Build key from parent+stem without stripping a second suffix in scene IDs.
        key = (rel.parent / stem).as_posix()
        if key in result:
            raise ValueError(f"Duplicate raster key {key!r}: {result[key]} and {p}")
        result[key] = p
    return result


def pair_rasters(images_dir: str | Path, masks_dir: str | Path) -> list[tuple[str, Path, Path]]:
    images, masks = raster_map(images_dir), raster_map(masks_dir)
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise ValueError("No pairs: match relative paths and stems between images_dir and masks_dir.")
    if images.keys() - masks.keys():
        raise ValueError(f"Images without masks: {sorted(images.keys() - masks.keys())[:12]}")
    if masks.keys() - images.keys():
        warnings.warn(f"{len(masks.keys() - images.keys())} masks have no corresponding image.")
    return [(k, images[k], masks[k]) for k in common]


def check_alignment(a, b):
    if (a.width, a.height) != (b.width, b.height):
        raise ValueError(f"Raster dimensions differ: {a.name} / {b.name}")
    if a.crs != b.crs or not a.transform.almost_equals(b.transform, precision=1e-7):
        raise ValueError(f"Raster CRS/transform mismatch: {a.name} / {b.name}; align masks using nearest-neighbor resampling first.")
    if b.count != 1:
        raise ValueError(f"Expected a single-band class-code mask: {b.name}")


def positions(length: int, tile_size: int, stride: int) -> list[int]:
    if min(length, tile_size, stride) <= 0 or stride > tile_size:
        raise ValueError("Require length>0 and 0<stride<=tile_size.")
    end = max(0, length - tile_size)
    result = list(range(0, end + 1, stride))
    if result[-1] != end:
        result.append(end)
    return result


def windows(height: int, width: int, tile_size: int, stride: int):
    for row in positions(height, tile_size, stride):
        for col in positions(width, tile_size, stride):
            yield Window(col, row, min(tile_size, width-col), min(tile_size, height-row))


def read_image(src, window: Window | None, in_channels: int, zero_is_nodata: bool = True):
    if src.count < in_channels:
        raise ValueError(f"{src.name} has {src.count} bands, expected at least {in_channels}; no silent channel padding.")
    indexes = list(range(1, in_channels + 1))
    image = src.read(indexes, window=window).astype(np.float32)
    valid = np.all(np.isfinite(image) & (src.read_masks(indexes, window=window) > 0), axis=0)
    if zero_is_nodata:
        valid &= np.any(image != 0, axis=0)
    image[:, ~valid] = 0
    return image, valid


def read_labels(src, window: Window | None, schema: ClassSchema, image_valid: np.ndarray):
    raw = src.read(1, window=window)
    support = src.read_masks(1, window=window) > 0
    # Some binary masks incorrectly declare their foreground (255) or background (0)
    # as NoData. Explicit class codes win over a nodata-derived mask, never over an
    # explicit internal/external GDAL validity mask. See docs/DATA_CONTRACT.md.
    if src.nodata in schema.values and MaskFlags.nodata in src.mask_flag_enums[0]:
        support |= raw == src.nodata
    return schema.encode(raw, image_valid & support)


def normalize_image(image: np.ndarray, valid: np.ndarray, mode: str):
    """Per-tile, per-band normalization; invalid support stays zero in every mode."""
    result = np.zeros(image.shape, dtype=np.float32)
    if not valid.any():
        return result
    for c, band in enumerate(image):
        vals = band[valid].astype(np.float32)
        if mode == "none":
            scaled = vals
        elif mode == "zscore":
            scaled = (vals - vals.mean()) / max(float(vals.std()), 1e-6)
        elif mode in {"percentile", "minmax"}:
            low, high = (np.percentile(vals, [2, 98]) if mode == "percentile"
                         else (vals.min(), vals.max()))
            scaled = np.clip((vals-low) / (high-low), 0, 1) if high > low else np.zeros_like(vals)
        else:
            raise ValueError(f"Unknown normalization: {mode}")
        result[c, valid] = scaled
    if not np.isfinite(result).all():
        raise ValueError("Normalization produced nonfinite values; inspect raster dynamic range.")
    return result


def pad(image: np.ndarray, size: int, fill=0):
    h, w = image.shape[-2:]
    if max(h, w) > size:
        raise ValueError("Cannot pad a raster larger than the target tile.")
    spec = [(0, 0)] * (image.ndim - 2) + [(0, size-h), (0, size-w)]
    return np.pad(image, spec, mode="constant", constant_values=fill)


def output_profile(src, count: int, dtype: str, nodata=None):
    # Fresh profile: old scale, palette, JPEG, NBITS and predictors must not leak.
    return dict(driver="GTiff", width=src.width, height=src.height, count=count,
                crs=src.crs, transform=src.transform, dtype=dtype, nodata=nodata,
                compress="deflate", predictor=3 if dtype == "float32" else 2,
                tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")


def write_raster(path: str | Path, data: np.ndarray, src, valid: np.ndarray,
                 nodata=None, descriptions=None, tags=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None]
    tmp = path.with_name(path.stem + ".partial.tif")
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(tmp, "w", **output_profile(src, data.shape[0], str(data.dtype), nodata)) as dst:
                dst.write(data)
                dst.write_mask(valid.astype(np.uint8) * 255)
                if descriptions:
                    dst.descriptions = tuple(descriptions)
                if tags:
                    dst.update_tags(**{k: json.dumps(v) if not isinstance(v, str) else v for k, v in tags.items()})
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
