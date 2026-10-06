#!/usr/bin/env python3
"""
Train a TransUNet-style or SegFormer-style semantic segmentation model for ocean eddy recognition
from GeoTIFF optical images and annotated raster masks.

Main features
-------------
- Reads images from one folder and masks from another folder.
- Pairs files by basename, ignoring extension.
- Supports 1-channel images now and multi-channel GeoTIFFs later.
- Tiles images/masks into 512 x 512 patches on the fly.
- Skips tiles where the image contains no real pixels:
    all pixels are 0 and/or NaN across all channels.
- Trains either a TransUNet-style CNN + Transformer + U-Net decoder model or a SegFormer-style model.
- TransUNet includes one additional decoder layer compared with a common 4-level U-Net decoder.
- SegFormer uses hierarchical transformer features and an MLP decoder.
- Saves best checkpoint by validation loss.

Example
-------
python train_transunet_eddies.py \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --output_dir /path/to/output \
    --tile_size 512 \
    --stride 512 \
    --in_channels 1 \
    --num_classes 2 \
    --epochs 50 \
    --batch_size 4

Mask convention
---------------
For binary segmentation with --num_classes 2:
    mask values equal to --foreground_value are treated as eddy class 1, others as background 0.
    Default: --foreground_value 255.

For multiclass segmentation with --num_classes > 2:
    mask values are used directly as class IDs: 0, 1, 2, ... num_classes-1.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


IMAGE_EXTS = {".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_geotiffs(folder: str) -> Dict[str, Path]:
    folder_path = Path(folder)
    files = {}
    for p in folder_path.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            files[p.stem] = p
    return files


def pair_images_and_masks(images_dir: str, masks_dir: str) -> List[Tuple[Path, Path]]:
    images = list_geotiffs(images_dir)
    masks = list_geotiffs(masks_dir)

    common = sorted(set(images.keys()) & set(masks.keys()))
    pairs = [(images[k], masks[k]) for k in common]

    if not pairs:
        raise RuntimeError(
            "No image/mask pairs found. Files are paired by basename, e.g. image_001.tif and image_001.tif."
        )

    missing_masks = sorted(set(images.keys()) - set(masks.keys()))
    missing_images = sorted(set(masks.keys()) - set(images.keys()))

    if missing_masks:
        print(f"Warning: {len(missing_masks)} images have no matching mask.")
    if missing_images:
        print(f"Warning: {len(missing_images)} masks have no matching image.")

    return pairs


def window_is_valid_image_tile(
    image_path: Path,
    x: int,
    y: int,
    tile_size: int,
    in_channels: Optional[int],
    valid_pixel_fraction: float,
) -> bool:
    with rasterio.open(image_path) as src:
        channels = src.count if in_channels is None else min(in_channels, src.count)
        window = rasterio.windows.Window(x, y, tile_size, tile_size)
        arr = src.read(indexes=list(range(1, channels + 1)), window=window).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    finite = np.isfinite(arr)
    nonzero = arr != 0
    valid = finite & nonzero

    # A pixel is real if at least one channel is finite and non-zero.
    valid_pixel_map = np.any(valid, axis=0)
    frac = float(valid_pixel_map.mean())
    return frac >= valid_pixel_fraction


def build_tile_index(
    pairs: Sequence[Tuple[Path, Path]],
    tile_size: int,
    stride: int,
    in_channels: Optional[int],
    valid_pixel_fraction: float,
    max_tiles_per_image: Optional[int] = None,
) -> List[Tuple[Path, Path, int, int]]:
    tile_index = []

    iterator = pairs
    if tqdm is not None:
        iterator = tqdm(pairs, desc="Indexing valid tiles")

    for image_path, mask_path in iterator:
        with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
            if src_img.width != src_mask.width or src_img.height != src_mask.height:
                raise RuntimeError(
                    f"Image and mask sizes differ:\n  {image_path}\n  {mask_path}\n"
                    f"Image: {src_img.width}x{src_img.height}, mask: {src_mask.width}x{src_mask.height}"
                )
            width, height = src_img.width, src_img.height

        image_tiles = []
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                if window_is_valid_image_tile(
                    image_path=image_path,
                    x=x,
                    y=y,
                    tile_size=tile_size,
                    in_channels=in_channels,
                    valid_pixel_fraction=valid_pixel_fraction,
                ):
                    image_tiles.append((image_path, mask_path, x, y))

        if max_tiles_per_image is not None and len(image_tiles) > max_tiles_per_image:
            image_tiles = random.sample(image_tiles, max_tiles_per_image)

        tile_index.extend(image_tiles)

    if not tile_index:
        raise RuntimeError("No valid tiles found. Try lowering --valid_pixel_fraction or checking your input values.")

    return tile_index


class EddyTileDataset(Dataset):
    def __init__(
        self,
        tile_index: Sequence[Tuple[Path, Path, int, int]],
        tile_size: int = 512,
        in_channels: Optional[int] = None,
        num_classes: int = 2,
        normalize: str = "robust",
        augment: bool = False,
        foreground_value: int = 255,
    ):
        self.tile_index = list(tile_index)
        self.tile_size = tile_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.normalize = normalize
        self.augment = augment
        self.foreground_value = foreground_value

    def __len__(self) -> int:
        return len(self.tile_index)

    def _read_image(self, path: Path, x: int, y: int) -> np.ndarray:
        with rasterio.open(path) as src:
            channels = src.count if self.in_channels is None else min(self.in_channels, src.count)
            window = rasterio.windows.Window(x, y, self.tile_size, self.tile_size)
            arr = src.read(indexes=list(range(1, channels + 1)), window=window).astype(np.float32)
            nodata = src.nodata

        if nodata is not None:
            arr[arr == nodata] = np.nan

        arr[~np.isfinite(arr)] = 0.0
        arr = self._normalize_image(arr)
        return arr

    def _normalize_image(self, arr: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return arr

        out = arr.copy()
        for c in range(out.shape[0]):
            band = out[c]
            valid = np.isfinite(band) & (band != 0)
            if not np.any(valid):
                out[c] = 0
                continue

            if self.normalize == "standard":
                mean = float(np.mean(band[valid]))
                std = float(np.std(band[valid]))
                if std < 1e-6:
                    std = 1.0
                out[c] = (band - mean) / std
            elif self.normalize == "robust":
                p2, p98 = np.percentile(band[valid], [2, 98])
                if abs(p98 - p2) < 1e-6:
                    out[c] = 0
                else:
                    out[c] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                raise ValueError(f"Unknown normalization mode: {self.normalize}")
        return out

    def _read_mask(self, path: Path, x: int, y: int) -> np.ndarray:
        with rasterio.open(path) as src:
            window = rasterio.windows.Window(x, y, self.tile_size, self.tile_size)
            mask = src.read(1, window=window)
            nodata = src.nodata

        if nodata is not None:
            mask = np.where(mask == nodata, 0, mask)

        if self.num_classes == 2:
            mask = (mask == self.foreground_value).astype(np.int64)
        else:
            mask = mask.astype(np.int64)
            mask = np.clip(mask, 0, self.num_classes - 1)

        return mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Geometry-preserving augmentations for segmentation masks.
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()

        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Small integer shift with reflective image padding and constant background mask padding.
        if random.random() < 0.35:
            max_shift = max(1, int(0.05 * self.tile_size))
            dy = random.randint(-max_shift, max_shift)
            dx = random.randint(-max_shift, max_shift)
            image = np.stack([np.roll(ch, shift=(dy, dx), axis=(0, 1)) for ch in image], axis=0)
            mask = np.roll(mask, shift=(dy, dx), axis=(0, 1))
            if dy > 0:
                image[:, :dy, :] = 0; mask[:dy, :] = 0
            elif dy < 0:
                image[:, dy:, :] = 0; mask[dy:, :] = 0
            if dx > 0:
                image[:, :, :dx] = 0; mask[:, :dx] = 0
            elif dx < 0:
                image[:, :, dx:] = 0; mask[:, dx:] = 0

        # Radiometric augmentations only for images.
        if random.random() < 0.4:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.08, 0.08)
            image = image * gain + bias

        if random.random() < 0.25:
            gamma = random.uniform(0.8, 1.25)
            img_min = image.min(axis=(1, 2), keepdims=True)
            img_max = image.max(axis=(1, 2), keepdims=True)
            denom = np.maximum(img_max - img_min, 1e-6)
            image01 = np.clip((image - img_min) / denom, 0, 1)
            image = np.power(image01, gamma) * denom + img_min

        if random.random() < 0.25:
            noise_std = random.uniform(0.005, 0.03)
            image = image + np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)

        # Future multi-channel robustness: randomly remove one channel when more than one exists.
        if image.shape[0] > 1 and random.random() < 0.15:
            c = random.randrange(image.shape[0])
            image[c] = 0

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return image, mask

    def __getitem__(self, idx: int):
        image_path, mask_path, x, y = self.tile_index[idx]
        image = self._read_image(image_path, x, y)
        mask = self._read_mask(mask_path, x, y)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 2),
            ConvBNReLU(out_ch, out_ch, 3, 1),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class TransUNetExtraLayer(nn.Module):
    """
    Compact TransUNet-style model:
    - CNN stem extracts hierarchical features.
    - Transformer encoder operates on the bottleneck tokens.
    - U-Net decoder reconstructs full-resolution mask.
    - Extra decoder layer refines the final full-resolution feature map.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        embed_dim: int = 512,
        transformer_depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, c1, 3, 1),
            ConvBNReLU(c1, c1, 3, 1),
        )
        self.enc1 = EncoderBlock(c1, c2)   # 1/2
        self.enc2 = EncoderBlock(c2, c3)   # 1/4
        self.enc3 = EncoderBlock(c3, c4)   # 1/8
        self.enc4 = EncoderBlock(c4, c4)   # 1/16

        self.to_embed = nn.Conv2d(c4, embed_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.from_embed = nn.Conv2d(embed_dim, c4, kernel_size=1)

        self.dec4 = DecoderBlock(c4, c4, c4)  # 1/16 -> 1/8
        self.dec3 = DecoderBlock(c4, c3, c3)  # 1/8 -> 1/4
        self.dec2 = DecoderBlock(c3, c2, c2)  # 1/4 -> 1/2
        self.dec1 = DecoderBlock(c2, c1, c1)  # 1/2 -> 1/1

        # Additional layer requested: full-resolution refinement after normal decoder.
        self.extra_refine = nn.Sequential(
            ConvBNReLU(c1, c1, 3, 1),
            ConvBNReLU(c1, c1, 3, 1),
        )

        self.out = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)      # full
        s1 = self.enc1(s0)     # 1/2
        s2 = self.enc2(s1)     # 1/4
        s3 = self.enc3(s2)     # 1/8
        x = self.enc4(s3)      # 1/16

        b, _, h, w = x.shape
        x = self.to_embed(x)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(b, -1, h, w)
        x = self.from_embed(x)

        x = self.dec4(x, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = self.extra_refine(x)
        return self.out(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h: int, w: int):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_img = x.transpose(1, 2).reshape(b, c, h, w)
            x_reduced = self.sr(x_img).reshape(b, c, -1).transpose(1, 2)
            x_reduced = self.norm(x_reduced)
        else:
            x_reduced = x

        kv = self.kv(x_reduced).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return self.proj_drop(out)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h: int, w: int):
        b, n, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SegFormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sr_ratio: int, drop: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio=sr_ratio, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, h: int, w: int):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class SegFormerEddy(nn.Module):
    """
    SegFormer-style hierarchical transformer for eddy segmentation.
    This is implemented locally, so it does not require Hugging Face or internet access.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dims: Tuple[int, int, int, int] = (64, 128, 320, 512),
        depths: Tuple[int, int, int, int] = (3, 4, 6, 3),
        num_heads: Tuple[int, int, int, int] = (1, 2, 5, 8),
        sr_ratios: Tuple[int, int, int, int] = (8, 4, 2, 1),
        decoder_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_channels, embed_dims[0], patch_size=7, stride=4),
            OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2),
            OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2),
            OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2),
        ])

        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        cur = 0
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(4):
            stage = nn.ModuleList([
                SegFormerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratios[i],
                    drop=dropout,
                    drop_path=dpr[cur + j],
                )
                for j in range(depths[i])
            ])
            self.blocks.append(stage)
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            cur += depths[i]

        self.decode_projs = nn.ModuleList([nn.Conv2d(dim, decoder_dim, kernel_size=1) for dim in embed_dims])
        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_dim * 4, decoder_dim, kernel_size=1),
            ConvBNReLU(decoder_dim, decoder_dim, kernel_size=3),
        )
        self.out = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = []
        for patch_embed, blocks, norm in zip(self.patch_embeds, self.blocks, self.norms):
            tokens, h, w = patch_embed(x)
            for blk in blocks:
                tokens = blk(tokens, h, w)
            tokens = norm(tokens)
            b, _, c = tokens.shape
            x = tokens.transpose(1, 2).reshape(b, c, h, w)
            features.append(x)

        target_size = features[0].shape[-2:]
        decoded = []
        for feat, proj in zip(features, self.decode_projs):
            d = proj(feat)
            if d.shape[-2:] != target_size:
                d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
            decoded.append(d)

        x = self.fuse(torch.cat(decoded, dim=1))
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.out(x)


def parse_int_tuple(text: str, expected_len: int, name: str) -> Tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated integers. Got: {text}")
    return values


SEGFORMER_PRESETS = {
    # Lightweight, good for quick experiments / smaller GPUs.
    "segformer_b0": {
        "embed_dims": (32, 64, 160, 256),
        "depths": (2, 2, 2, 2),
        "heads": (1, 2, 5, 8),
        "decoder_dim": 128,
        "drop_path_rate": 0.1,
    },
    # Balanced default. Similar capacity to SegFormer-B2 style.
    "segformer": {
        "embed_dims": (64, 128, 320, 512),
        "depths": (3, 4, 6, 3),
        "heads": (1, 2, 5, 8),
        "decoder_dim": 256,
        "drop_path_rate": 0.1,
    },
    "segformer_b2": {
        "embed_dims": (64, 128, 320, 512),
        "depths": (3, 4, 6, 3),
        "heads": (1, 2, 5, 8),
        "decoder_dim": 256,
        "drop_path_rate": 0.1,
    },
    # More complex model for robust learning. Requires more GPU memory.
    "segformer_b4": {
        "embed_dims": (64, 128, 320, 512),
        "depths": (3, 8, 27, 3),
        "heads": (1, 2, 5, 8),
        "decoder_dim": 512,
        "drop_path_rate": 0.2,
    },
    # Largest local implementation here. Use small batch size, usually 1-2.
    "segformer_b5": {
        "embed_dims": (64, 128, 320, 512),
        "depths": (3, 6, 40, 3),
        "heads": (1, 2, 5, 8),
        "decoder_dim": 768,
        "drop_path_rate": 0.3,
    },
}


def build_model(args) -> nn.Module:
    if args.architecture == "transunet":
        return TransUNetExtraLayer(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
            embed_dim=args.embed_dim,
            transformer_depth=args.transformer_depth,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
        )

    if args.architecture in SEGFORMER_PRESETS:
        preset = SEGFORMER_PRESETS[args.architecture]

        # Custom CLI values override the preset when explicitly provided.
        embed_dims = parse_int_tuple(args.segformer_embed_dims, 4, "--segformer_embed_dims") if args.segformer_embed_dims else preset["embed_dims"]
        depths = parse_int_tuple(args.segformer_depths, 4, "--segformer_depths") if args.segformer_depths else preset["depths"]
        heads = parse_int_tuple(args.segformer_heads, 4, "--segformer_heads") if args.segformer_heads else preset["heads"]
        sr_ratios = parse_int_tuple(args.segformer_sr_ratios, 4, "--segformer_sr_ratios")
        decoder_dim = args.segformer_decoder_dim if args.segformer_decoder_dim is not None else preset["decoder_dim"]
        drop_path_rate = args.segformer_drop_path_rate if args.segformer_drop_path_rate is not None else preset["drop_path_rate"]

        return SegFormerEddy(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=heads,
            sr_ratios=sr_ratios,
            decoder_dim=decoder_dim,
            mlp_ratio=args.segformer_mlp_ratio,
            dropout=args.dropout,
            drop_path_rate=drop_path_rate,
        )

    raise ValueError(f"Unknown architecture: {args.architecture}")


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_1hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_1hot, dims)
        cardinality = torch.sum(probs + target_1hot, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        # Exclude background from Dice when possible.
        if self.num_classes > 1:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        return (1.0 - self.dice_weight) * ce + self.dice_weight * dice


@torch.no_grad()
def compute_basic_metrics(logits, target, num_classes: int) -> Dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    metrics = {}

    if num_classes == 2:
        tp = torch.sum((pred == 1) & (target == 1)).float()
        fp = torch.sum((pred == 1) & (target == 0)).float()
        fn = torch.sum((pred == 0) & (target == 1)).float()
        tn = torch.sum((pred == 0) & (target == 0)).float()

        eps = 1e-6
        metrics["iou"] = (tp / (tp + fp + fn + eps)).item()
        metrics["precision"] = (tp / (tp + fp + eps)).item()
        metrics["recall"] = (tp / (tp + fn + eps)).item()
        metrics["accuracy"] = ((tp + tn) / (tp + tn + fp + fn + eps)).item()
    else:
        ious = []
        for cls in range(1, num_classes):
            tp = torch.sum((pred == cls) & (target == cls)).float()
            fp = torch.sum((pred == cls) & (target != cls)).float()
            fn = torch.sum((pred != cls) & (target == cls)).float()
            iou = tp / (tp + fp + fn + 1e-6)
            ious.append(iou.item())
        metrics["mean_iou_no_background"] = float(np.mean(ious)) if ious else 0.0
        metrics["accuracy"] = torch.mean((pred == target).float()).item()

    return metrics


def train_one_epoch(model, loader, optimizer, loss_fn, device, num_classes, scaler=None):
    model.train()
    total_loss = 0.0
    metric_sum = {}
    n_batches = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Train", leave=False)

    for images, masks in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        batch_metrics = compute_basic_metrics(logits.detach(), masks, num_classes)
        for k, v in batch_metrics.items():
            metric_sum[k] = metric_sum.get(k, 0.0) + v
        n_batches += 1

    avg = {f"train_{k}": v / max(n_batches, 1) for k, v in metric_sum.items()}
    avg["train_loss"] = total_loss / max(n_batches, 1)
    return avg


@torch.no_grad()
def validate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    metric_sum = {}
    n_batches = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Val", leave=False)

    for images, masks in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)

        total_loss += float(loss.item())
        batch_metrics = compute_basic_metrics(logits, masks, num_classes)
        for k, v in batch_metrics.items():
            metric_sum[k] = metric_sum.get(k, 0.0) + v
        n_batches += 1

    avg = {f"val_{k}": v / max(n_batches, 1) for k, v in metric_sum.items()}
    avg["val_loss"] = total_loss / max(n_batches, 1)
    return avg


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if not train_idx:
        train_idx = val_idx
    return train_idx, val_idx


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, best_val_loss: float) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Train TransUNet or SegFormer model for ocean eddy segmentation.")

    parser.add_argument("--images_dir", required=True, help="Folder with GeoTIFF images")
    parser.add_argument("--masks_dir", required=True, help="Folder with GeoTIFF masks")
    parser.add_argument("--output_dir", required=True, help="Output folder")

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--valid_pixel_fraction", type=float, default=0.01,
                        help="Minimum fraction of image pixels that must be finite and non-zero")
    parser.add_argument("--max_tiles_per_image", type=int, default=None)

    parser.add_argument("--in_channels", type=int, default=1,
                        help="Number of input channels to read. Use the number of raster bands you want to train with.")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--foreground_value", type=int, default=255,
                        help="For binary masks, pixels equal to this value are eddies. Default: 255")
    parser.add_argument(
        "--architecture",
        choices=["transunet", "segformer", "segformer_b0", "segformer_b2", "segformer_b4", "segformer_b5"],
        default="transunet",
        help="Model architecture. Use segformer_b4 or segformer_b5 for a more complex model."
    )

    # TransUNet parameters
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--transformer_depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # SegFormer parameters. Leave embed/depth/head/decoder/drop-path as None to use architecture preset.
    # You can still override them manually, for example: --segformer_depths 3,8,27,3
    parser.add_argument("--segformer_embed_dims", default=None)
    parser.add_argument("--segformer_depths", default=None)
    parser.add_argument("--segformer_heads", default=None)
    parser.add_argument("--segformer_sr_ratios", default="8,4,2,1")
    parser.add_argument("--segformer_decoder_dim", type=int, default=None)
    parser.add_argument("--segformer_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--segformer_drop_path_rate", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--normalize", choices=["robust", "standard", "none"], default="robust")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Pairing images and masks...")
    pairs = pair_images_and_masks(args.images_dir, args.masks_dir)
    print(f"Found pairs: {len(pairs)}")

    print("Building tile index and skipping empty image tiles...")
    tile_index = build_tile_index(
        pairs=pairs,
        tile_size=args.tile_size,
        stride=args.stride,
        in_channels=args.in_channels,
        valid_pixel_fraction=args.valid_pixel_fraction,
        max_tiles_per_image=args.max_tiles_per_image,
    )
    print(f"Valid tiles: {len(tile_index)}")

    with open(output_dir / "tile_index.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"image": str(i), "mask": str(m), "x": x, "y": y}
                for i, m, x, y in tile_index
            ],
            f,
            indent=2,
        )

    train_idx, val_idx = split_indices(len(tile_index), args.val_fraction, args.seed)
    print(f"Train tiles: {len(train_idx)}")
    print(f"Val tiles: {len(val_idx)}")

    full_train_dataset = EddyTileDataset(
        tile_index=tile_index,
        tile_size=args.tile_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        normalize=args.normalize,
        augment=not args.no_augment,
        foreground_value=args.foreground_value,
    )
    full_val_dataset = EddyTileDataset(
        tile_index=tile_index,
        tile_size=args.tile_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        normalize=args.normalize,
        augment=False,
        foreground_value=args.foreground_value,
    )

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_val_dataset, val_idx) if val_idx else Subset(full_val_dataset, train_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = build_model(args).to(device)
    print(f"Using architecture: {args.architecture}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = CombinedLoss(num_classes=args.num_classes, dice_weight=args.dice_weight)

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_val_loss = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(model, train_loader, optimizer, loss_fn, device, args.num_classes, scaler)
        val_stats = validate(model, val_loader, loss_fn, device, args.num_classes)
        scheduler.step()

        stats = {"epoch": epoch, **train_stats, **val_stats, "lr": scheduler.get_last_lr()[0]}
        history.append(stats)

        print(" | ".join([f"{k}: {v:.5f}" for k, v in stats.items() if k != "epoch"]))

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(output_dir / "last_checkpoint.pt", model, optimizer, epoch, args, best_val_loss)

        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            save_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, epoch, args, best_val_loss)
            print(f"Saved new best checkpoint: val_loss={best_val_loss:.5f}")

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best checkpoint: {output_dir / 'best_checkpoint.pt'}")


if __name__ == "__main__":
    main()
