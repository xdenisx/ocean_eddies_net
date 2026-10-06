#!/usr/bin/env python3
"""
Train ocean eddy segmentation models from GeoTIFF images and masks.

Key features
------------
- Reads image GeoTIFFs from one folder and mask GeoTIFFs from another folder.
- Pairs files by basename, accepting .tif and .tiff.
- Supports 1-channel images now and multi-channel images later.
- Tiles images/masks on the fly, default 512 x 512.
- Skips image tiles with no valid pixels: all zero and/or NaN.
- Uses mask value 255 as eddy foreground; everything else is background.
- Optional eddy-aware sampling using --min_positive_fraction.
- Training augmentations for remote-sensing segmentation.
- BatchNorm-safe training: train drop_last=True by default and optional --freeze_batchnorm.
- Multiple architectures:
    transunet
    segformer_b0, segformer_b2, segformer_b4, segformer_b5
    unetpp_effb4, unetpp_effb5
    deeplabv3plus_resnet50, deeplabv3plus_resnet101
    upernet_swin_t, upernet_swin_s
    hrnet_w18, hrnet_w32
- Multiple losses:
    ce, dice, dice_ce, focal, dice_focal, tversky, focal_tversky

Notes
-----
For advanced architectures this script uses optional dependencies:
- segmentation_models_pytorch for U-Net++, DeepLabV3+, and SegFormer if available.
- transformers for SegFormer fallback and UPerNet-Swin if available.
- timm for HRNet-like fallback if available.

Install recommended dependencies:
    pip install torch torchvision rasterio albumentations opencv-python scikit-learn tqdm
    pip install segmentation-models-pytorch transformers timm

Example
-------
python train_ocean_eddy_segmentation.py \
    --images_dir images \
    --masks_dir masks \
    --output_dir output_unetpp \
    --architecture unetpp_effb4 \
    --loss dice_focal \
    --in_channels 1 \
    --num_classes 2 \
    --tile_size 512 \
    --stride 512 \
    --min_positive_fraction 0.001 \
    --batch_size 2 \
    --epochs 50 \
    --freeze_batchnorm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import albumentations as A
except Exception:
    A = None

try:
    import cv2
except Exception:
    cv2 = None


# -----------------------------
# Utilities
# -----------------------------

IMAGE_EXTENSIONS = {".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def list_geotiffs(folder: str | Path) -> Dict[str, Path]:
    folder = Path(folder)
    files: Dict[str, Path] = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files[p.stem] = p
    return files


def pair_image_masks(images_dir: str | Path, masks_dir: str | Path) -> List[Tuple[Path, Path]]:
    images = list_geotiffs(images_dir)
    masks = list_geotiffs(masks_dir)
    common = sorted(set(images) & set(masks))
    pairs = [(images[k], masks[k]) for k in common]
    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found. Images: {images_dir}; masks: {masks_dir}. "
            "Files are paired by basename, e.g. scene01.tif and scene01.tif."
        )
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks:
        print(f"Warning: {len(missing_masks)} image files have no mask match.")
    if missing_images:
        print(f"Warning: {len(missing_images)} mask files have no image match.")
    return pairs


def normalize_image(tile: np.ndarray, mode: str = "percentile") -> np.ndarray:
    """Normalize C,H,W float image tile."""
    tile = tile.astype(np.float32, copy=False)
    valid = np.isfinite(tile) & (tile != 0)
    tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "none":
        return tile

    out = np.zeros_like(tile, dtype=np.float32)
    for c in range(tile.shape[0]):
        band = tile[c]
        v = valid[c]
        if not np.any(v):
            continue
        if mode == "zscore":
            mean = float(np.mean(band[v]))
            std = float(np.std(band[v]))
            if std < 1e-6:
                std = 1.0
            out[c] = (band - mean) / std
        elif mode == "minmax":
            lo = float(np.min(band[v]))
            hi = float(np.max(band[v]))
            if hi <= lo:
                out[c] = 0.0
            else:
                out[c] = (band - lo) / (hi - lo)
        else:  # percentile
            lo, hi = np.percentile(band[v], [2, 98])
            if hi <= lo:
                out[c] = 0.0
            else:
                out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def pad_to_tile(arr: np.ndarray, tile_size: int, is_mask: bool = False) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
        pad_h = max(0, tile_size - h)
        pad_w = max(0, tile_size - w)
        if pad_h == 0 and pad_w == 0:
            return arr
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    c, h, w = arr.shape
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


@dataclass
class TileIndex:
    image_path: str
    mask_path: str
    row: int
    col: int
    height: int
    width: int
    positive_fraction: float


class EddyTileDataset(Dataset):
    def __init__(
        self,
        tile_indices: Sequence[TileIndex],
        in_channels: int,
        tile_size: int,
        mask_value: int = 255,
        normalize: str = "percentile",
        augment: bool = False,
        augment_level: str = "medium",
    ):
        self.tile_indices = list(tile_indices)
        self.in_channels = in_channels
        self.tile_size = tile_size
        self.mask_value = mask_value
        self.normalize = normalize
        self.augment = augment
        self.augment_level = augment_level
        self.transform = self._build_transform(augment_level) if augment else None

    def _build_transform(self, level: str = "medium"):
        if A is None:
            print("Warning: albumentations is not installed; augmentation disabled.")
            return None

        level = level.lower()
        if level not in {"light", "medium", "heavy"}:
            raise ValueError(f"Unknown augment_level: {level}")

        # Light: mostly geometry-safe orientation changes.
        # Medium: original-style augmentation, a bit stronger.
        # Heavy: stronger but still physically reasonable for ocean eddies.
        if level == "light":
            flip_p = 0.5
            rotate90_p = 0.5
            shift_limit = 0.03
            scale_limit = 0.05
            rotate_limit = 15
            shift_p = 0.25
            bc_limit = 0.10
            bc_p = 0.25
            gamma_limit = (90, 110)
            gamma_p = 0.15
            noise_p = 0.10
            blur_p = 0.05
            dropout_p = 0.05
        elif level == "medium":
            flip_p = 0.6
            rotate90_p = 0.6
            shift_limit = 0.05
            scale_limit = 0.10
            rotate_limit = 30
            shift_p = 0.45
            bc_limit = 0.18
            bc_p = 0.40
            gamma_limit = (80, 125)
            gamma_p = 0.30
            noise_p = 0.25
            blur_p = 0.15
            dropout_p = 0.10
        else:  # heavy
            flip_p = 0.7
            rotate90_p = 0.7
            shift_limit = 0.08
            scale_limit = 0.15
            rotate_limit = 45
            shift_p = 0.60
            bc_limit = 0.25
            bc_p = 0.60
            gamma_limit = (70, 140)
            gamma_p = 0.40
            noise_p = 0.40
            blur_p = 0.30
            dropout_p = 0.15

        transforms = [
            A.HorizontalFlip(p=flip_p),
            A.VerticalFlip(p=flip_p),
            A.RandomRotate90(p=rotate90_p),
        ]

        if cv2 is not None:
            transforms.extend(
                [
                    A.Rotate(
                        limit=180 if level == "heavy" else rotate_limit,
                        border_mode=cv2.BORDER_REFLECT_101,
                        value=0,
                        mask_value=0,
                        p=0.35 if level == "light" else (0.55 if level == "medium" else 0.70),
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=shift_limit,
                        scale_limit=scale_limit,
                        rotate_limit=rotate_limit,
                        border_mode=cv2.BORDER_REFLECT_101,
                        value=0,
                        mask_value=0,
                        p=shift_p,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 5 if level != "heavy" else 7), p=1.0),
                            A.MotionBlur(blur_limit=5 if level != "heavy" else 7, p=1.0),
                        ],
                        p=blur_p,
                    ),
                ]
            )

        transforms.extend(
            [
                A.RandomBrightnessContrast(brightness_limit=bc_limit, contrast_limit=bc_limit, p=bc_p),
                A.RandomGamma(gamma_limit=gamma_limit, p=gamma_p),
                A.GaussNoise(var_limit=(1e-5, 2e-3 if level != "heavy" else 5e-3), p=noise_p),
                A.CoarseDropout(
                    max_holes=8 if level != "heavy" else 12,
                    max_height=24 if level != "heavy" else 40,
                    max_width=24 if level != "heavy" else 40,
                    fill_value=0,
                    mask_fill_value=0,
                    p=dropout_p,
                ),
            ]
        )

        return A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.tile_indices)

    def __getitem__(self, idx: int):
        ti = self.tile_indices[idx]
        window = rasterio.windows.Window(ti.col, ti.row, ti.width, ti.height)

        with rasterio.open(ti.image_path) as src:
            count = src.count
            if self.in_channels <= count:
                indexes = list(range(1, self.in_channels + 1))
                img = src.read(indexes, window=window, boundless=True, fill_value=0).astype(np.float32)
            else:
                img0 = src.read(window=window, boundless=True, fill_value=0).astype(np.float32)
                if img0.ndim == 2:
                    img0 = img0[None]
                if img0.shape[0] < self.in_channels:
                    pad = np.zeros((self.in_channels - img0.shape[0], img0.shape[1], img0.shape[2]), dtype=np.float32)
                    img = np.concatenate([img0, pad], axis=0)
                else:
                    img = img0[: self.in_channels]

        with rasterio.open(ti.mask_path) as src:
            mask = src.read(1, window=window, boundless=True, fill_value=0)

        img = pad_to_tile(img, self.tile_size)
        mask = pad_to_tile(mask, self.tile_size)

        mask = (mask == self.mask_value).astype(np.uint8)
        img = normalize_image(img, self.normalize)

        # Albumentations expects H,W,C
        if self.transform is not None:
            img_hwc = np.moveaxis(img, 0, -1)
            aug = self.transform(image=img_hwc, mask=mask)
            img = np.moveaxis(aug["image"].astype(np.float32), -1, 0)
            mask = aug["mask"].astype(np.uint8)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


def build_tile_index(
    pairs: Sequence[Tuple[Path, Path]],
    tile_size: int,
    stride: int,
    mask_value: int,
    min_valid_fraction: float,
    min_positive_fraction: float,
    max_tiles: Optional[int] = None,
) -> List[TileIndex]:
    indices: List[TileIndex] = []
    for image_path, mask_path in tqdm(pairs, desc="Indexing tiles"):
        with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
            h, w = img_src.height, img_src.width
            if mask_src.height != h or mask_src.width != w:
                raise ValueError(
                    f"Image/mask size mismatch for {image_path.name}: "
                    f"image={w}x{h}, mask={mask_src.width}x{mask_src.height}"
                )

            rows = list(range(0, max(h - tile_size + 1, 1), stride))
            cols = list(range(0, max(w - tile_size + 1, 1), stride))
            if not rows or rows[-1] != max(h - tile_size, 0):
                rows.append(max(h - tile_size, 0))
            if not cols or cols[-1] != max(w - tile_size, 0):
                cols.append(max(w - tile_size, 0))

            for row in rows:
                for col in cols:
                    height = min(tile_size, h - row)
                    width = min(tile_size, w - col)
                    window = rasterio.windows.Window(col, row, width, height)

                    # Read first image band only for fast validity check.
                    img = img_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
                    valid = np.isfinite(img) & (img != 0)
                    valid_fraction = float(np.mean(valid))
                    if valid_fraction < min_valid_fraction:
                        continue

                    mask = mask_src.read(1, window=window, boundless=True, fill_value=0)
                    positive_fraction = float(np.mean(mask == mask_value))
                    if positive_fraction < min_positive_fraction:
                        continue

                    indices.append(
                        TileIndex(
                            image_path=str(image_path),
                            mask_path=str(mask_path),
                            row=int(row),
                            col=int(col),
                            height=int(height),
                            width=int(width),
                            positive_fraction=positive_fraction,
                        )
                    )
                    if max_tiles is not None and len(indices) >= max_tiles:
                        return indices
    return indices


# -----------------------------
# Losses and metrics
# -----------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        target_f = (target == 1).float()
        dims = (1, 2)
        intersection = torch.sum(probs * target_f, dims)
        denom = torch.sum(probs, dims) + torch.sum(target_f, dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        target_f = (target == 1).float()
        dims = (1, 2)
        tp = torch.sum(probs * target_f, dims)
        fp = torch.sum(probs * (1.0 - target_f), dims)
        fn = torch.sum((1.0 - probs) * target_f, dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.tversky(logits, target) ** self.gamma


class CombinedLoss(nn.Module):
    def __init__(self, kind: str, focal_alpha: float, focal_gamma: float, tversky_alpha: float, tversky_beta: float):
        super().__init__()
        self.kind = kind
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal_tversky = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "ce":
            return self.ce(logits, target)
        if self.kind == "dice":
            return self.dice(logits, target)
        if self.kind == "dice_ce":
            return 0.5 * self.ce(logits, target) + 0.5 * self.dice(logits, target)
        if self.kind == "focal":
            return self.focal(logits, target)
        if self.kind == "dice_focal":
            return 0.5 * self.dice(logits, target) + 0.5 * self.focal(logits, target)
        if self.kind == "tversky":
            return self.tversky(logits, target)
        if self.kind == "focal_tversky":
            return self.focal_tversky(logits, target)
        raise ValueError(f"Unknown loss: {self.kind}")


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    target = target.long()
    tp = torch.sum((pred == 1) & (target == 1)).item()
    fp = torch.sum((pred == 1) & (target == 0)).item()
    fn = torch.sum((pred == 0) & (target == 1)).item()
    tn = torch.sum((pred == 0) & (target == 0)).item()
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "accuracy": acc}


# -----------------------------
# Models
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleTransUNet(nn.Module):
    """Lightweight TransUNet-like fallback with one extra decoder layer."""
    def __init__(self, in_channels: int, num_classes: int, base: int = 48, heads: int = 4, layers: int = 2):
        super().__init__()
        self.enc1 = ConvBNReLU(in_channels, base)
        self.enc2 = ConvBNReLU(base, base * 2)
        self.enc3 = ConvBNReLU(base * 2, base * 4)
        self.enc4 = ConvBNReLU(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        enc_layer = nn.TransformerEncoderLayer(d_model=base * 8, nhead=heads, dim_feedforward=base * 16, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBNReLU(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBNReLU(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBNReLU(base * 2, base)
        # Extra refinement decoder layer requested in original TransUNet version.
        self.refine = ConvBNReLU(base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b, c, h, w = e4.shape
        tokens = e4.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        e4 = tokens.transpose(1, 2).reshape(b, c, h, w)
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.refine(d1)
        return self.out(d1)


class SMPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class HFSegFormerWrapper(nn.Module):
    def __init__(self, model_name: str, in_channels: int, num_classes: int):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        config = SegformerConfig.from_pretrained(model_name)
        config.num_channels = in_channels
        config.num_labels = num_classes
        config.id2label = {0: "background", 1: "eddy"}
        config.label2id = {"background": 0, "eddy": 1}
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        out = self.model(pixel_values=x).logits
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class HFUPerNetWrapper(nn.Module):
    def __init__(self, model_name: str, in_channels: int, num_classes: int):
        super().__init__()
        from transformers import UperNetConfig, UperNetForSemanticSegmentation
        config = UperNetConfig.from_pretrained(model_name)
        # Backbone configs usually expect 3 channels. For non-3 channels, we insert adapter.
        self.adapter = nn.Identity() if in_channels == 3 else nn.Conv2d(in_channels, 3, 1)
        config.num_labels = num_classes
        self.model = UperNetForSemanticSegmentation(config)

    def forward(self, x):
        x0 = self.adapter(x)
        out = self.model(pixel_values=x0).logits
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class TimmHRNetSeg(nn.Module):
    def __init__(self, backbone_name: str, in_channels: int, num_classes: int):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True, in_chans=in_channels)
        chs = self.backbone.feature_info.channels()
        self.proj = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in chs])
        self.head = nn.Sequential(
            ConvBNReLU(128 * len(chs), 256),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        size = x.shape[-2:]
        ups = []
        for f, p in zip(feats, self.proj):
            y = p(f)
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            ups.append(y)
        return self.head(torch.cat(ups, dim=1))


class SAMLikeUNet(nn.Module):
    """SAM-style lightweight substitute: strong encoder-decoder with attention bottleneck.

    This is not full SAM fine-tuning, but gives a SAM-like image encoder/decoder option
    without requiring prompt engineering or huge GPU memory.
    """
    def __init__(self, in_channels: int, num_classes: int, base: int = 64):
        super().__init__()
        self.net = SimpleTransUNet(in_channels, num_classes, base=base, heads=8, layers=4)

    def forward(self, x):
        return self.net(x)


def build_model(architecture: str, in_channels: int, num_classes: int) -> nn.Module:
    arch = architecture.lower()

    if arch == "transunet":
        return SimpleTransUNet(in_channels, num_classes)

    segformer_map = {
        "segformer": "nvidia/segformer-b2-finetuned-ade-512-512",
        "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "segformer_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "segformer_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }

    # Prefer segmentation_models_pytorch when available for some models.
    try:
        import segmentation_models_pytorch as smp
        if arch in {"unetpp_effb4", "unetpp_effb5"}:
            enc = "timm-efficientnet-b4" if arch.endswith("b4") else "timm-efficientnet-b5"
            return SMPWrapper(smp.UnetPlusPlus(encoder_name=enc, encoder_weights="imagenet", in_channels=in_channels, classes=num_classes))
        if arch in {"deeplabv3plus_resnet50", "deeplabv3plus_resnet101"}:
            enc = "resnet50" if arch.endswith("50") else "resnet101"
            return SMPWrapper(smp.DeepLabV3Plus(encoder_name=enc, encoder_weights="imagenet", in_channels=in_channels, classes=num_classes))
        if arch in segformer_map and hasattr(smp, "Segformer"):
            enc_name = {
                "segformer": "mit_b2",
                "segformer_b0": "mit_b0",
                "segformer_b2": "mit_b2",
                "segformer_b4": "mit_b4",
                "segformer_b5": "mit_b5",
            }[arch]
            return SMPWrapper(smp.Segformer(encoder_name=enc_name, encoder_weights="imagenet", in_channels=in_channels, classes=num_classes))
    except Exception as e:
        print(f"segmentation_models_pytorch unavailable or failed for {arch}: {e}")

    if arch in segformer_map:
        return HFSegFormerWrapper(segformer_map[arch], in_channels, num_classes)

    if arch in {"upernet_swin_t", "upernet_swin_s"}:
        model_name = "openmmlab/upernet-swin-tiny" if arch.endswith("_t") else "openmmlab/upernet-swin-small"
        try:
            return HFUPerNetWrapper(model_name, in_channels, num_classes)
        except Exception as e:
            raise RuntimeError(
                f"Could not build {arch}. Install transformers and check model availability. Original error: {e}"
            )

    if arch in {"hrnet_w18", "hrnet_w32"}:
        backbone = "hrnet_w18" if arch.endswith("18") else "hrnet_w32"
        try:
            return TimmHRNetSeg(backbone, in_channels, num_classes)
        except Exception as e:
            raise RuntimeError(
                f"Could not build {arch}. Install timm. If timm lacks this HRNet name, try hrnet_w18 first. Original error: {e}"
            )

    if arch == "sam_vit_unet":
        return SAMLikeUNet(in_channels, num_classes)

    raise ValueError(f"Unknown architecture: {architecture}")




def freeze_batchnorm_layers(model: nn.Module) -> None:
    """Keep BatchNorm statistics fixed during training.

    This is useful for DeepLabV3+/ResNet models when GPU memory forces
    batch_size=1, or when the deepest feature map becomes 1 x 1.
    BatchNorm layers are put into eval mode, while the rest of the model
    remains trainable.
    """
    for module in model.modules():
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp: bool, grad_clip: float, freeze_batchnorm: bool = False) -> Dict[str, float]:
    model.train()
    if freeze_batchnorm:
        freeze_batchnorm_layers(model)
    total_loss = 0.0
    totals = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    n = 0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        m = compute_metrics(logits.detach(), masks)
        for k in totals:
            totals[k] += m[k] * bs
        n += bs
    out = {"loss": total_loss / max(n, 1)}
    out.update({k: v / max(n, 1) for k, v in totals.items()})
    return out


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    totals = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    n = 0
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        m = compute_metrics(logits, masks)
        for k in totals:
            totals[k] += m[k] * bs
        n += bs
    out = {"loss": total_loss / max(n, 1)}
    out.update({k: v / max(n, 1) for k, v in totals.items()})
    return out


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_iou: float, args, train_stats, val_stats) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_iou": best_iou,
        "args": vars(args),
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Train ocean eddy segmentation from GeoTIFF images and masks.")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--architecture",
        default="unetpp_effb4",
        choices=[
            "transunet",
            "segformer", "segformer_b0", "segformer_b2", "segformer_b4", "segformer_b5",
            "unetpp_effb4", "unetpp_effb5",
            "deeplabv3plus_resnet50", "deeplabv3plus_resnet101",
            "upernet_swin_t", "upernet_swin_s",
            "hrnet_w18", "hrnet_w32",
            "sam_vit_unet",
        ],
    )
    parser.add_argument("--loss", default="dice_focal", choices=["ce", "dice", "dice_ce", "focal", "dice_focal", "tversky", "focal_tversky"])

    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--mask_value", type=int, default=255)
    parser.add_argument("--min_valid_fraction", type=float, default=0.01, help="Minimum fraction of nonzero finite image pixels in a tile.")
    parser.add_argument("--min_positive_fraction", type=float, default=0.0, help="Minimum fraction of mask pixels equal to mask_value. Try 0.0005 to 0.005 for eddy-aware training.")
    parser.add_argument("--normalize", default="percentile", choices=["percentile", "zscore", "minmax", "none"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tiles", type=int, default=None)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument(
        "--augment_level",
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Training augmentation strength. Ignored when --no_augment is used.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience by validation IoU. <=0 disables.")
    parser.add_argument(
        "--freeze_batchnorm",
        action="store_true",
        help=(
            "Freeze BatchNorm layers and keep them in eval mode during training. "
            "Useful for DeepLabV3+/ResNet when batch_size=1 or deepest feature maps become 1x1."
        ),
    )
    parser.add_argument(
        "--no_drop_last",
        action="store_true",
        help=(
            "Do not drop the last incomplete training batch. By default the script uses "
            "drop_last=True for training to avoid BatchNorm errors when the final batch has size 1."
        ),
    )

    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--tversky_alpha", type=float, default=0.3, help="Penalty for false positives.")
    parser.add_argument("--tversky_beta", type=float, default=0.7, help="Penalty for false negatives.")

    args = parser.parse_args()

    if args.num_classes != 2:
        raise ValueError("This script currently expects binary segmentation with --num_classes 2.")

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Pairing GeoTIFF images and masks...")
    pairs = pair_image_masks(args.images_dir, args.masks_dir)
    print(f"Found {len(pairs)} paired scenes.")

    print("Building tile index...")
    tile_indices = build_tile_index(
        pairs=pairs,
        tile_size=args.tile_size,
        stride=args.stride,
        mask_value=args.mask_value,
        min_valid_fraction=args.min_valid_fraction,
        min_positive_fraction=args.min_positive_fraction,
        max_tiles=args.max_tiles,
    )
    if not tile_indices:
        raise RuntimeError(
            "No tiles left after filtering. Lower --min_valid_fraction or --min_positive_fraction."
        )

    pos_fracs = np.array([t.positive_fraction for t in tile_indices], dtype=np.float32)
    print(f"Tiles: {len(tile_indices)}")
    print(f"Positive tile fraction stats: mean={pos_fracs.mean():.6f}, median={np.median(pos_fracs):.6f}, max={pos_fracs.max():.6f}")

    train_idx, val_idx = train_test_split(
        tile_indices,
        test_size=args.val_fraction,
        random_state=args.seed,
        shuffle=True,
    )
    print(f"Train tiles: {len(train_idx)} | Val tiles: {len(val_idx)}")

    if args.batch_size <= 1 and not args.freeze_batchnorm:
        print(
            "Warning: batch_size <= 1 with trainable BatchNorm can fail for DeepLab/ResNet-like models. "
            "If you see 'Expected more than 1 value per channel', rerun with --freeze_batchnorm."
        )
    if len(train_idx) < args.batch_size and not args.no_drop_last:
        raise RuntimeError(
            "Training set has fewer tiles than batch_size and drop_last=True would remove all batches. "
            "Use a smaller --batch_size or add --no_drop_last."
        )

    with open(out_dir / "tile_index_summary.json", "w") as f:
        json.dump(
            {
                "num_pairs": len(pairs),
                "num_tiles": len(tile_indices),
                "num_train_tiles": len(train_idx),
                "num_val_tiles": len(val_idx),
                "positive_fraction_mean": float(pos_fracs.mean()),
                "positive_fraction_median": float(np.median(pos_fracs)),
                "positive_fraction_max": float(pos_fracs.max()),
                "args": vars(args),
            },
            f,
            indent=2,
        )

    train_ds = EddyTileDataset(
        train_idx,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        mask_value=args.mask_value,
        normalize=args.normalize,
        augment=not args.no_augment,
        augment_level=args.augment_level,
    )
    val_ds = EddyTileDataset(
        val_idx,
        in_channels=args.in_channels,
        tile_size=args.tile_size,
        mask_value=args.mask_value,
        normalize=args.normalize,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=not args.no_drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Building model: {args.architecture}")
    model = build_model(args.architecture, args.in_channels, args.num_classes).to(device)
    if args.freeze_batchnorm:
        freeze_batchnorm_layers(model)
        print("BatchNorm layers frozen and kept in eval mode.")

    criterion = CombinedLoss(
        args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = GradScaler(enabled=args.amp)

    best_iou = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, args.amp, args.grad_clip, args.freeze_batchnorm)
        val_stats = validate(model, val_loader, criterion, device, args.amp)
        scheduler.step()

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)

        print(
            f"train loss={train_stats['loss']:.4f}, IoU={train_stats['iou']:.4f}, Dice={train_stats['dice']:.4f} | "
            f"val loss={val_stats['loss']:.4f}, IoU={val_stats['iou']:.4f}, Dice={val_stats['dice']:.4f}, "
            f"P={val_stats['precision']:.4f}, R={val_stats['recall']:.4f}"
        )

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(out_dir / "last_checkpoint.pt", model, optimizer, epoch, best_iou, args, train_stats, val_stats)

        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            epochs_without_improvement = 0
            save_checkpoint(out_dir / "best_checkpoint.pt", model, optimizer, epoch, best_iou, args, train_stats, val_stats)
            print(f"Saved best checkpoint with val IoU={best_iou:.4f}")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation IoU improvement.")
                break

    print("\nTraining complete.")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
