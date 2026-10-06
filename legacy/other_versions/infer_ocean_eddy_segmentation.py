#!/usr/bin/env python3
"""
Inference for ocean eddy segmentation from GeoTIFF images.

Compatible with checkpoints produced by train_ocean_eddy_segmentation.py.

Features
--------
- Reads one GeoTIFF or all GeoTIFFs in a folder/subfolders.
- Rebuilds model architecture from checkpoint metadata when available.
- Supports the same architecture names as the training script:
    transunet
    segformer, segformer_b0, segformer_b2, segformer_b4, segformer_b5
    unetpp_effb4, unetpp_effb5
    deeplabv3plus_resnet50, deeplabv3plus_resnet101
    upernet_swin_t, upernet_swin_s
    hrnet_w18, hrnet_w32
    sam_vit_unet
- Tiled inference with overlap blending.
- Saves georeferenced prediction masks using 0 = background and 255 = eddy.
- Optionally saves foreground probability GeoTIFFs.
- Optional simple test-time augmentation using flips.

Example
-------
python infer_ocean_eddy_segmentation.py \
    --input images \
    --checkpoint output_unetpp/best_checkpoint.pt \
    --output_dir predictions \
    --tile_size 512 \
    --stride 256 \
    --threshold 0.5 \
    --save_prob \
    --tta
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

IMAGE_EXTENSIONS = {".tif", ".tiff"}


# =============================================================================
# Utilities
# =============================================================================

def list_geotiffs(path: str | Path, recursive: bool = True) -> List[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Input file is not a GeoTIFF: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    iterator = path.rglob("*") if recursive else path.glob("*")
    files = sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if not files:
        raise RuntimeError(f"No GeoTIFF files found in {path}")
    return files


def normalize_image(tile: np.ndarray, mode: str = "percentile") -> np.ndarray:
    """Normalize C,H,W float image tile exactly like the training script."""
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
        else:
            lo, hi = np.percentile(band[v], [2, 98])
            if hi <= lo:
                out[c] = 0.0
            else:
                out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def pad_chw(arr: np.ndarray, tile_size: int) -> np.ndarray:
    c, h, w = arr.shape
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


def make_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def load_image_window(src: rasterio.io.DatasetReader, row: int, col: int, height: int, width: int, in_channels: int) -> np.ndarray:
    window = rasterio.windows.Window(col, row, width, height)
    count = src.count
    if in_channels <= count:
        indexes = list(range(1, in_channels + 1))
        img = src.read(indexes, window=window, boundless=True, fill_value=0).astype(np.float32)
    else:
        img0 = src.read(window=window, boundless=True, fill_value=0).astype(np.float32)
        if img0.ndim == 2:
            img0 = img0[None]
        if img0.shape[0] < in_channels:
            pad = np.zeros((in_channels - img0.shape[0], img0.shape[1], img0.shape[2]), dtype=np.float32)
            img = np.concatenate([img0, pad], axis=0)
        else:
            img = img0[:in_channels]
    return img


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


# =============================================================================
# Model definitions copied from training script for standalone inference
# =============================================================================

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
        self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True, in_chans=in_channels)
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

    try:
        import segmentation_models_pytorch as smp
        if arch in {"unetpp_effb4", "unetpp_effb5"}:
            enc = "timm-efficientnet-b4" if arch.endswith("b4") else "timm-efficientnet-b5"
            return SMPWrapper(smp.UnetPlusPlus(encoder_name=enc, encoder_weights=None, in_channels=in_channels, classes=num_classes))
        if arch in {"deeplabv3plus_resnet50", "deeplabv3plus_resnet101"}:
            enc = "resnet50" if arch.endswith("50") else "resnet101"
            return SMPWrapper(smp.DeepLabV3Plus(encoder_name=enc, encoder_weights=None, in_channels=in_channels, classes=num_classes))
        if arch in segformer_map and hasattr(smp, "Segformer"):
            enc_name = {
                "segformer": "mit_b2",
                "segformer_b0": "mit_b0",
                "segformer_b2": "mit_b2",
                "segformer_b4": "mit_b4",
                "segformer_b5": "mit_b5",
            }[arch]
            return SMPWrapper(smp.Segformer(encoder_name=enc_name, encoder_weights=None, in_channels=in_channels, classes=num_classes))
    except Exception as e:
        print(f"segmentation_models_pytorch unavailable or failed for {arch}: {e}")

    if arch in segformer_map:
        return HFSegFormerWrapper(segformer_map[arch], in_channels, num_classes)

    if arch in {"upernet_swin_t", "upernet_swin_s"}:
        model_name = "openmmlab/upernet-swin-tiny" if arch.endswith("_t") else "openmmlab/upernet-swin-small"
        return HFUPerNetWrapper(model_name, in_channels, num_classes)

    if arch in {"hrnet_w18", "hrnet_w32"}:
        backbone = "hrnet_w18" if arch.endswith("18") else "hrnet_w32"
        return TimmHRNetSeg(backbone, in_channels, num_classes)

    if arch == "sam_vit_unet":
        return SAMLikeUNet(in_channels, num_classes)

    raise ValueError(f"Unknown architecture: {architecture}")


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict_batch(model: nn.Module, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Return foreground probability B,H,W."""
    if not tta:
        logits = model(x)
        return torch.softmax(logits, dim=1)[:, 1]

    probs = []
    logits = model(x)
    probs.append(torch.softmax(logits, dim=1)[:, 1])

    x_h = torch.flip(x, dims=[3])
    p_h = torch.softmax(model(x_h), dim=1)[:, 1]
    probs.append(torch.flip(p_h, dims=[2]))

    x_v = torch.flip(x, dims=[2])
    p_v = torch.softmax(model(x_v), dim=1)[:, 1]
    probs.append(torch.flip(p_v, dims=[1]))

    x_hv = torch.flip(x, dims=[2, 3])
    p_hv = torch.softmax(model(x_hv), dim=1)[:, 1]
    probs.append(torch.flip(p_hv, dims=[1, 2]))

    return torch.stack(probs, dim=0).mean(dim=0)


@torch.no_grad()
def infer_one_image(
    image_path: Path,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    in_channels: int,
    tile_size: int,
    stride: int,
    normalize: str,
    threshold: float,
    save_prob: bool,
    batch_size: int,
    amp: bool,
    tta: bool,
    skip_empty_tiles: bool,
    min_valid_fraction: float,
    suffix: str,
) -> Tuple[Path, Optional[Path]]:
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        profile = src.profile.copy()
        rows = make_positions(h, tile_size, stride)
        cols = make_positions(w, tile_size, stride)

        prob_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        batch_tiles = []
        batch_windows = []

        def flush_batch():
            nonlocal batch_tiles, batch_windows, prob_sum, weight_sum
            if not batch_tiles:
                return
            x = torch.from_numpy(np.stack(batch_tiles, axis=0)).float().to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                probs = predict_batch(model, x, tta=tta)
            probs_np = probs.detach().cpu().numpy()
            for p, (row, col, height, width) in zip(probs_np, batch_windows):
                p = p[:height, :width]
                prob_sum[row:row + height, col:col + width] += p.astype(np.float32)
                weight_sum[row:row + height, col:col + width] += 1.0
            batch_tiles = []
            batch_windows = []

        total_tiles = len(rows) * len(cols)
        for row in tqdm(rows, desc=f"{image_path.name} rows", leave=False):
            for col in cols:
                height = min(tile_size, h - row)
                width = min(tile_size, w - col)
                img = load_image_window(src, row, col, height, width, in_channels)

                if skip_empty_tiles:
                    valid = np.isfinite(img[0]) & (img[0] != 0)
                    if float(np.mean(valid)) < min_valid_fraction:
                        continue

                img = pad_chw(img, tile_size)
                img = normalize_image(img, normalize)
                batch_tiles.append(img)
                batch_windows.append((row, col, height, width))

                if len(batch_tiles) >= batch_size:
                    flush_batch()

        flush_batch()

    prob = np.zeros((h, w), dtype=np.float32)
    valid_pred = weight_sum > 0
    prob[valid_pred] = prob_sum[valid_pred] / weight_sum[valid_pred]
    pred = (prob >= threshold).astype(np.uint8) * 255

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pred = output_dir / f"{image_path.stem}{suffix}.tif"

    pred_profile = profile.copy()
    pred_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    # Remove photometric interpretation/color metadata from source if incompatible.
    pred_profile.pop("photometric", None)

    with rasterio.open(out_pred, "w", **pred_profile) as dst:
        dst.write(pred, 1)
        try:
            dst.write_colormap(1, {0: (0, 0, 0, 255), 255: (255, 0, 0, 255)})
        except Exception:
            pass

    out_prob = None
    if save_prob:
        out_prob = output_dir / f"{image_path.stem}_prob.tif"
        prob_profile = profile.copy()
        prob_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=0.0,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        prob_profile.pop("photometric", None)
        with rasterio.open(out_prob, "w", **prob_profile) as dst:
            dst.write(prob.astype(np.float32), 1)

    return out_pred, out_prob


def main():
    parser = argparse.ArgumentParser(description="Run ocean eddy segmentation inference on GeoTIFF images.")
    parser.add_argument("--input", required=True, help="Input GeoTIFF file or folder with GeoTIFFs.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint from train_ocean_eddy_segmentation.py.")
    parser.add_argument("--output_dir", required=True, help="Output folder.")

    parser.add_argument("--architecture", default=None, help="Override architecture. Default: read from checkpoint.")
    parser.add_argument("--in_channels", type=int, default=None, help="Override input channels. Default: read from checkpoint.")
    parser.add_argument("--num_classes", type=int, default=None, help="Override number of classes. Default: read from checkpoint.")
    parser.add_argument("--tile_size", type=int, default=None, help="Tile size. Default: read from checkpoint or 512.")
    parser.add_argument("--stride", type=int, default=None, help="Inference stride. Default: tile_size // 2.")
    parser.add_argument("--normalize", choices=["percentile", "zscore", "minmax", "none"], default=None, help="Normalization. Default: read from checkpoint.")

    parser.add_argument("--threshold", type=float, default=0.5, help="Foreground probability threshold.")
    parser.add_argument("--batch_size", type=int, default=4, help="Tile batch size for inference.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--tta", action="store_true", help="Use flip test-time augmentation.")
    parser.add_argument("--save_prob", action="store_true", help="Save foreground probability GeoTIFF.")
    parser.add_argument("--recursive", action="store_true", help="Search input folder recursively.")
    parser.add_argument("--suffix", default="_pred", help="Suffix for prediction mask filenames.")

    parser.add_argument("--skip_empty_tiles", action="store_true", help="Skip tiles with too few valid pixels.")
    parser.add_argument("--min_valid_fraction", type=float, default=0.001, help="Minimum valid nonzero finite fraction when --skip_empty_tiles is used.")

    parser.add_argument("--device", default=None, help="cuda, cpu, or auto. Default: auto.")

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    architecture = args.architecture or ckpt.get("architecture") or ckpt.get("args", {}).get("architecture")
    in_channels = args.in_channels or ckpt.get("in_channels") or ckpt.get("args", {}).get("in_channels") or 1
    num_classes = args.num_classes or ckpt.get("num_classes") or ckpt.get("args", {}).get("num_classes") or 2
    tile_size = args.tile_size or ckpt.get("tile_size") or ckpt.get("args", {}).get("tile_size") or 512
    normalize = args.normalize or ckpt.get("normalize") or ckpt.get("args", {}).get("normalize") or "percentile"
    stride = args.stride or max(1, tile_size // 2)

    if architecture is None:
        raise RuntimeError("Architecture was not found in checkpoint. Please provide --architecture.")

    print(f"Architecture: {architecture}")
    print(f"Input channels: {in_channels}")
    print(f"Number of classes: {num_classes}")
    print(f"Tile size / stride: {tile_size} / {stride}")
    print(f"Normalize: {normalize}")
    print(f"Threshold: {args.threshold}")

    model = build_model(architecture, int(in_channels), int(num_classes)).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading checkpoint: {len(unexpected)}")
    model.eval()

    files = list_geotiffs(args.input, recursive=args.recursive)
    print(f"Found {len(files)} GeoTIFF file(s).")

    output_dir = Path(args.output_dir)
    for image_path in tqdm(files, desc="Images"):
        out_pred, out_prob = infer_one_image(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=output_dir,
            in_channels=int(in_channels),
            tile_size=int(tile_size),
            stride=int(stride),
            normalize=str(normalize),
            threshold=float(args.threshold),
            save_prob=bool(args.save_prob),
            batch_size=int(args.batch_size),
            amp=bool(args.amp and device.type == "cuda"),
            tta=bool(args.tta),
            skip_empty_tiles=bool(args.skip_empty_tiles),
            min_valid_fraction=float(args.min_valid_fraction),
            suffix=str(args.suffix),
        )
        print(f"Saved: {out_pred}")
        if out_prob is not None:
            print(f"Saved: {out_prob}")


if __name__ == "__main__":
    main()
