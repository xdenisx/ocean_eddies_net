#!/usr/bin/env python3
"""
Inference for TransUNet-style ocean eddy segmentation from GeoTIFF images.

Compatible with train_transunet_eddies.py checkpoints.

Main features
-------------
- Reads one GeoTIFF or a folder of GeoTIFFs recursively.
- Supports 1-channel images now and multi-channel GeoTIFFs later.
- Runs tiled inference with overlap blending.
- Skips fully empty tiles where all pixels are 0 and/or NaN.
- Saves georeferenced prediction GeoTIFFs.
- Optionally saves probability GeoTIFFs.

Example
-------
python infer_transunet_eddies.py \
    --input /path/to/images \
    --checkpoint /path/to/output/best_checkpoint.pt \
    --output_dir /path/to/predictions \
    --tile_size 512 \
    --stride 256 \
    --in_channels 1 \
    --num_classes 2 \
    --save_prob

For binary segmentation:
- prediction output is uint8 with 0=background, 1=eddy.
- probability output is float32 eddy probability if --num_classes 2.

For multiclass segmentation:
- prediction output is uint8/uint16 class ID map.
- probability output is a multiband float32 GeoTIFF with one band per class.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


IMAGE_EXTS = {".tif", ".tiff"}


def list_geotiffs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Input file is not a GeoTIFF: {p}")
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    files = sorted([x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS])
    if not files:
        raise RuntimeError(f"No .tif/.tiff files found in: {p}")
    return files


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
    Same model definition as train_transunet_eddies.py.
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
        self.enc1 = EncoderBlock(c1, c2)
        self.enc2 = EncoderBlock(c2, c3)
        self.enc3 = EncoderBlock(c3, c4)
        self.enc4 = EncoderBlock(c4, c4)

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

        self.dec4 = DecoderBlock(c4, c4, c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)

        self.extra_refine = nn.Sequential(
            ConvBNReLU(c1, c1, 3, 1),
            ConvBNReLU(c1, c1, 3, 1),
        )

        self.out = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)

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


def load_checkpoint_args(checkpoint_path: str) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt.get("args", {}) if isinstance(ckpt, dict) else {}


def load_model(checkpoint_path: str, args, device: torch.device) -> nn.Module:
    ckpt_args = load_checkpoint_args(checkpoint_path)

    in_channels = args.in_channels if args.in_channels is not None else int(ckpt_args.get("in_channels", 1))
    num_classes = args.num_classes if args.num_classes is not None else int(ckpt_args.get("num_classes", 2))
    base_channels = args.base_channels if args.base_channels is not None else int(ckpt_args.get("base_channels", 32))
    embed_dim = args.embed_dim if args.embed_dim is not None else int(ckpt_args.get("embed_dim", 512))
    transformer_depth = args.transformer_depth if args.transformer_depth is not None else int(ckpt_args.get("transformer_depth", 6))
    num_heads = args.num_heads if args.num_heads is not None else int(ckpt_args.get("num_heads", 8))
    mlp_dim = args.mlp_dim if args.mlp_dim is not None else int(ckpt_args.get("mlp_dim", 1024))
    dropout = args.dropout if args.dropout is not None else float(ckpt_args.get("dropout", 0.1))

    model = TransUNetExtraLayer(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        embed_dim=embed_dim,
        transformer_depth=transformer_depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    model.in_channels = in_channels
    model.num_classes = num_classes
    return model


def normalize_tile(arr: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "none":
        return arr

    out = arr.copy()
    for c in range(out.shape[0]):
        band = out[c]
        valid = np.isfinite(band) & (band != 0)
        if not np.any(valid):
            out[c] = 0
            continue

        if normalize == "standard":
            mean = float(np.mean(band[valid]))
            std = float(np.std(band[valid]))
            if std < 1e-6:
                std = 1.0
            out[c] = (band - mean) / std
        elif normalize == "robust":
            p2, p98 = np.percentile(band[valid], [2, 98])
            if abs(p98 - p2) < 1e-6:
                out[c] = 0
            else:
                out[c] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            raise ValueError(f"Unknown normalization mode: {normalize}")
    return out


def read_tile(src: rasterio.io.DatasetReader, x: int, y: int, tile_size: int, in_channels: int) -> Tuple[np.ndarray, int, int]:
    width = min(tile_size, src.width - x)
    height = min(tile_size, src.height - y)
    window = Window(x, y, width, height)

    channels = min(in_channels, src.count)
    arr = src.read(indexes=list(range(1, channels + 1)), window=window).astype(np.float32)

    if channels < in_channels:
        padded_channels = np.zeros((in_channels, height, width), dtype=np.float32)
        padded_channels[:channels] = arr
        arr = padded_channels

    nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid = np.any(np.isfinite(arr) & (arr != 0), axis=0)

    padded = np.zeros((in_channels, tile_size, tile_size), dtype=np.float32)
    padded[:, :height, :width] = arr
    padded[~np.isfinite(padded)] = 0.0

    return padded, height, width


def make_weight(tile_size: int, blend: str = "hann") -> np.ndarray:
    if blend == "flat":
        return np.ones((tile_size, tile_size), dtype=np.float32)

    one = np.hanning(tile_size)
    weight = np.outer(one, one).astype(np.float32)

    # Avoid zero weights at tile edges, especially for border tiles.
    weight = np.maximum(weight, 1e-3)
    return weight


def tile_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def predict_one_image(
    image_path: Path,
    output_dir: Path,
    model: nn.Module,
    device: torch.device,
    tile_size: int,
    stride: int,
    normalize: str,
    save_prob: bool,
    prob_dtype: str,
    blend: str,
    suffix: str,
    skip_empty_tiles: bool,
) -> None:
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        crs = src.crs
        transform = src.transform
        in_channels = int(model.in_channels)
        num_classes = int(model.num_classes)

        prob_sum = np.zeros((num_classes, height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        weight = make_weight(tile_size, blend=blend)

        xs = tile_positions(width, tile_size, stride)
        ys = tile_positions(height, tile_size, stride)
        positions = [(x, y) for y in ys for x in xs]

        iterator = positions
        if tqdm is not None:
            iterator = tqdm(positions, desc=f"Predict {image_path.name}", leave=False)

        for x, y in iterator:
            tile, real_h, real_w = read_tile(src, x, y, tile_size, in_channels)

            valid_pixel_map = np.any(np.isfinite(tile[:, :real_h, :real_w]) & (tile[:, :real_h, :real_w] != 0), axis=0)
            if skip_empty_tiles and not np.any(valid_pixel_map):
                continue

            tile = normalize_tile(tile, normalize=normalize)
            tensor = torch.from_numpy(tile[None]).float().to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs = probs[:, :real_h, :real_w]

            w = weight[:real_h, :real_w]
            prob_sum[:, y:y + real_h, x:x + real_w] += probs * w[None, :, :]
            weight_sum[y:y + real_h, x:x + real_w] += w

        safe_weight = np.maximum(weight_sum, 1e-6)
        probs_full = prob_sum / safe_weight[None, :, :]

        # If no tile contributed to a pixel, keep background probability as 1.
        empty = weight_sum <= 0
        if np.any(empty):
            probs_full[:, empty] = 0.0
            probs_full[0, empty] = 1.0

        pred = np.argmax(probs_full, axis=0)

        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / f"{image_path.stem}{suffix}.tif"

        pred_dtype = "uint8" if num_classes <= 255 else "uint16"
        pred_profile = profile.copy()
        pred_profile.update(
            driver="GTiff",
            count=1,
            dtype=pred_dtype,
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        # Remove photometric/color settings inherited from RGB files.
        for key in ["photometric", "interleave"]:
            pred_profile.pop(key, None)

        with rasterio.open(pred_path, "w", **pred_profile) as dst:
            dst.write(pred.astype(pred_dtype), 1)

        print(f"Saved prediction: {pred_path}")

        if save_prob:
            prob_path = output_dir / f"{image_path.stem}{suffix}_prob.tif"
            prob_profile = profile.copy()
            prob_profile.update(
                driver="GTiff",
                dtype=prob_dtype,
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            for key in ["photometric", "interleave", "nodata"]:
                prob_profile.pop(key, None)

            if num_classes == 2:
                prob_profile.update(count=1)
                prob_to_save = probs_full[1:2]
            else:
                prob_profile.update(count=num_classes)
                prob_to_save = probs_full

            if prob_dtype == "float32":
                prob_to_save = prob_to_save.astype(np.float32)
            elif prob_dtype == "uint8":
                prob_to_save = np.clip(prob_to_save * 255.0, 0, 255).astype(np.uint8)
            else:
                raise ValueError("--prob_dtype must be float32 or uint8")

            with rasterio.open(prob_path, "w", **prob_profile) as dst:
                dst.write(prob_to_save)

            print(f"Saved probability: {prob_path}")


def main():
    parser = argparse.ArgumentParser(description="Run TransUNet ocean eddy segmentation inference on GeoTIFF image(s).")

    parser.add_argument("--input", required=True, help="Input GeoTIFF file or folder with GeoTIFFs")
    parser.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt or last_checkpoint.pt")
    parser.add_argument("--output_dir", required=True, help="Folder for prediction GeoTIFFs")

    parser.add_argument("--tile_size", type=int, default=None, help="Tile size. Defaults to training value or 512.")
    parser.add_argument("--stride", type=int, default=None, help="Inference stride. Defaults to half tile size.")
    parser.add_argument("--in_channels", type=int, default=None, help="Override input channels. Defaults to checkpoint value.")
    parser.add_argument("--num_classes", type=int, default=None, help="Override class count. Defaults to checkpoint value.")

    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--transformer_depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--normalize", choices=["robust", "standard", "none"], default=None,
                        help="Normalization. Defaults to checkpoint value or robust.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_prob", action="store_true", help="Save probability GeoTIFF")
    parser.add_argument("--prob_dtype", choices=["float32", "uint8"], default="float32")
    parser.add_argument("--blend", choices=["hann", "flat"], default="hann", help="Overlap blending mode")
    parser.add_argument("--suffix", default="_pred", help="Output filename suffix before .tif")
    parser.add_argument("--no_skip_empty_tiles", action="store_true", help="Do not skip tiles that are all 0/NaN")

    args = parser.parse_args()

    ckpt_args = load_checkpoint_args(args.checkpoint)

    if args.tile_size is None:
        args.tile_size = int(ckpt_args.get("tile_size", 512))
    if args.stride is None:
        args.stride = max(1, args.tile_size // 2)
    if args.normalize is None:
        args.normalize = str(ckpt_args.get("normalize", "robust"))

    if args.stride > args.tile_size:
        raise ValueError("--stride should be <= --tile_size for full coverage")

    device = torch.device(args.device)
    model = load_model(args.checkpoint, args, device)

    print("Inference settings:")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  tile_size:  {args.tile_size}")
    print(f"  stride:     {args.stride}")
    print(f"  channels:   {model.in_channels}")
    print(f"  classes:    {model.num_classes}")
    print(f"  normalize:  {args.normalize}")
    print(f"  device:     {device}")

    files = list_geotiffs(args.input)
    output_dir = Path(args.output_dir)

    for image_path in files:
        predict_one_image(
            image_path=image_path,
            output_dir=output_dir,
            model=model,
            device=device,
            tile_size=args.tile_size,
            stride=args.stride,
            normalize=args.normalize,
            save_prob=args.save_prob,
            prob_dtype=args.prob_dtype,
            blend=args.blend,
            suffix=args.suffix,
            skip_empty_tiles=not args.no_skip_empty_tiles,
        )

    print("Inference finished.")


if __name__ == "__main__":
    main()
