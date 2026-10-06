"""Scene-first splitting, on-demand GeoTIFF tiles and seeded augmentation."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm
from .augment import augment
from .raster import (check_alignment, windows, read_image, read_labels, normalize_image, pad)
from .schema import IGNORE_INDEX


@dataclass
class Tile:
    scene_id: str
    image_path: str
    mask_path: str
    col: int
    row: int
    width: int
    height: int
    foreground_fraction: float

    def window(self):
        return rasterio.windows.Window(self.col,self.row,self.width,self.height)


def split_scenes(pairs, fraction: float, seed: int, split_file: str | None = None):
    keys = {p[0] for p in pairs}
    if split_file:
        with open(split_file,encoding="utf-8") as f:
            split = json.load(f)
        tr, va = set(split["train"]), set(split["val"])
        if not tr or not va or tr & va or (tr | va)-keys:
            raise ValueError("Split JSON needs nonempty, disjoint train/val lists of known relative scene IDs.")
        # Other scenes may deliberately be reserved for a final held-out test set.
    else:
        if len(pairs) < 2:
            raise ValueError("Scene split needs at least two scenes. Supply independent validation scenes; tile split is an explicit, leakage-prone fallback.")
        order = np.random.default_rng(seed).permutation(len(pairs))
        nv = min(len(pairs)-1, max(1, int(np.ceil(len(pairs)*fraction))))
        va = {pairs[i][0] for i in order[:nv]}
        tr = keys-va
    return [p for p in pairs if p[0] in tr], [p for p in pairs if p[0] in va]


def index_tiles(pairs, cfg, training: bool):
    schema = cfg.schema()
    result = []
    for key, image_path, mask_path in tqdm(pairs,desc="Index train" if training else "Index validation"):
        with rasterio.open(image_path) as src, rasterio.open(mask_path) as ms:
            check_alignment(src,ms)
            if ms.nodata in schema.values:
                warnings.warn(f"{ms.name}: declared nodata={ms.nodata} is a class code. Class values take priority over nodata-derived masks; explicit GDAL masks still apply.")
            for win in windows(src.height,src.width,cfg.tile_size,cfg.stride):
                image, valid = read_image(src,win,cfg.in_channels,cfg.zero_is_nodata)
                # Even min_valid_fraction=0 must NEVER admit an entirely empty tile.
                if not valid.any() or valid.mean() < cfg.min_valid_fraction:
                    continue
                target = read_labels(ms,win,schema,valid)
                supervised = target != IGNORE_INDEX
                if not supervised.any():
                    continue
                positive = float(np.count_nonzero((target>0)&supervised)/supervised.sum())
                if training and positive < cfg.min_positive_fraction:
                    continue
                result.append(Tile(key,str(image_path),str(mask_path),int(win.col_off),int(win.row_off),
                                   int(win.width),int(win.height),positive))
    if training and cfg.max_tiles is not None and len(result)>cfg.max_tiles:
        indices = np.random.default_rng(cfg.seed).choice(len(result),cfg.max_tiles,replace=False)
        result = [result[i] for i in sorted(indices)]
    if not result:
        raise ValueError("No usable tiles after filtering; check image validity, labels and filter thresholds.")
    return result


class EddyDataset(Dataset):
    def __init__(self, tiles, cfg, training=False):
        self.tiles=list(tiles)
        self.cfg=cfg
        self.schema=cfg.schema()
        self.training=training
        self.epoch=0

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self,index):
        ti=self.tiles[index]
        with rasterio.open(ti.image_path) as src:
            image,valid=read_image(src,ti.window(),self.cfg.in_channels,self.cfg.zero_is_nodata)
        with rasterio.open(ti.mask_path) as src:
            target=read_labels(src,ti.window(),self.schema,valid)
        image=normalize_image(image,valid,self.cfg.normalize)
        image=pad(image,self.cfg.tile_size)
        valid=pad(valid,self.cfg.tile_size,False)
        target=pad(target,self.cfg.tile_size,IGNORE_INDEX)
        if self.training and not self.cfg.no_augment:
            # Epoch+tile deterministic stream; worker scheduling does not alter it.
            rng=np.random.default_rng(np.random.SeedSequence([self.cfg.seed,self.epoch,index]))
            image,target,valid=augment(image,target,valid,self.cfg.augment_level,
                                       self.cfg.normalize,rng,self.cfg.allow_flips)
        return torch.from_numpy(np.ascontiguousarray(image)),torch.from_numpy(np.ascontiguousarray(target))
