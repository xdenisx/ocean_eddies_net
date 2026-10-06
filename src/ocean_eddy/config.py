"""Serializable configuration with unknown-key rejection."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import yaml
from .schema import make_schema


@dataclass
class TrainConfig:
    images_dir: str = "images"
    masks_dir: str = "masks"
    output_dir: str = "runs/experiment"
    architecture: str = "deeplabv3plus_resnet101"
    backend: str = "auto"
    pretrained: bool = True
    in_channels: int = 1
    class_values: tuple = (0, 255)
    class_names: tuple | None = None
    ignore_values: tuple = ()
    unknown_mask_policy: str = "error"
    tile_size: int = 512
    stride: int = 512
    min_valid_fraction: float = 0.01
    min_positive_fraction: float = 0.0
    zero_is_nodata: bool = True
    normalize: str = "percentile"
    split_mode: str = "scene"
    split_file: str | None = None
    val_fraction: float = 0.2
    max_tiles: int | None = None
    augment_level: str = "medium"
    no_augment: bool = False
    allow_flips: bool = True
    epochs: int = 100
    batch_size: int = 2
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    loss: str = "dice_focal"
    class_weights: tuple | None = None
    include_background_loss: bool = False
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_tversky_gamma: float = 1.33
    freeze_batchnorm: bool = False
    drop_last: bool = True
    amp: bool = False
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    patience: int = 15
    early_stop_monitor: str = "val_iou"
    min_delta: float = 0.0
    device: str = "auto"
    resume: str | None = None

    def schema(self):
        return make_schema(self.class_values, self.class_names, self.ignore_values,
                           self.unknown_mask_policy)

    def validate(self):
        schema = self.schema()
        for name in ("in_channels", "tile_size", "stride", "epochs", "batch_size", "accumulation_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.tile_size % 32:
            raise ValueError("tile_size must be a multiple of 32 for the supported model registry.")
        if self.stride > self.tile_size:
            raise ValueError("stride > tile_size would leave gaps.")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1.")
        for name in ("min_valid_fraction", "min_positive_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1].")
        if self.split_mode not in {"scene", "tile"}:
            raise ValueError("split_mode must be scene or tile.")
        if self.normalize not in {"percentile", "minmax", "zscore", "none"}:
            raise ValueError("Unknown normalize mode.")
        if self.augment_level not in {"light", "medium", "heavy"}:
            raise ValueError("augment_level must be light, medium or heavy.")
        if self.early_stop_monitor not in {"val_iou", "val_dice", "val_loss"}:
            raise ValueError("early_stop_monitor must be val_iou, val_dice or val_loss.")
        if self.num_workers < 0 or self.min_delta < 0 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid worker, delta, learning rate or weight decay setting.")
        if self.max_tiles is not None and self.max_tiles < 1:
            raise ValueError("max_tiles must be positive or null.")
        if self.class_weights is not None:
            if len(self.class_weights) != schema.count or any(w <= 0 or not math.isfinite(w) for w in self.class_weights):
                raise ValueError("class_weights must have one finite positive weight per class.")
        if not 0 < self.focal_alpha < 1 or self.focal_gamma < 0:
            raise ValueError("focal_alpha must be in (0,1); focal_gamma must be >=0.")
        if min(self.tversky_alpha, self.tversky_beta) < 0 or self.tversky_alpha + self.tversky_beta <= 0:
            raise ValueError("Invalid Tversky penalties.")
        if self.focal_tversky_gamma <= 0:
            raise ValueError("focal_tversky_gamma must be positive.")
        return self

    def to_dict(self):
        # JSON-compatible primitive values for safe torch.load(weights_only=True).
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict):
        allowed = {f.name for f in fields(cls)}
        extra = set(data) - allowed
        if extra:
            raise ValueError(f"Unknown configuration keys: {sorted(extra)}")
        return cls(**data).validate()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return result
