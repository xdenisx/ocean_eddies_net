"""Explicit raster-code <-> contiguous model-index conversion."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
import numpy as np

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ClassSchema:
    values: tuple[int, ...] = (0, 255)
    names: tuple[str, ...] = ("background", "eddy")
    ignore_values: tuple[int, ...] = ()
    unknown_policy: str = "error"

    def __post_init__(self):
        if len(self.values) < 2 or len(self.values) != len(self.names):
            raise ValueError("Specify >=2 class_values and exactly one class_name per value.")
        if len(set(self.values)) != len(self.values) or len(set(self.names)) != len(self.names):
            raise ValueError("Class codes and names must be unique.")
        if any(not isinstance(v, (int, np.integer)) or v < 0 or v > 65534 for v in self.values):
            raise ValueError("Class values must be integers in 0..65534 (65535 is output NoData).")
        if set(self.values) & set(self.ignore_values):
            raise ValueError("ignore_values cannot also be class_values; 255 can be a class.")
        if self.unknown_policy not in {"error", "ignore", "background"}:
            raise ValueError("unknown_policy must be error, ignore or background.")

    @property
    def count(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ClassSchema":
        return cls(tuple(d["values"]), tuple(d["names"]),
                   tuple(d.get("ignore_values", [])), d.get("unknown_policy", "error"))

    def encode(self, raw: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """Nonfinite, explicitly ignored and invalid-support pixels -> IGNORE_INDEX."""
        if raw.ndim != 2:
            raise ValueError("Masks must be single-band integer class-code rasters, not RGB.")
        eligible = np.isfinite(raw)
        if valid is not None:
            eligible &= valid
        if self.ignore_values:
            eligible &= ~np.isin(raw, self.ignore_values)
        out = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
        known = np.zeros(raw.shape, dtype=bool)
        for index, code in enumerate(self.values):
            hit = eligible & (raw == code)
            out[hit] = index
            known |= hit
        unknown = eligible & ~known
        if unknown.any():
            if self.unknown_policy == "error":
                examples = np.unique(raw[unknown])[:12].tolist()
                raise ValueError(f"Unknown mask values {examples}; set class_values/ignore_values explicitly.")
            if self.unknown_policy == "background":
                out[unknown] = 0
        return out

    def decode(self, indices: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """Write original codes in uint16; 65535 is unambiguously NoData."""
        usable = (indices >= 0) & (indices < self.count)
        if valid is not None:
            usable &= valid
        out = np.full(indices.shape, 65535, dtype=np.uint16)
        out[usable] = np.asarray(self.values, np.uint16)[indices[usable]]
        return out


def make_schema(values: Sequence[int], names: Sequence[str] | None = None,
                ignore_values: Sequence[int] = (), unknown_policy: str = "error") -> ClassSchema:
    values = tuple(values)
    if names is None:
        names = ("background", "eddy") if values == (0, 255) else tuple(
            "background" if i == 0 else f"class_{i}" for i in range(len(values)))
    return ClassSchema(values, tuple(names), tuple(ignore_values), unknown_policy)
