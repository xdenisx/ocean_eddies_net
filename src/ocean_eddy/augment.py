"""Testable NumPy/OpenCV augmentation, without version-sensitive Albumentations APIs.

Geometry is shared by image, categorical mask and validity; only the image receives
radiometric changes. No elastic/perspective deformation or label-erasing dropout.
"""
from __future__ import annotations
import cv2
import numpy as np
from .schema import IGNORE_INDEX

PRESETS = {
    "light": dict(affine_p=.25, angle=15, shift=.03, scale=.05, bc_p=.25,
                  bc=.10, gamma_p=.15, gamma=(.90,1.10), noise_p=.10, noise=.015, blur_p=.05),
    "medium": dict(affine_p=.55, angle=45, shift=.05, scale=.10, bc_p=.40,
                   bc=.18, gamma_p=.30, gamma=(.80,1.25), noise_p=.25, noise=.035, blur_p=.15),
    "heavy": dict(affine_p=.70, angle=180, shift=.08, scale=.15, bc_p=.60,
                  bc=.25, gamma_p=.40, gamma=(.70,1.40), noise_p=.40, noise=.060, blur_p=.30),
}


def augment(image: np.ndarray, target: np.ndarray, valid: np.ndarray, level: str,
            mode: str, rng: np.random.Generator, allow_flips: bool = True):
    if level not in PRESETS:
        raise ValueError(f"Unknown augmentation level: {level}")
    p = PRESETS[level]
    # p=0.5 makes reflected and unreflected orientations equally likely. A higher
    # flip probability does not increase geometric diversity.
    if allow_flips:
        for axis in (0, 1):
            if rng.random() < .5:
                image = np.flip(image, axis=axis+1)
                target = np.flip(target, axis=axis)
                valid = np.flip(valid, axis=axis)
    k = int(rng.integers(4))
    image = np.rot90(image, k, axes=(1, 2)).copy()
    target = np.rot90(target, k).copy()
    valid = np.rot90(valid, k).copy()
    h, w = target.shape
    if rng.random() < p["affine_p"]:
        matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), rng.uniform(-p["angle"],p["angle"]),
                                       rng.uniform(1-p["scale"],1+p["scale"]))
        matrix[:, 2] += [rng.uniform(-p["shift"],p["shift"])*w,
                         rng.uniform(-p["shift"],p["shift"])*h]
        warp = lambda a, flags, fill: cv2.warpAffine(np.ascontiguousarray(a), matrix, (w,h),
                          flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        image = np.stack([warp(b, cv2.INTER_LINEAR, 0) for b in image])
        target = warp(target.astype(np.float32), cv2.INTER_NEAREST, IGNORE_INDEX).astype(np.int64)
        # Linear-support test excludes synthetic borders and interpolation across NoData.
        valid = warp(valid.astype(np.float32), cv2.INTER_LINEAR, 0) > .999
    if valid.any():
        if mode in {"percentile", "minmax"}:
            if rng.random() < p["bc_p"]:
                image = np.clip(image*rng.uniform(1-p["bc"],1+p["bc"]) + rng.uniform(-p["bc"],p["bc"]),0,1)
            if rng.random() < p["gamma_p"]:
                image = np.clip(image,0,1)**rng.uniform(*p["gamma"])
            if rng.random() < p["noise_p"]:
                image += rng.normal(0, rng.uniform(0,p["noise"]),image.shape).astype(np.float32)
            image = np.clip(image,0,1)
        elif mode == "zscore":
            # Gamma and [0,1] clipping are invalid for standardized signed data.
            if rng.random() < p["bc_p"]:
                image = image*rng.uniform(1-p["bc"],1+p["bc"]) + rng.uniform(-p["bc"],p["bc"])
            if rng.random() < p["noise_p"]:
                image += rng.normal(0,p["noise"],image.shape).astype(np.float32)
        # normalize=none: geometry + blur only; raw physical dynamic range is unknown.
        if rng.random() < p["blur_p"]:
            size = int(rng.choice([3,5,7] if level == "heavy" else [3,5]))
            support = cv2.GaussianBlur(valid.astype(np.float32),(size,size),0)
            image = np.stack([cv2.GaussianBlur(b*valid,(size,size),0) / np.maximum(support,1e-6) for b in image])
    image[:,~valid] = 0
    target[~valid] = IGNORE_INDEX
    return np.ascontiguousarray(image,np.float32), np.ascontiguousarray(target,np.int64), valid
