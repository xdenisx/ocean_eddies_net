"""Minmax -> optional global HistEq -> skimage CLAHE -> cv2 uint8 minmax.

The requested chain is applied independently per band. Georeferencing is retained.
Global HistEq uses a valid-pixel mask; CLAHE has no mask argument and temporarily
fills invalid pixels from the nearest valid pixel before processing. This limits
black-padding effects but is not true masked CLAHE. Output restores original support.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from skimage import exposure
from tqdm.auto import tqdm
from .raster import list_rasters,read_image,write_raster


def enhance_band(band,valid,clip_limit=.03,kernel_size=None,histogram_first=True):
    result=np.zeros(band.shape,np.uint8)
    if not valid.any():
        return result
    values=band[valid].astype(np.float32)
    if float(values.max())<=float(values.min()):
        # No contrast can be inferred from a constant band. Keep a finite mid-gray.
        result[valid]=127
        return result
    normalized=np.zeros(band.shape,np.float32)
    normalized[valid]=cv2.normalize(values,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F).ravel()
    normalized=np.clip(normalized,0,1)
    hist=exposure.equalize_hist(normalized,mask=valid) if histogram_first else normalized
    if not valid.all():
        nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        hist=hist[tuple(nearest)]
    hist=exposure.equalize_adapthist(np.clip(hist,0,1),clip_limit=clip_limit,kernel_size=kernel_size)
    result[valid]=cv2.normalize(hist[valid].astype(np.float32),None,0,255,
                               cv2.NORM_MINMAX,dtype=cv2.CV_8U).ravel()
    return result


def preprocess(args):
    if not 0<args.clip_limit<=1 or args.kernel_size<0:
        raise ValueError("Require 0<clip_limit<=1 and kernel_size>=0 (0=automatic).")
    source=Path(args.input_dir).resolve(); output=Path(args.output_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source==output or source in output.parents:
        raise ValueError("output_dir must be outside input_dir.")
    files=list_rasters(source,args.recursive)
    if not files:
        raise ValueError("No GeoTIFFs found.")
    for path in tqdm(files,desc="HistEq + CLAHE"):
        rel=path.relative_to(source)
        destination=output/rel.parent/(rel.stem+args.suffix+".tif")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(destination)
        with rasterio.open(path) as src:
            image,valid=read_image(src,None,src.count,args.zero_is_nodata)
            out=np.stack([enhance_band(b,valid,args.clip_limit,args.kernel_size or None,
                                       not args.skip_equalize_hist) for b in image])
            write_raster(destination,out,src,valid,None,
                         [d or f"band_{i}" for i,d in enumerate(src.descriptions,1)],
                         {"processing":"cv2_minmax_01 + "+("" if args.skip_equalize_hist else "skimage_histEq + ")+"skimage_CLAHE + cv2_minmax_0255",
                          "clip_limit":args.clip_limit,"kernel_size":args.kernel_size,
                          "zero_is_valid_in_output":True,
                          "radiometry":"contrast-enhanced uint8; not original physical measurements"})
    print("Output uses an internal GDAL mask, not nodata=0. Train/infer with zero_is_nodata=false for these enhanced images.")
