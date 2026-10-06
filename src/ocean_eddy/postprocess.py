"""Center-weighted blending and class-aware, NoData-aware postprocessing."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def blend_weights(size,mode="distance",min_weight=.05,power=1.):
    if size<1 or not 0<min_weight<=1 or power<=0:
        raise ValueError("Invalid blend window parameters.")
    yy,xx=np.mgrid[:size,:size].astype(np.float32)
    if mode=="uniform":
        result=np.ones((size,size),np.float32)
    elif mode=="distance":
        result=np.minimum.reduce([yy+1,xx+1,size-yy,size-xx])
    elif mode=="hann":
        result=np.outer(np.hanning(size),np.hanning(size)).astype(np.float32)
    elif mode=="gaussian":
        result=np.exp(-.5*((yy-(size-1)/2)**2+(xx-(size-1)/2)**2)/(size/4)**2)
    else:
        raise ValueError(f"Unknown blend_mode {mode}")
    result=result/max(float(result.max()),1e-6)
    return np.maximum(result**power,min_weight).astype(np.float32)


def smooth_probabilities_inplace(probs,valid,sigma):
    """Mask-normalized Gaussian, band by band, followed by class renormalization."""
    if sigma<0:
        raise ValueError("smooth_sigma cannot be negative.")
    if sigma>0 and valid.any():
        denominator=ndi.gaussian_filter(valid.astype(np.float32),sigma,mode="constant",cval=0)
        for c in range(probs.shape[0]):
            values=np.where(valid,probs[c],0)
            numerator=ndi.gaussian_filter(values,sigma,mode="constant",cval=0)
            probs[c]=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    # Block-wise normalization avoids holding C*H*W copies in RAM.
    for row in range(0,probs.shape[1],512):
        block=probs[:,row:row+512,:]
        support=valid[row:row+512,:]
        block[:]=np.nan_to_num(block,nan=0,posinf=0,neginf=0)
        total=block.sum(axis=0)
        np.divide(block,total[None],out=block,where=total[None]>0)
        block[:,~support]=np.nan


def classify(probs,valid,threshold=None):
    k,h,w=probs.shape
    if threshold is not None and k!=2:
        raise ValueError("--threshold is binary-only; multiclass uses mutually exclusive argmax.")
    threshold=.5 if threshold is None else threshold
    if not 0<=threshold<=1:
        raise ValueError("threshold must be between 0 and 1.")
    result=np.zeros((h,w),np.int32)
    for row in range(0,h,512):
        block=probs[:,row:row+512]
        result[row:row+512]=(block[1]>=threshold if k==2 else np.argmax(block,axis=0))
    result[~valid]=-100
    return result


def clean_labels(labels,valid,num_classes,min_object_size=0,min_hole_size=0,
                 class_min_sizes=None,connectivity=8):
    """Remove small per-class components; fill only background-only enclosed holes.

    Returned labels remain mutually exclusive. Never overwrite another foreground
    class while filling holes, and never fill across NoData or image boundaries.
    class_min_sizes uses MODEL INDICES (the CLI translates original mask codes).
    """
    if min_object_size<0 or min_hole_size<0 or connectivity not in (4,8):
        raise ValueError("Sizes must be >=0 and connectivity must be 4 or 8.")
    structure=ndi.generate_binary_structure(2,1 if connectivity==4 else 2)
    result=labels.copy()
    class_min_sizes=class_min_sizes or {}
    for c in range(1,num_classes):
        threshold=int(class_min_sizes.get(c,min_object_size))
        if threshold<0:
            raise ValueError("class_min_sizes cannot be negative.")
        if threshold>1:
            cc,n=ndi.label((result==c)&valid,structure)
            sizes=np.bincount(cc.ravel())
            small=sizes<threshold; small[0]=False
            result[small[cc]]=0
    if min_hole_size>1:
        cc,n=ndi.label((result==0)&valid,structure)
        counts=np.bincount(cc.ravel())
        for i,slc in enumerate(ndi.find_objects(cc),start=1):
            if slc is None or counts[i]>=min_hole_size:
                continue
            ys,xs=slc
            if ys.start==0 or xs.start==0 or ys.stop==result.shape[0] or xs.stop==result.shape[1]:
                continue
            patch=(slice(ys.start-1,ys.stop+1),slice(xs.start-1,xs.stop+1))
            hole=cc[patch]==i
            boundary=ndi.binary_dilation(hole,structure=structure)&~hole
            if not valid[patch][boundary].all():
                continue
            neighbors=np.unique(result[patch][boundary])
            if len(neighbors)==1 and 0<neighbors[0]<num_classes:
                region=result[patch]
                region[hole]=neighbors[0]
    result[~valid]=-100
    return result
