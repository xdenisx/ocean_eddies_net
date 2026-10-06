"""Batched tiled inference with disk-backed weighted multiclass accumulation."""
from __future__ import annotations
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import rasterio
import torch
from tqdm.auto import tqdm
from .checkpoint import load_model
from .postprocess import blend_weights, smooth_probabilities_inplace, classify, clean_labels
from .raster import list_rasters, windows, read_image, normalize_image, pad, write_raster
from .train import choose_device


def legacy_normalize(tile,mode):
    """Exact per-band normalization formula used by archived training scripts.

    The package still applies GeoTIFF validity masks at the IO boundary. Conversion
    preserves this formula, not historical bugs in missing-data evaluation.
    """
    tile=tile.astype(np.float32,copy=False)
    valid=np.isfinite(tile)&(tile!=0)
    tile=np.nan_to_num(tile,nan=0,posinf=0,neginf=0)
    if mode=="none":
        return tile
    out=np.zeros_like(tile)
    for c,band in enumerate(tile):
        v=valid[c]
        if not v.any():
            continue
        if mode=="zscore":
            mean=float(np.mean(band[v])); std=float(np.std(band[v]))
            out[c]=(band-mean)/(1. if std<1e-6 else std)
        else:
            lo,hi=np.percentile(band[v],[2,98]) if mode=="percentile" else (band[v].min(),band[v].max())
            if hi>lo:
                out[c]=(band-lo)/(hi-lo)
                if mode=="percentile":
                    out[c]=np.clip(out[c],0,1)
    return out


@torch.inference_mode()
def predict(model,x,amp=False,tta=False):
    flips=[(),(-1,),(-2,),(-2,-1)] if tta else [()]
    total=None
    for dims in flips:
        with torch.autocast(device_type=x.device.type,enabled=amp):
            logits=model(torch.flip(x,dims) if dims else x)
        probabilities=torch.softmax(logits.float(),dim=1)
        if dims:
            probabilities=torch.flip(probabilities,dims)
        total=probabilities if total is None else total+probabilities
    return (total/len(flips)).cpu().numpy()


def infer_image(path,output_path,model,schema,preprocessing,device,tile_size,stride,batch_size=2,
                blend_mode="distance",blend_min_weight=.05,blend_power=1.,smooth_sigma=0.,
                min_object_size=0,min_hole_size=0,class_min_sizes=None,connectivity=8,
                threshold=None,save_prob=False,save_raw_prob=False,tta=False,amp=False,
                work_dir=None,overwrite=False,in_channels=1):
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    prob_path=output_path.with_name(output_path.stem+"_prob.tif")
    raw_path=output_path.with_name(output_path.stem+"_raw_prob.tif")
    outputs=[output_path]+([prob_path] if save_prob else [])+([raw_path] if save_raw_prob else [])
    if not overwrite and any(p.exists() for p in outputs):
        raise FileExistsError("An inference output already exists; use --overwrite or a new output directory.")
    weight=blend_weights(tile_size,blend_mode,blend_min_weight,blend_power)
    with rasterio.open(path) as src:
        h,w=src.height,src.width
        scratch=Path(work_dir or output_path.parent)
        scratch.mkdir(parents=True,exist_ok=True)
        needed=(schema.count+1)*h*w*4
        if shutil.disk_usage(scratch).free < needed+1024**2:
            raise OSError(f"Insufficient scratch space: accumulation requires approximately {needed/1024**3:.2f} GiB before output files.")
        print(f"{path.name}: {w}x{h}, {schema.count} probability bands; scratch {needed/1024**3:.2f} GiB")
        with tempfile.TemporaryDirectory(prefix="eddy_",dir=scratch) as tmp:
            sums=np.memmap(Path(tmp)/"sums.dat",dtype="float32",mode="w+",shape=(schema.count,h,w))
            weights=np.memmap(Path(tmp)/"weights.dat",dtype="float32",mode="w+",shape=(h,w))
            try:
                sums[:]=0; weights[:]=0
                batches=[]; records=[]
                def flush():
                    if not batches:
                        return
                    x=torch.from_numpy(np.stack(batches)).to(device)
                    probs=predict(model,x,amp,tta)
                    if probs.shape[1]!=schema.count or probs.shape[2:]!=(tile_size,tile_size):
                        raise ValueError("Model output does not match the checkpoint class count / tile shape.")
                    if not np.isfinite(probs).all():
                        raise FloatingPointError("Model returned nonfinite probabilities.")
                    for pred,(win,valid) in zip(probs,records):
                        r,c,hh,ww=int(win.row_off),int(win.col_off),int(win.height),int(win.width)
                        local_weight=weight[:hh,:ww]*valid
                        sums[:,r:r+hh,c:c+ww]+=pred[:,:hh,:ww]*local_weight[None]
                        weights[r:r+hh,c:c+ww]+=local_weight
                    batches.clear(); records.clear()
                for win in windows(h,w,tile_size,stride):
                    image,valid=read_image(src,win,in_channels,preprocessing["zero_is_nodata"])
                    if not valid.any():
                        continue
                    if preprocessing.get("legacy_normalization",False):
                        # Legacy training padded before normalization.
                        image=legacy_normalize(pad(image,tile_size),preprocessing["normalize"])
                    else:
                        image=pad(normalize_image(image,valid,preprocessing["normalize"]),tile_size)
                    batches.append(image); records.append((win,valid))
                    if len(batches)>=batch_size:
                        flush()
                flush()
                valid=weights>0
                for row in range(0,h,512):
                    block=sums[:,row:row+512]
                    denominator=weights[row:row+512]
                    np.divide(block,denominator[None],out=block,where=denominator[None]>0)
                    block[:,denominator<=0]=np.nan
                meta={"class_schema":schema.to_dict(),"source_image":Path(path).name,"in_channels":in_channels,
                      "preprocessing":preprocessing,"tile_size":tile_size,"stride":stride,"blend_mode":blend_mode,
                      "smooth_sigma":smooth_sigma,"threshold":threshold,"min_object_size":min_object_size,
                      "min_hole_size":min_hole_size,"class_min_sizes_model_indices":class_min_sizes or {},
                      "connectivity":connectivity,"tta":tta,"valid_pixels":int(valid.sum())}
                descriptions=[f"P({name}); mask_value={value}" for value,name in zip(schema.values,schema.names)]
                if save_raw_prob:
                    write_raster(raw_path,sums,src,valid,np.nan,descriptions,{**meta,"probability_stage":"blended_before_smoothing"})
                smooth_probabilities_inplace(sums,valid,smooth_sigma)
                if save_prob:
                    write_raster(prob_path,sums,src,valid,np.nan,descriptions,{**meta,"probability_stage":"after_smoothing_before_object_cleanup"})
                labels=classify(sums,valid,threshold)
                labels=clean_labels(labels,valid,schema.count,min_object_size,min_hole_size,class_min_sizes,connectivity)
                write_raster(output_path,schema.decode(labels,valid),src,valid,65535,["class_code"],meta)
            finally:
                # Explicit closure is required before TemporaryDirectory cleanup on Windows.
                sums.flush(); weights.flush()
                sums._mmap.close(); weights._mmap.close()
    return output_path


def infer(args):
    device=choose_device(args.device)
    model,schema,ckpt=load_model(args.checkpoint,device)
    preprocessing=ckpt["preprocessing"]
    tile_size=args.tile_size or preprocessing["tile_size"]
    stride=args.stride or max(1,tile_size//2)
    if tile_size%32 or tile_size<32 or not 0<stride<=tile_size:
        raise ValueError("Require tile_size multiple of 32 and 0<stride<=tile_size.")
    if args.batch_size<1:
        raise ValueError("batch_size must be positive.")
    if schema.count!=2 and args.threshold is not None:
        raise ValueError("--threshold is binary-only; multiclass uses argmax.")
    source=Path(args.input).resolve(); output=Path(args.output_dir).resolve()
    if source.is_dir() and (source==output or source in output.parents):
        raise ValueError("output_dir must be outside input folder to prevent recursive reprocessing.")
    code_sizes=json.loads(args.class_min_sizes) if args.class_min_sizes else {}
    if not isinstance(code_sizes,dict):
        raise ValueError("class_min_sizes must be a JSON object keyed by original mask codes.")
    index_sizes={}
    for code,value in code_sizes.items():
        code=int(code)
        if code not in schema.values[1:]:
            raise ValueError(f"class_min_sizes key {code} is not a foreground mask code.")
        if int(value)!=value or value<0:
            raise ValueError("Minimum sizes must be nonnegative integers.")
        index_sizes[schema.values.index(code)]=int(value)
    paths=list_rasters(source,args.recursive)
    if not paths:
        raise ValueError("No input GeoTIFFs found.")
    results=[]
    for path in tqdm(paths,desc="Inference"):
        relative=path.relative_to(source) if source.is_dir() else Path(path.name)
        destination=output/relative.parent/(relative.stem+args.suffix+".tif")
        results.append(infer_image(path,destination,model,schema,preprocessing,device,tile_size,stride,
                     args.batch_size,args.blend_mode,args.blend_min_weight,args.blend_power,args.smooth_sigma,
                     args.min_object_size,args.min_hole_size,index_sizes,args.connectivity,args.threshold,
                     args.save_prob,args.save_raw_prob,args.tta,bool(args.amp and device.type=="cuda"),
                     args.work_dir,args.overwrite,ckpt["model_spec"]["in_channels"]))
    return results
