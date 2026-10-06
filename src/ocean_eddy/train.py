"""Training orchestration. All model/data metadata is stored in each checkpoint."""
from __future__ import annotations
import csv
import json
import math
import random
import warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from . import __version__
from .checkpoint import FORMAT_VERSION, atomic_save, environment_versions, safe_load
from .config import TrainConfig
from .data import EddyDataset, index_tiles, split_scenes
from .losses import SegmentationLoss
from .metrics import summarize
from .models import ModelSpec, build_model, freeze_batchnorm
from .raster import pair_rasters
from .schema import IGNORE_INDEX


def choose_device(value="auto"):
    if value=="auto":
        value="cuda" if torch.cuda.is_available() else "cpu"
    device=torch.device(value)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment.")
    return device


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed); np.random.seed(seed)


def run_epoch(model,loader,criterion,device,num_classes,schema,amp=False,optimizer=None,
              scaler=None,grad_clip=1.,accumulation_steps=1,freeze_bn=False):
    training=optimizer is not None
    model.train(training)
    if training and freeze_bn:
        freeze_batchnorm(model)
    cm=torch.zeros(num_classes,num_classes,dtype=torch.int64,device=device)
    loss_sum=0.; n_valid=0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step,(images,labels) in enumerate(tqdm(loader,desc="Train" if training else "Val",leave=False)):
        images=images.to(device,non_blocking=True)
        labels=labels.to(device,non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(images)
                loss=criterion(logits,labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; inspect normalization, labels and learning rate.")
            if training:
                group_start=(step//accumulation_steps)*accumulation_steps
                actual_steps=min(accumulation_steps,len(loader)-group_start)
                scaler.scale(loss/actual_steps).backward()
                if (step+1)%accumulation_steps==0 or step+1==len(loader):
                    if grad_clip>0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        valid=labels!=IGNORE_INDEX
        nv=int(valid.sum().item())
        loss_sum+=float(loss.detach())*nv; n_valid+=nv
        if nv:
            pred=logits.detach().argmax(1)
            cm+=torch.bincount(labels[valid]*num_classes+pred[valid],minlength=num_classes**2).reshape(num_classes,num_classes)
    if n_valid==0:
        raise ValueError("Epoch contains no supervised pixels; reduce geometric augmentation or inspect masks.")
    result=summarize(cm.cpu().numpy(),schema)
    result["loss"]=loss_sum/n_valid
    return result


def _write_history(output,history):
    with open(output/"history.json","w",encoding="utf-8") as f:
        json.dump(history,f,indent=2,allow_nan=False)
    with open(output/"training_log.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)


def train(cfg: TrainConfig):
    cfg.validate()
    schema=cfg.schema()
    set_seed(cfg.seed)
    device=choose_device(cfg.device)
    amp=bool(cfg.amp and device.type=="cuda")
    output=Path(cfg.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"last_checkpoint.pt").exists() and not cfg.resume:
        raise FileExistsError("output_dir already contains a run. Use a new directory or --resume explicitly.")
    pairs=pair_rasters(cfg.images_dir,cfg.masks_dir)
    if cfg.split_mode=="scene":
        tr_pairs,va_pairs=split_scenes(pairs,cfg.val_fraction,cfg.seed,cfg.split_file)
        tr=index_tiles(tr_pairs,cfg,True); va=index_tiles(va_pairs,cfg,False)
    else:
        warnings.warn("TILE SPLIT: neighboring/overlapping tiles can leak scene information. Do not report this as independent generalization.")
        if cfg.split_file:
            raise ValueError("split_file is only supported with split_mode=scene.")
        all_tiles=index_tiles(pairs,cfg,False)
        if len(all_tiles)<2:
            raise ValueError("Need at least two usable tiles for tile splitting.")
        order=np.random.default_rng(cfg.seed).permutation(len(all_tiles))
        nv=min(len(order)-1,max(1,int(np.ceil(len(order)*cfg.val_fraction))))
        va=[all_tiles[i] for i in order[:nv]]
        tr=[all_tiles[i] for i in order[nv:] if all_tiles[i].foreground_fraction>=cfg.min_positive_fraction]
        if cfg.max_tiles:
            tr=tr[:cfg.max_tiles]
        if not tr:
            raise ValueError("No training tiles after foreground filtering.")
    split={"split_mode":cfg.split_mode,"train":sorted({t.scene_id for t in tr}),"val":sorted({t.scene_id for t in va}),
           "train_tiles":len(tr),"val_tiles":len(va),"validation_positive_filter":False}
    tr_ds=EddyDataset(tr,cfg,True); va_ds=EddyDataset(va,cfg,False)
    if cfg.drop_last and len(tr)<cfg.batch_size:
        raise ValueError("Fewer train tiles than batch_size with drop_last=True; lower batch_size or use --no_drop_last.")
    common=dict(batch_size=cfg.batch_size,num_workers=cfg.num_workers,pin_memory=device.type=="cuda",worker_init_fn=seed_worker)
    generator=torch.Generator().manual_seed(cfg.seed)
    tr_loader=DataLoader(tr_ds,shuffle=True,drop_last=cfg.drop_last,generator=generator,**common)
    va_loader=DataLoader(va_ds,shuffle=False,drop_last=False,**common)
    old=safe_load(cfg.resume) if cfg.resume else None
    if old:
        if old.get("format_version")!=FORMAT_VERSION or "optimizer_state_dict" not in old:
            raise ValueError("Resume needs a package training checkpoint, not legacy-converted inference weights.")
        if old["class_schema"]!=schema.to_dict():
            raise ValueError("Resume class schema does not match.")
        previous=old["config"]
        # Epoch count affects the cosine schedule; exact resume uses the same planned run.
        allowed_changes={"resume","device","num_workers"}
        changed=[k for k,v in cfg.to_dict().items() if k not in allowed_changes and previous.get(k)!=v]
        if changed:
            raise ValueError(f"Resume configuration changed: {changed}. Resume the original run configuration.")
        spec=ModelSpec(**old["model_spec"])
    else:
        spec=ModelSpec(cfg.architecture,cfg.in_channels,schema.count,cfg.backend)
    model=build_model(spec,pretrained=bool(cfg.pretrained and old is None)).to(device)
    if old:
        model.load_state_dict(old["model_state_dict"],strict=True)
    has_bn=any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if has_bn and not cfg.freeze_batchnorm and (cfg.batch_size==1 or (not cfg.drop_last and len(tr)%cfg.batch_size==1)):
        raise ValueError("A batch of size 1 can fail in pooled BatchNorm. Use --freeze_batchnorm, or batch_size>=2 with drop_last=True. Gradient accumulation alone does not solve this.")
    if cfg.freeze_batchnorm:
        print(f"Freeze running statistics in {freeze_batchnorm(model)} BatchNorm layers (affine parameters remain trainable).")
    criterion=SegmentationLoss(schema.count,cfg.loss,cfg.class_weights,cfg.include_background_loss,
                               cfg.focal_alpha,cfg.focal_gamma,cfg.tversky_alpha,cfg.tversky_beta,cfg.focal_tversky_gamma).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs)
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    maximize=cfg.early_stop_monitor!="val_loss"
    best=-math.inf if maximize else math.inf
    anchor=best; bad=0; start=1; history=[]
    if old:
        optimizer.load_state_dict(old["optimizer_state_dict"])
        scheduler.load_state_dict(old["scheduler_state_dict"])
        scaler.load_state_dict(old["scaler_state_dict"])
        start=old["epoch"]+1; best=old["best_metric"]; anchor=old["early_stop_anchor"]; bad=old["bad_epochs"]
        history=old["history"]
        generator.set_state(old["loader_generator_state"])
        torch.set_rng_state(old["torch_rng_state"])
        if device.type=="cuda" and old.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(old["cuda_rng_states"])
    for name,obj in (("config.json",cfg.to_dict()),("class_schema.json",schema.to_dict()),("split.json",split),("environment.json",environment_versions())):
        with open(output/name,"w",encoding="utf-8") as f:
            json.dump(obj,f,indent=2,allow_nan=False)
    print(f"{__version__} | {spec.architecture} [{spec.backend}] | {schema.count} classes | {len(tr)} train / {len(va)} val tiles | {device}")
    for epoch in range(start,cfg.epochs+1):
        tr_ds.epoch=epoch
        print(f"Epoch {epoch}/{cfg.epochs}")
        lr=optimizer.param_groups[0]["lr"]
        ts=run_epoch(model,tr_loader,criterion,device,schema.count,schema,amp,optimizer,scaler,
                     cfg.grad_clip,cfg.accumulation_steps,cfg.freeze_batchnorm)
        vs=run_epoch(model,va_loader,criterion,device,schema.count,schema,amp)
        scheduler.step()
        row={"epoch":epoch,"lr":lr}
        for prefix,stats in (("train",ts),("val",vs)):
            for key in ("loss","iou","dice","macro_iou","macro_dice","accuracy","foreground_precision","foreground_recall"):
                row[f"{prefix}_{key}"]=stats[key]
        history.append(row)
        value=row[cfg.early_stop_monitor]
        if value is None or not math.isfinite(value):
            raise ValueError(f"{cfg.early_stop_monitor} is undefined; no evaluable foreground. Check validation split or monitor val_loss.")
        improved=value>best if maximize else value<best
        if improved:
            best=value
        meaningful=value>anchor+cfg.min_delta if maximize else value<anchor-cfg.min_delta
        if meaningful:
            anchor=value; bad=0
        else:
            bad+=1
        _write_history(output,history)
        ckpt={"format_version":FORMAT_VERSION,"package_version":__version__,"model_spec":spec.to_dict(),
              "class_schema":schema.to_dict(),"model_state_dict":model.state_dict(),"config":cfg.to_dict(),
              "preprocessing":{"normalize":cfg.normalize,"zero_is_nodata":cfg.zero_is_nodata,"tile_size":cfg.tile_size,"legacy_normalization":False},
              "optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),
              "scaler_state_dict":scaler.state_dict(),"epoch":epoch,"best_metric":best,"monitor":cfg.early_stop_monitor,
              "early_stop_anchor":anchor,"bad_epochs":bad,"history":history,"train_stats":ts,"val_stats":vs,
              "environment":environment_versions(),"loader_generator_state":generator.get_state(),
              "torch_rng_state":torch.get_rng_state(),"cuda_rng_states":torch.cuda.get_rng_state_all() if device.type=="cuda" else []}
        # Best checkpoint tracks every strict improvement, independently of min_delta.
        if improved:
            atomic_save(output/"best_checkpoint.pt",ckpt)
        atomic_save(output/"last_checkpoint.pt",ckpt)
        print(f"train_loss={ts['loss']:.5f} val_loss={vs['loss']:.5f} {cfg.early_stop_monitor}={value:.5f} best={best:.5f} patience={bad}/{cfg.patience}")
        if cfg.patience>0 and bad>=cfg.patience:
            print("Early stopping.")
            break
    return output
