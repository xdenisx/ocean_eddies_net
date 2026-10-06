"""Explicit checkpoint contract, strict loading and optional legacy conversion."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import torch
from . import __version__
from .models import ModelSpec, build_model, resolve_backend
from .schema import ClassSchema, make_schema

FORMAT_VERSION=1


def environment_versions():
    result={"ocean-eddy-segmentation":__version__}
    for name in ("torch","torchvision","numpy","rasterio","scipy","scikit-image",
                 "opencv-python-headless","segmentation-models-pytorch","timm","transformers"):
        try:
            result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name]=None
    return result


def safe_load(path):
    # Never silently fall back to arbitrary pickle execution.
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(checkpoint,dict):
        raise ValueError("Expected a checkpoint dictionary.")
    return checkpoint


def atomic_save(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".partial")
    try:
        torch.save(data,tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_model(path, device="cpu"):
    ckpt=safe_load(path)
    if ckpt.get("format_version")!=FORMAT_VERSION:
        raise ValueError("This is not a package-format checkpoint. Use eddy-convert-legacy for a trusted legacy checkpoint, or the unchanged legacy inference script.")
    spec=ModelSpec(**ckpt["model_spec"])
    schema=ClassSchema.from_dict(ckpt["class_schema"])
    if schema.count!=spec.num_classes:
        raise ValueError("Class schema and model output count disagree.")
    model=build_model(spec,pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.to(device).eval()
    return model,schema,ckpt


def convert_legacy(input_path, output_path, backend="auto", overwrite=False):
    """Convert binary legacy weights only; verifies exact state-dict compatibility.

    Does not turn a binary-trained classifier into a multiclass-trained classifier.
    Unusual third-party legacy checkpoints may need the original package versions.
    """
    output_path=Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    old=safe_load(input_path)
    if "format_version" in old:
        raise ValueError("Already a package-format checkpoint.")
    args=old.get("args",{})
    arch=args.get("architecture",old.get("architecture"))
    if not arch:
        raise ValueError("Legacy checkpoint has no architecture metadata.")
    classes=int(args.get("num_classes",2))
    if classes!=2:
        raise ValueError("Legacy converter only handles the original two-class contract.")
    state=old.get("model_state_dict")
    if state is None:
        raise ValueError("Missing model_state_dict.")
    state={k.removeprefix("module."):v for k,v in state.items()}
    if arch.startswith("segformer") and backend=="auto":
        if any(k.startswith("model.segformer.") for k in state):
            backend="hf"
        elif any(k.startswith("model.encoder.") for k in state):
            backend="smp"
        else:
            raise ValueError("Cannot identify legacy SegFormer backend; set --backend.")
    spec=ModelSpec(arch,int(args.get("in_channels",1)),2,resolve_backend(arch,backend))
    if spec.backend=="hf":
        # Old files did not preserve full HF configs. Fetch configuration once;
        # the converted file embeds it, so subsequent inference can be offline.
        from transformers import SegformerConfig,UperNetConfig
        if arch.startswith("segformer"):
            v="b2" if arch=="segformer" else arch.split("_")[-1]
            cfg=SegformerConfig.from_pretrained(f"nvidia/segformer-{v}-finetuned-ade-{'640-640' if v=='b5' else '512-512'}")
            cfg.num_channels=spec.in_channels
        else:
            source="openmmlab/upernet-swin-tiny" if arch.endswith("_t") else "openmmlab/upernet-swin-small"
            cfg=UperNetConfig.from_pretrained(source)
        cfg.num_labels=2
        spec.hf_config=cfg.to_dict()
    model=build_model(spec,pretrained=False)
    model.load_state_dict(state,strict=True)
    mask_value=int(args.get("mask_value",255))
    schema=make_schema((0,mask_value),("background","eddy"),unknown_policy="background")
    result={"format_version":FORMAT_VERSION,"package_version":__version__,"model_spec":spec.to_dict(),
            "class_schema":schema.to_dict(),"model_state_dict":model.state_dict(),
            "preprocessing":{"normalize":args.get("normalize","percentile"),"zero_is_nodata":True,
                             "tile_size":int(args.get("tile_size",512)),"legacy_normalization":True},
            "legacy_args":args,"environment":environment_versions(),"converted_from":Path(input_path).name}
    atomic_save(output_path,result)
    print(f"Saved converted binary checkpoint: {output_path}")
    return result
