"""Check dependencies in the interpreter actually running the package."""
from __future__ import annotations
import importlib
import json
import platform
import sys
from .checkpoint import environment_versions


def doctor(args):
    report={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "versions":environment_versions(),"imports":{}}
    for name in ("torch","rasterio","cv2","scipy","skimage","segmentation_models_pytorch","timm","transformers"):
        try:
            importlib.import_module(name)
            report["imports"][name]="OK"
        except Exception as exc:
            report["imports"][name]=f"{type(exc).__name__}: {exc}"
    import torch
    report["cuda_available"]=torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"]=torch.cuda.get_device_name(0)
    if args.architecture:
        from .models import ModelSpec,build_model,freeze_batchnorm
        try:
            spec=ModelSpec(args.architecture,args.in_channels,args.num_classes,args.backend)
            model=build_model(spec,pretrained=False)
            model.eval()
            if args.forward:
                with torch.inference_mode():
                    out=model(torch.zeros(1,args.in_channels,args.tile_size,args.tile_size))
                report["output_shape"]=list(out.shape)
            report["model_spec"]=spec.to_dict()
            report["parameters"]=sum(p.numel() for p in model.parameters())
        except Exception as exc:
            report["model_error"]=f"{type(exc).__name__}: {exc}"
    print(json.dumps(report,indent=2))
    if "model_error" in report:
        raise SystemExit(1)
    return report
