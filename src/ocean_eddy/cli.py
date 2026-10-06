"""Unified CLI; supports familiar underscore flags and hyphenated aliases."""
from __future__ import annotations
import argparse
from dataclasses import fields
import sys
from . import __version__
from .config import TrainConfig,load_yaml
from .models import ARCHITECTURES
from .losses import LOSSES


def add(parser,name,**kwargs):
    options=["--"+name]
    if "_" in name:
        options.append("--"+name.replace("_","-"))
    return parser.add_argument(*options,**kwargs)


def boolean(parser,name,default=None,help=None):
    add(parser,name,action=argparse.BooleanOptionalAction,default=default,help=help)


def make_parser():
    parser=argparse.ArgumentParser(prog="python -m ocean_eddy",description="GeoTIFF binary / multiclass semantic segmentation")
    parser.add_argument("--version",action="version",version=__version__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("train",help="Train from matched images and masks")
    add(p,"config",help="YAML config. Paths resolve relative to the current working directory.")
    for key in ("images_dir","masks_dir","output_dir","split_file","resume"):
        add(p,key,default=None)
    add(p,"architecture",choices=ARCHITECTURES,default=None)
    add(p,"backend",choices=["auto","native","smp","hf","timm"],default=None)
    add(p,"loss",choices=LOSSES,default=None)
    add(p,"normalize",choices=["percentile","minmax","zscore","none"],default=None)
    add(p,"augment_level",choices=["light","medium","heavy"],default=None)
    add(p,"split_mode",choices=["scene","tile"],default=None)
    add(p,"unknown_mask_policy",choices=["error","ignore","background"],default=None)
    add(p,"early_stop_monitor",choices=["val_iou","val_dice","val_loss"],default=None)
    add(p,"device",default=None)
    for key in ("in_channels","tile_size","stride","epochs","batch_size","num_workers","seed","max_tiles","accumulation_steps","patience"):
        add(p,key,type=int,default=None)
    for key in ("min_valid_fraction","min_positive_fraction","lr","weight_decay","val_fraction","focal_alpha",
                "focal_gamma","tversky_alpha","tversky_beta","focal_tversky_gamma","min_delta","grad_clip"):
        add(p,key,type=float,default=None)
    for key,kind in (("class_values",int),("class_names",str),("ignore_values",int),("class_weights",float)):
        add(p,key,nargs="+",type=kind,default=None)
    for key in ("pretrained","zero_is_nodata","freeze_batchnorm","drop_last","amp","include_background_loss","allow_flips"):
        boolean(p,key,None)
    add(p,"no_augment",action="store_true",default=None)
    p.add_argument("--no_pretrained",dest="pretrained",action="store_false",default=None)
    p.add_argument("--no_drop_last",dest="drop_last",action="store_false",default=None)
    add(p,"no_flips",dest="allow_flips",action="store_false",default=None)
    add(p,"num_classes",type=int,default=None,help="Optional consistency check against len(class_values).")
    add(p,"mask_value",type=int,default=None,help="Legacy binary shorthand for --class_values 0 VALUE.")

    p=sub.add_parser("infer",help="Infer from a package checkpoint")
    for name in ("input","checkpoint","output_dir"):
        add(p,name,required=True)
    for name in ("tile_size","stride"):
        add(p,name,type=int,default=None)
    add(p,"batch_size",type=int,default=2)
    add(p,"threshold",type=float,default=None,help="Binary-only foreground threshold; default 0.5. Multiclass uses argmax.")
    add(p,"blend_mode",choices=["uniform","distance","hann","gaussian"],default="distance")
    add(p,"blend_min_weight",type=float,default=.05)
    add(p,"blend_power",type=float,default=1.)
    add(p,"smooth_sigma",type=float,default=0.)
    add(p,"min_object_size",type=int,default=0)
    add(p,"min_hole_size",type=int,default=0)
    add(p,"class_min_sizes",default=None,help='JSON keyed by original foreground mask values, e.g. {"100":200,"255":100}. Sizes are pixels.')
    add(p,"connectivity",type=int,choices=[4,8],default=8)
    add(p,"suffix",default="_pred")
    add(p,"work_dir",default=None)
    add(p,"device",default="auto")
    boolean(p,"recursive",True)
    for name in ("tta","amp","save_prob","save_raw_prob","overwrite"):
        add(p,name,action="store_true")

    p=sub.add_parser("validate",help="Evaluate class-code prediction GeoTIFFs")
    for name in ("pred_dir","mask_dir","output_dir"):
        add(p,name,required=True)
    add(p,"pred_suffix",default="_pred")
    add(p,"checkpoint",default=None)
    add(p,"images_dir",default=None,help="Recommended: original input images define valid evaluation support.")
    add(p,"in_channels",type=int,default=None)
    boolean(p,"zero_is_nodata",None)
    for key,kind in (("class_values",int),("class_names",str),("ignore_values",int)):
        add(p,key,nargs="+",type=kind,default=None)
    add(p,"unknown_mask_policy",choices=["error","ignore","background"],default="error")
    add(p,"missing_prediction_policy",choices=["error","ignore"],default="error")
    add(p,"allow_missing",action="store_true",help="Allow unevaluated reference scenes, listed in the report.")
    add(p,"overwrite",action="store_true")

    p=sub.add_parser("preprocess",help="GeoTIFF minmax + HistEq + skimage CLAHE + uint8")
    add(p,"input_dir",required=True); add(p,"output_dir",required=True)
    add(p,"clip_limit",type=float,default=.03)
    add(p,"kernel_size",type=int,default=0,help="Pixels; 0 lets skimage choose its default.")
    add(p,"skip_equalize_hist",action="store_true")
    add(p,"suffix",default="",help="Default preserves stems for image/mask matching.")
    boolean(p,"recursive",True); boolean(p,"zero_is_nodata",True)
    add(p,"overwrite",action="store_true")

    p=sub.add_parser("doctor",help="Report interpreter, package imports and optional model construction")
    add(p,"architecture",choices=ARCHITECTURES,default=None)
    add(p,"backend",choices=["auto","native","smp","hf","timm"],default="auto")
    add(p,"in_channels",type=int,default=1)
    add(p,"num_classes",type=int,default=3)
    add(p,"tile_size",type=int,default=64)
    add(p,"forward",action="store_true",help="Run one CPU forward pass, no pretrained download.")

    p=sub.add_parser("convert-legacy",help="Strictly verify and repackage a legacy binary checkpoint")
    add(p,"input",required=True); add(p,"output",required=True)
    add(p,"backend",choices=["auto","native","smp","hf","timm"],default="auto")
    add(p,"overwrite",action="store_true")
    return parser


def training_config(args):
    base=load_yaml(args.config) if args.config else {}
    allowed={f.name for f in fields(TrainConfig)}
    for key,value in vars(args).items():
        if key in allowed and value is not None:
            base[key]=value
    if args.mask_value is not None:
        if "class_values" in base:
            raise ValueError("Do not combine mask_value and class_values.")
        base["class_values"]=[0,args.mask_value]
    cfg=TrainConfig.from_dict(base)
    if args.num_classes is not None and args.num_classes!=cfg.schema().count:
        raise ValueError("num_classes must match len(class_values), including background. E.g. --class_values 0 100 255 --num_classes 3.")
    return cfg


def main(argv=None):
    args=make_parser().parse_args(argv)
    if args.command=="train":
        from .train import train
        train(training_config(args))
    elif args.command=="infer":
        from .infer import infer
        infer(args)
    elif args.command=="validate":
        from .evaluate import evaluate
        evaluate(args)
    elif args.command=="preprocess":
        from .preprocess import preprocess
        preprocess(args)
    elif args.command=="doctor":
        from .doctor import doctor
        doctor(args)
    else:
        from .checkpoint import convert_legacy
        convert_legacy(args.input,args.output,args.backend,args.overwrite)


def train_main():
    main(["train",*sys.argv[1:]])


def infer_main():
    main(["infer",*sys.argv[1:]])


def validate_main():
    main(["validate",*sys.argv[1:]])


def preprocess_main():
    main(["preprocess",*sys.argv[1:]])


def doctor_main():
    main(["doctor",*sys.argv[1:]])


def convert_main():
    main(["convert-legacy",*sys.argv[1:]])
