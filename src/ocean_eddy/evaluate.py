"""Streaming per-scene and pooled multiclass validation of georeferenced masks."""
from __future__ import annotations
import csv
import json
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import rasterio
from tqdm.auto import tqdm
from .checkpoint import safe_load
from .metrics import confusion_matrix,summarize
from .raster import raster_map,check_alignment,windows,read_image,read_labels
from .schema import ClassSchema,IGNORE_INDEX,make_schema


def _csv(path,rows):
    if not rows:
        return
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate(args):
    predictions=raster_map(args.pred_dir,args.pred_suffix)
    masks=raster_map(args.mask_dir)
    if not predictions:
        raise ValueError("No prediction files matching pred_suffix.")
    missing=sorted(set(masks)-set(predictions))
    extra=sorted(set(predictions)-set(masks))
    if extra:
        raise ValueError(f"Predictions without reference masks: {extra[:12]}")
    if missing and not args.allow_missing:
        raise ValueError(f"Missing predictions for {missing[:12]}; use a held-out mask directory or explicitly --allow_missing.")
    first=next(iter(predictions.values()))
    checkpoint=safe_load(args.checkpoint) if args.checkpoint else None
    with rasterio.open(first) as src:
        first_tags=src.tags()
    inherited_preprocessing=(checkpoint["preprocessing"] if checkpoint else
                             json.loads(first_tags.get("preprocessing","{}")))
    in_channels=(args.in_channels if args.in_channels is not None else
                 int(checkpoint["model_spec"]["in_channels"] if checkpoint else first_tags.get("in_channels",1)))
    zero_is_nodata=(args.zero_is_nodata if args.zero_is_nodata is not None else
                    inherited_preprocessing.get("zero_is_nodata",True))
    if args.class_values is None and (args.class_names is not None or args.ignore_values is not None):
        raise ValueError("To override class_names or ignore_values, also provide class_values explicitly.")
    if args.class_values is not None:
        schema=make_schema(args.class_values,args.class_names,args.ignore_values or (),args.unknown_mask_policy)
    elif args.checkpoint:
        schema=ClassSchema.from_dict(checkpoint["class_schema"])
    else:
        with rasterio.open(first) as src:
            raw=src.tags().get("class_schema")
        if raw is None:
            raise ValueError("No class schema metadata; pass --checkpoint or --class_values / --class_names.")
        schema=ClassSchema.from_dict(json.loads(raw))
    # External predictions must use the explicit class codes, not unknown->background.
    pred_schema=ClassSchema(schema.values,schema.names,(),"error")
    images=raster_map(args.images_dir) if args.images_dir else None
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    if (output/"metrics_summary.json").exists() and not args.overwrite:
        raise FileExistsError("Validation output exists; use a new output_dir or --overwrite.")
    pooled=np.zeros((schema.count,schema.count),np.int64)
    scene_rows=[]; class_rows=[]; report_rows=[]
    excluded=0; missing_total=0
    for key in tqdm(sorted(predictions),desc="Validation"):
        cm=np.zeros_like(pooled); skipped=0; missing_count=0; reference_count=0
        with ExitStack() as stack:
            ps=stack.enter_context(rasterio.open(predictions[key]))
            ms=stack.enter_context(rasterio.open(masks[key]))
            check_alignment(ps,ms)
            if ps.count!=1:
                raise ValueError("Validate discrete class masks, not multiband probabilities.")
            tag=ps.tags().get("class_schema")
            if tag:
                saved=ClassSchema.from_dict(json.loads(tag))
                if saved.values!=schema.values or saved.names!=schema.names:
                    raise ValueError(f"Prediction schema differs for {key}.")
            im=None
            if images is not None:
                if key not in images:
                    raise ValueError(f"No validity image for {key}")
                im=stack.enter_context(rasterio.open(images[key]))
                check_alignment(im,ms)
            for win in windows(ms.height,ms.width,1024,1024):
                support=np.ones((int(win.height),int(win.width)),bool)
                if im:
                    _,support=read_image(im,win,in_channels,zero_is_nodata)
                truth=read_labels(ms,win,schema,support)
                supervised=truth!=IGNORE_INDEX
                reference_count+=int(supervised.sum())
                raw=ps.read(1,window=win)
                good_prediction=(ps.read_masks(1,window=win)>0)&np.isfinite(raw)
                # Declared output NoData is never a class in package predictions.
                missing_here=supervised&~good_prediction
                count=int(missing_here.sum()); missing_count+=count
                if count and args.missing_prediction_policy=="error":
                    raise ValueError(f"{key}: {count} labelled pixels lack predictions in one block. Pass --images_dir to define the original image support, or explicitly --missing_prediction_policy ignore (coverage will be reported).")
                good=supervised&good_prediction
                pred=pred_schema.encode(raw,good)
                truth[~good]=IGNORE_INDEX
                skipped+=int((~good).sum())
                cm+=confusion_matrix(truth,pred,schema.count)
        summary=summarize(cm,schema)
        pooled+=cm; excluded+=skipped; missing_total+=missing_count
        flat={k:v for k,v in summary.items() if k not in {"per_class","confusion_matrix","confusion_axes"}}
        scene_rows.append({"scene_id":key,**flat,"reference_pixels":reference_count,
                           "missing_prediction_pixels":missing_count,"excluded_pixels":skipped})
        for row in summary["per_class"]:
            class_rows.append({"scene_id":key,**row})
        report_rows.append({"scene_id":key,**summary,"reference_pixels":reference_count,
                            "missing_prediction_pixels":missing_count,"excluded_pixels":skipped})
    report=summarize(pooled,schema)
    if not report["valid_pixels"]:
        raise ValueError("No valid reference/prediction pixels to evaluate.")
    report.update(class_schema=schema.to_dict(),num_images=len(scene_rows),unmatched_reference_scenes=missing,
                  excluded_pixels=excluded,missing_prediction_pixels=missing_total,
                  missing_prediction_policy=args.missing_prediction_policy,
                  image_validity={"in_channels":in_channels,"zero_is_nodata":zero_is_nodata},
                  scope="Pixel metrics on evaluated support; not instance/eddy-object metrics.")
    _csv(output/"metrics_per_image.csv",scene_rows)
    _csv(output/"metrics_per_image_class.csv",class_rows)
    _csv(output/"metrics_per_class.csv",report["per_class"])
    with open(output/"confusion_matrix.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["reference\\prediction",*schema.names])
        for name,row in zip(schema.names,pooled):
            writer.writerow([name,*row.tolist()])
    for filename,content in (("metrics_summary.json",report),("metrics_per_image.json",report_rows)):
        with open(output/filename,"w",encoding="utf-8") as f:
            json.dump(content,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ("valid_pixels","macro_iou","foreground_iou","macro_dice","accuracy")},indent=2))
    return report
