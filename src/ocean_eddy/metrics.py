"""Confusion-matrix aggregation: count pixels first, then calculate ratios."""
from __future__ import annotations
import numpy as np
from .schema import IGNORE_INDEX


def confusion_matrix(target, prediction, num_classes):
    target=np.asarray(target).ravel()
    prediction=np.asarray(prediction).ravel()
    good=target!=IGNORE_INDEX
    t,p=target[good].astype(np.int64),prediction[good].astype(np.int64)
    if np.any((t<0)|(t>=num_classes)|(p<0)|(p>=num_classes)):
        raise ValueError("Invalid class index in confusion-matrix input.")
    return np.bincount(t*num_classes+p,minlength=num_classes**2).reshape(num_classes,num_classes)


def ratio(a,b):
    return float(a/b) if b>0 else None


def mean_defined(values):
    defined=[v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def summarize(cm, schema):
    cm=np.asarray(cm,dtype=np.int64)
    if cm.shape!=(schema.count,schema.count):
        raise ValueError("Confusion matrix does not match schema.")
    total=int(cm.sum())
    rows=[]
    for c,(value,name) in enumerate(zip(schema.values,schema.names)):
        tp=int(cm[c,c]); fn=int(cm[c].sum()-tp); fp=int(cm[:,c].sum()-tp); tn=total-tp-fp-fn
        rows.append(dict(class_index=c,mask_value=value,class_name=name,tp=tp,fp=fp,fn=fn,tn=tn,
                         support=tp+fn,predicted_pixels=tp+fp,iou=ratio(tp,tp+fp+fn),
                         dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
                         specificity=ratio(tn,tn+fp),accuracy_ovr=ratio(tp+tn,total)))
    result={"valid_pixels":total,"accuracy":ratio(np.trace(cm),total),"per_class":rows,
            "confusion_matrix":cm.tolist(),"confusion_axes":"rows=reference, columns=prediction"}
    for name in ("iou","dice","precision","recall","specificity"):
        result["macro_"+name]=mean_defined([r[name] for r in rows])
        result["foreground_"+name]=mean_defined([r[name] for r in rows[1:]])
    result["frequency_weighted_iou"]=(sum(r["support"]*(r["iou"] or 0) for r in rows)/total if total else None)
    # Compact names used by early stopping. Binary -> foreground eddy class;
    # multiclass -> unweighted mean of defined non-background class metrics.
    result["iou"]=result["foreground_iou"]
    result["dice"]=result["foreground_dice"]
    return result
