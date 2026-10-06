"""Binary/multiclass softmax losses with explicit ignored-pixel handling."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .schema import IGNORE_INDEX

LOSSES = ("ce","dice","dice_ce","focal","dice_focal","tversky","focal_tversky")


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes: int, kind="dice_focal", class_weights=None,
                 include_background=False, focal_alpha=.75, focal_gamma=2.,
                 tversky_alpha=.3, tversky_beta=.7, focal_tversky_gamma=1.33):
        super().__init__()
        if kind not in LOSSES:
            raise ValueError(f"Unknown loss {kind}")
        self.num_classes=num_classes
        self.kind=kind
        self.include_background=include_background
        weights=torch.ones(num_classes) if class_weights is None else torch.tensor(class_weights,dtype=torch.float32)
        if weights.numel()!=num_classes or not torch.isfinite(weights).all() or (weights<=0).any():
            raise ValueError("Need one finite positive weight per class.")
        self.register_buffer("weights",weights)
        alpha=torch.ones(num_classes)
        if num_classes==2:
            alpha=torch.tensor([1-focal_alpha,focal_alpha])
        self.register_buffer("focal_weights",weights*alpha)
        self.gamma=focal_gamma
        self.tv_alpha=tversky_alpha
        self.tv_beta=tversky_beta
        self.ft_gamma=focal_tversky_gamma

    def forward(self, logits, target):
        if logits.shape[1]!=self.num_classes or logits.shape[0]!=target.shape[0] or logits.shape[2:]!=target.shape[1:]:
            raise ValueError("Logits must be N,C,H,W and targets N,H,W.")
        valid=target!=IGNORE_INDEX
        if not valid.any():
            return logits.sum()*0.
        if ((target[valid]<0)|(target[valid]>=self.num_classes)).any():
            raise ValueError("Target contains an invalid model class index.")
        safe=target.masked_fill(~valid,0)
        log_probs=F.log_softmax(logits.float(),dim=1)
        log_pt=log_probs.gather(1,safe.unsqueeze(1)).squeeze(1)
        weights=self.weights[safe][valid]
        ce=(-log_pt[valid]*weights).sum()/weights.sum()
        focal=(-log_pt[valid]*(1-log_pt[valid].exp()).pow(self.gamma)*self.focal_weights[safe][valid]).mean()
        probs=log_probs.exp()*valid.unsqueeze(1)
        truth=F.one_hot(safe,self.num_classes).permute(0,3,1,2).float()*valid.unsqueeze(1)
        dims=(0,2,3)
        tp=(probs*truth).sum(dims)
        fp=(probs*(1-truth)).sum(dims)
        fn=((1-probs)*truth).sum(dims)
        dice=1-(2*tp+1)/(2*tp+fp+fn+1)
        tv=1-(tp+1)/(tp+self.tv_alpha*fp+self.tv_beta*fn+1)
        start=0 if self.include_background else 1
        aggregate=lambda x: (x[start:]*self.weights[start:]).sum()/self.weights[start:].sum()
        values={"ce":ce, "focal":focal, "dice":aggregate(dice), "tversky":aggregate(tv),
                "focal_tversky":aggregate(tv.pow(self.ft_gamma))}
        values["dice_ce"]=(values["dice"]+ce)/2
        values["dice_focal"]=(values["dice"]+focal)/2
        return values[self.kind]
