"""Shared model registry with explicit backends and checkpoint reconstruction.

Native blocks preserve parameter names from the stable script for conversion.
The `transunet` and `sam_vit_unet` names are legacy custom implementations,
not claims of official TransUNet/SAM reproduction.
"""
from __future__ import annotations
import warnings
from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleTransUNet(nn.Module):
    """Lightweight TransUNet-like fallback with one extra decoder layer."""
    def __init__(self, in_channels: int, num_classes: int, base: int = 48, heads: int = 4, layers: int = 2):
        super().__init__()
        self.enc1 = ConvBNReLU(in_channels, base)
        self.enc2 = ConvBNReLU(base, base * 2)
        self.enc3 = ConvBNReLU(base * 2, base * 4)
        self.enc4 = ConvBNReLU(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        enc_layer = nn.TransformerEncoderLayer(d_model=base * 8, nhead=heads, dim_feedforward=base * 16, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBNReLU(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBNReLU(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBNReLU(base * 2, base)
        # Extra refinement decoder layer requested in original TransUNet version.
        self.refine = ConvBNReLU(base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b, c, h, w = e4.shape
        tokens = e4.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        e4 = tokens.transpose(1, 2).reshape(b, c, h, w)
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.refine(d1)
        return self.out(d1)


class SMPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class TimmHRNetSeg(nn.Module):
    def __init__(self, backbone_name: str, in_channels: int, num_classes: int, pretrained: bool = False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        chs = self.backbone.feature_info.channels()
        self.proj = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in chs])
        self.head = nn.Sequential(
            ConvBNReLU(128 * len(chs), 256),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        size = x.shape[-2:]
        ups = []
        for f, p in zip(feats, self.proj):
            y = p(f)
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            ups.append(y)
        return self.head(torch.cat(ups, dim=1))


class SAMLikeUNet(nn.Module):
    """Legacy alias: larger SimpleTransUNet; NOT Meta SAM or MedSAM."""
    def __init__(self, in_channels: int, num_classes: int, base: int = 64):
        super().__init__()
        self.net = SimpleTransUNet(in_channels, num_classes, base=base, heads=8, layers=4)

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """Small CPU smoke-test model, not an ocean-eddy benchmark recommendation."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(in_channels,8,3,padding=1),nn.GroupNorm(2,8),nn.GELU())
        self.mid=nn.Sequential(nn.Conv2d(8,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU())
        self.head=nn.Conv2d(24,num_classes,1)

    def forward(self,x):
        e=self.enc(x)
        y=self.mid(F.avg_pool2d(e,2))
        return self.head(torch.cat([e,F.interpolate(y,size=x.shape[-2:],mode="bilinear",align_corners=False)],1))


ARCHITECTURES=("transunet","segformer","segformer_b0","segformer_b2","segformer_b4","segformer_b5",
               "unetpp_effb4","unetpp_effb5","deeplabv3plus_resnet50","deeplabv3plus_resnet101",
               "upernet_swin_t","upernet_swin_s","hrnet_w18","hrnet_w32","sam_vit_unet","tiny_unet")


@dataclass
class ModelSpec:
    architecture: str
    in_channels: int
    num_classes: int
    backend: str = "auto"
    hf_config: dict | None = None

    def to_dict(self):
        return asdict(self)


def resolve_backend(architecture, backend="auto"):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture}; choices: {ARCHITECTURES}")
    default=("native" if architecture in {"transunet","sam_vit_unet","tiny_unet"} else
             "timm" if architecture.startswith("hrnet_") else
             "hf" if architecture.startswith("upernet_") else "smp")
    chosen=default if backend=="auto" else backend
    allowed={default,"hf"} if architecture.startswith("segformer") else {default}
    if chosen not in allowed:
        raise ValueError(f"Backend {chosen!r} is not supported for {architecture}; use {sorted(allowed)}")
    return chosen


def _need(module, extra):
    try:
        return __import__(module)
    except Exception as exc:
        raise RuntimeError(f"Cannot import {module}: {exc}. In the active Python environment run: "
                           f'python -m pip install -e ".[{extra}]". Run eddy-doctor for diagnostics.') from exc


class HFWrapper(nn.Module):
    def __init__(self, model, adapter=None):
        super().__init__()
        self.model=model
        if adapter is not None:
            self.adapter=adapter

    def forward(self,x):
        z=self.adapter(x) if hasattr(self,"adapter") else x
        logits=self.model(pixel_values=z).logits
        return F.interpolate(logits,size=x.shape[-2:],mode="bilinear",align_corners=False) if logits.shape[-2:]!=x.shape[-2:] else logits


def _hf_model(spec, pretrained):
    tr=_need("transformers","hf")
    arch=spec.architecture
    if arch.startswith("segformer"):
        variant="b2" if arch=="segformer" else arch.split("_")[-1]
        source=f"nvidia/segformer-{variant}-finetuned-ade-{'640-640' if variant=='b5' else '512-512'}"
        dims=[32,64,160,256] if variant=="b0" else [64,128,320,512]
        depths={"b0":[2,2,2,2],"b2":[3,4,6,3],"b4":[3,8,27,3],"b5":[3,6,40,3]}[variant]
        config=(tr.SegformerConfig.from_dict(spec.hf_config) if spec.hf_config is not None else
                tr.SegformerConfig(num_channels=spec.in_channels,num_labels=spec.num_classes,
                                   hidden_sizes=dims,depths=depths,decoder_hidden_size=256 if variant=="b0" else 768))
        if pretrained:
            # Load actual pretrained weights, rather than just the pretrained config.
            base=tr.SegformerConfig.from_pretrained(source)
            base.num_labels=spec.num_classes
            base.id2label={i:f"class_{i}" for i in range(spec.num_classes)}
            base.label2id={v:k for k,v in base.id2label.items()}
            model=tr.SegformerForSemanticSegmentation.from_pretrained(source,config=base,ignore_mismatched_sizes=True)
            if spec.in_channels!=3:
                old=model.segformer.encoder.patch_embeddings[0].proj
                new=nn.Conv2d(spec.in_channels,old.out_channels,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
                with torch.no_grad():
                    new.weight.copy_(old.weight.sum(1,keepdim=True) if spec.in_channels==1 else
                                     old.weight.mean(1,keepdim=True).repeat(1,spec.in_channels,1,1)*(3/spec.in_channels))
                    if old.bias is not None:
                        new.bias.copy_(old.bias)
                model.segformer.encoder.patch_embeddings[0].proj=new
                model.config.num_channels=spec.in_channels
        else:
            model=tr.SegformerForSemanticSegmentation(config)
        spec.hf_config=model.config.to_dict()
        return HFWrapper(model)
    source="openmmlab/upernet-swin-tiny" if arch.endswith("_t") else "openmmlab/upernet-swin-small"
    if spec.hf_config is not None:
        config=tr.UperNetConfig.from_dict(spec.hf_config)
    elif pretrained:
        config=tr.UperNetConfig.from_pretrained(source)
        config.num_labels=spec.num_classes
        config.id2label={i:f"class_{i}" for i in range(spec.num_classes)}
        config.label2id={v:k for k,v in config.id2label.items()}
        # External configurable loss is used; do not claim auxiliary supervision.
        config.use_auxiliary_head=False
    else:
        backbone=tr.SwinConfig(embed_dim=96,depths=[2,2,6 if arch.endswith("_t") else 18,2],
                               num_heads=[3,6,12,24],window_size=7,
                               out_features=["stage1","stage2","stage3","stage4"])
        config=tr.UperNetConfig(backbone_config=backbone,num_labels=spec.num_classes,use_auxiliary_head=False)
    model=(tr.UperNetForSemanticSegmentation.from_pretrained(source,config=config,ignore_mismatched_sizes=True)
           if pretrained else tr.UperNetForSemanticSegmentation(config))
    adapter=nn.Identity() if spec.in_channels==3 else nn.Conv2d(spec.in_channels,3,1)
    if isinstance(adapter,nn.Conv2d):
        with torch.no_grad():
            adapter.weight.fill_(1/spec.in_channels)
            adapter.bias.zero_()
    spec.hf_config=model.config.to_dict()
    return HFWrapper(model,adapter)


def build_model(spec: ModelSpec, pretrained: bool = False):
    """No silent fallback to another model/backend. Inference never downloads weights."""
    spec.backend=resolve_backend(spec.architecture,spec.backend)
    arch=spec.architecture
    if spec.in_channels<1 or spec.num_classes<2:
        raise ValueError("Need >=1 input channel and >=2 mutually exclusive classes.")
    if spec.backend=="native":
        if pretrained:
            warnings.warn(f"{arch}: no bundled pretrained weights; initializing the custom native model randomly.")
        if arch=="tiny_unet":
            return TinyUNet(spec.in_channels,spec.num_classes)
        if arch=="sam_vit_unet":
            warnings.warn("sam_vit_unet is a legacy custom transformer U-Net, NOT pretrained SAM/MedSAM.")
            return SAMLikeUNet(spec.in_channels,spec.num_classes)
        return SimpleTransUNet(spec.in_channels,spec.num_classes)
    if spec.backend=="timm":
        _need("timm","timm")
        return TimmHRNetSeg(arch,spec.in_channels,spec.num_classes,pretrained)
    if spec.backend=="hf":
        return _hf_model(spec,pretrained)
    smp=_need("segmentation_models_pytorch","smp")
    kwargs=dict(encoder_weights="imagenet" if pretrained else None,
                in_channels=spec.in_channels,classes=spec.num_classes)
    if arch.startswith("unetpp_"):
        return SMPWrapper(smp.UnetPlusPlus(encoder_name="timm-efficientnet-"+arch[-2:],**kwargs))
    if arch.startswith("deeplabv3plus_"):
        return SMPWrapper(smp.DeepLabV3Plus(encoder_name=arch.split("_")[-1],**kwargs))
    variant="b2" if arch=="segformer" else arch.split("_")[-1]
    return SMPWrapper(smp.Segformer(encoder_name="mit_"+variant,**kwargs))


def freeze_batchnorm(model):
    """Freeze running-stat updates, retaining trainable affine parameters.

    Must be called immediately after EVERY model.train() call. This avoids the
    pooled N=1,H=W=1 training error without changing checkpoint structure.
    """
    count=0
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats:
                raise ValueError("Cannot freeze BatchNorm without running statistics.")
            module.eval()
            count+=1
    return count
