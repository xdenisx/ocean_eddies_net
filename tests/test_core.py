import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ocean_eddy.schema import ClassSchema,make_schema,IGNORE_INDEX
from ocean_eddy.config import TrainConfig
from ocean_eddy.raster import positions,normalize_image
from ocean_eddy.losses import SegmentationLoss,LOSSES
from ocean_eddy.metrics import confusion_matrix,summarize
from ocean_eddy.augment import augment
from ocean_eddy.cli import make_parser,training_config
from ocean_eddy.models import ModelSpec,build_model,freeze_batchnorm
from ocean_eddy.postprocess import blend_weights,smooth_probabilities_inplace,classify,clean_labels


def test_legacy_bytes_unchanged():
    root=Path(__file__).parents[1]
    manifest=json.loads((root/'legacy/SOURCE_MANIFEST.json').read_text())
    assert sum(r['stable'] for r in manifest)==1
    for record in manifest:
        assert hashlib.sha256((root/record['path']).read_bytes()).hexdigest()==record['sha256']


def test_binary_and_multiclass_codes():
    s=make_schema((0,100,255),ignore_values=(65535,))
    raw=np.array([[0,100,255],[65535,np.nan,100.]])
    encoded=s.encode(raw)
    assert encoded.tolist()==[[0,1,2],[-100,-100,1]]
    assert s.decode(encoded).tolist()==[[0,100,255],[65535,65535,100]]
    binary=make_schema((0,255))
    assert binary.encode(np.array([[0,255]])).tolist()==[[0,1]]
    with pytest.raises(ValueError,match='Unknown'):
        binary.encode(np.array([[1]]))


@pytest.mark.parametrize('values,ignore', [((0,0),()),((0,65535),()),((0,255),(255,))])
def test_bad_schema(values,ignore):
    with pytest.raises(ValueError):
        make_schema(values,ignore_values=ignore)


@pytest.mark.parametrize('mode',['none','zscore','minmax','percentile'])
def test_normalization_keeps_invalid_zero(mode):
    img=np.array([[[100,2,3],[4,5,6]]],np.float32)
    valid=np.array([[False,True,True],[True,True,True]])
    result=normalize_image(img,valid,mode)
    assert result[0,0,0]==0
    assert np.isfinite(result).all()


@pytest.mark.parametrize('n,t,s',[(7,32,16),(70,32,16),(65,32,32),(128,64,17)])
def test_positions_full_coverage(n,t,s):
    coverage=np.zeros(n,int)
    for start in positions(n,t,s):
        coverage[start:min(start+t,n)]+=1
    assert (coverage>0).all()
    assert len(positions(n,t,s))==len(set(positions(n,t,s)))


@pytest.mark.parametrize('kind',LOSSES)
@pytest.mark.parametrize('classes',[2,3,5])
def test_multiclass_loss_gradients_and_ignore(kind,classes):
    x=torch.randn(2,classes,16,16,requires_grad=True)
    y=torch.randint(classes,(2,16,16)); y[:,0:2,:]=-100
    loss=SegmentationLoss(classes,kind)(x,y)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad[:,:,:2].abs().sum()==0
    assert torch.isfinite(x.grad).all()
    ignored=torch.full_like(y,-100)
    assert SegmentationLoss(classes,kind)(x,ignored)==0


def test_focal_uses_unweighted_pt():
    logits=torch.tensor([[[[0.]],[[1.]],[[2.]]]],requires_grad=True)
    y=torch.tensor([[[2]]])
    crit=SegmentationLoss(3,'focal',class_weights=[1,2,3],focal_gamma=2)
    p=torch.softmax(logits,1)[0,2,0,0]
    expected=-torch.log(p)*(1-p)**2*3
    torch.testing.assert_close(crit(logits,y),expected)


def test_metrics_exact_and_absent():
    schema=make_schema((0,100,255))
    y=np.array([[0,1,1,2,2,-100]])
    p=np.array([[0,1,0,2,1,0]])
    cm=confusion_matrix(y,p,3)
    assert cm.tolist()==[[1,0,0],[1,1,0],[0,1,1]]
    r=summarize(cm,schema)
    assert r['accuracy']==.6
    assert r['per_class'][1]['iou']==pytest.approx(1/3)
    assert r['per_class'][2]['recall']==.5
    absent=summarize(np.diag([10,0,0]),schema)
    assert absent['per_class'][1]['iou'] is None
    assert absent['foreground_iou'] is None


@pytest.mark.parametrize('level',['light','medium','heavy'])
@pytest.mark.parametrize('channels',[1,4])
@pytest.mark.parametrize('mode',['percentile','zscore','none'])
def test_augmentation_labels_validity_determinism(level,channels,mode):
    img=np.random.default_rng(0).random((channels,64,64),dtype=np.float32)
    valid=np.ones((64,64),bool); valid[:4]=False
    target=np.zeros((64,64),np.int64); target[12:30]=1; target[30:40]=2; target[~valid]=-100
    img[:,~valid]=0
    if mode=='zscore': img[:,valid]=(img[:,valid]-.5)*4
    a=augment(img.copy(),target.copy(),valid.copy(),level,mode,np.random.default_rng(7))
    b=augment(img.copy(),target.copy(),valid.copy(),level,mode,np.random.default_rng(7))
    assert a[0].shape==(channels,64,64)
    assert set(np.unique(a[1])).issubset({-100,0,1,2})
    assert (a[0][:,~a[2]]==0).all()
    assert (a[1][~a[2]]==-100).all()
    assert np.isfinite(a[0]).all()
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])


@pytest.mark.parametrize('mode',['uniform','distance','hann','gaussian'])
def test_blend_positive_symmetric(mode):
    w=blend_weights(32,mode)
    assert w.min()>0 and w.max()<=1
    np.testing.assert_allclose(w,w[::-1])
    np.testing.assert_allclose(w,w[:,::-1])


def test_masked_smoothing_probability_sum():
    p=np.stack([np.full((32,32),.2),np.full((32,32),.3),np.full((32,32),.5)]).astype(np.float32)
    valid=np.ones((32,32),bool); valid[:5]=False
    p[:,~valid]=np.nan
    smooth_probabilities_inplace(p,valid,2)
    np.testing.assert_allclose(p[:,valid].sum(0),1,atol=1e-6)
    np.testing.assert_allclose(p[2,valid],.5,atol=1e-6)
    assert np.isnan(p[:,~valid]).all()
    assert (classify(p,valid)[valid]==2).all()
    with pytest.raises(ValueError,match='binary-only'):
        classify(p,valid,.3)


def test_classwise_cleanup_does_not_overwrite_other_class():
    labels=np.zeros((32,32),np.int32); valid=np.ones_like(labels,bool)
    labels[4:20,4:20]=1; labels[8:10,8:10]=0; labels[12:15,12:15]=2
    labels[28,28]=2
    result=clean_labels(labels,valid,3,2,6)
    assert (result[8:10,8:10]==1).all()
    assert (result[12:15,12:15]==2).all()
    assert result[28,28]==0
    valid[8:10,8:10]=False
    result=clean_labels(labels,valid,3,2,6)
    assert (result[8:10,8:10]==-100).all()


def test_hole_connected_to_image_boundary_not_filled():
    labels=np.ones((8,8),np.int32); valid=np.ones_like(labels,bool)
    labels[0:2,2]=0
    result=clean_labels(labels,valid,2,0,100)
    assert (result[0:2,2]==0).all()


def test_cli_all_promised_flags():
    p=make_parser()
    args=p.parse_args(['train','--architecture','hrnet_w32','--augment_level','heavy',
                      '--early_stop_monitor','val_dice','--min_delta','0.001',
                      '--freeze_batchnorm','--class_values','0','100','255','--num_classes','3'])
    cfg=training_config(args)
    assert cfg.augment_level=='heavy' and cfg.freeze_batchnorm
    assert cfg.min_delta==.001 and cfg.schema().count==3
    assert training_config(p.parse_args(['train','--no_pretrained','--no_drop_last'])).pretrained is False
    assert training_config(p.parse_args(['train','--no-pretrained','--no-drop-last'])).drop_last is False


@pytest.mark.parametrize('command',['train','infer','validate','preprocess','doctor','convert-legacy'])
def test_help(command):
    with pytest.raises(SystemExit) as exc:
        make_parser().parse_args([command,'--help'])
    assert exc.value.code==0


def test_unknown_yaml_keys_fail():
    with pytest.raises(ValueError,match='Unknown configuration'):
        TrainConfig.from_dict({'augument_level':'heavy'})


def test_native_multiclass_and_bn_freeze():
    for arch in ['tiny_unet','transunet']:
        model=build_model(ModelSpec(arch,1,3),pretrained=False)
        model.train(); freeze_batchnorm(model)
        logits=model(torch.randn(1,1,32,32))
        assert logits.shape==(1,3,32,32)
        logits.mean().backward()
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
                assert not m.training
