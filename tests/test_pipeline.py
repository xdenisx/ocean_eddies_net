import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
import torch
from ocean_eddy.cli import main,make_parser
from ocean_eddy.config import TrainConfig
from ocean_eddy.train import train
from ocean_eddy.checkpoint import load_model,convert_legacy
from ocean_eddy.data import split_scenes,index_tiles,EddyDataset
from ocean_eddy.raster import pair_rasters,read_image
from ocean_eddy.models import ModelSpec,build_model
from ocean_eddy.infer import infer_image
from ocean_eddy.schema import make_schema
from ocean_eddy.preprocess import enhance_band


def test_scene_split_and_tile_ignore(synthetic_data,tmp_path):
    images,masks=synthetic_data
    cfg=TrainConfig(images_dir=str(images),masks_dir=str(masks),output_dir=str(tmp_path/'run'),
                    architecture='tiny_unet',pretrained=False,class_values=(0,100,255),ignore_values=(65535,),
                    tile_size=64,stride=32,min_positive_fraction=.9)
    a,b=split_scenes(pair_rasters(images,masks),.25,42)
    assert not {p[0] for p in a}&{p[0] for p in b}
    # Positive filtering must not be applied to validation.
    val=index_tiles(b,cfg,False)
    assert val
    ds=EddyDataset(val,cfg,False)
    x,y=ds[0]
    assert x.shape==(1,64,64)
    assert (y[:5]==-100).all()
    assert torch.isfinite(x).all()


def test_empty_tiles_skipped_even_zero_threshold(tmp_path):
    from rasterio.transform import from_origin
    im=tmp_path/'images'; ma=tmp_path/'masks'; im.mkdir();ma.mkdir()
    for directory,dtype in [(im,'float32'),(ma,'uint8')]:
        with rasterio.open(directory/'x.tif','w',driver='GTiff',width=32,height=32,count=1,
                           dtype=dtype,transform=from_origin(0,1,1,1),crs='EPSG:3413') as dst:
            dst.write(np.zeros((1,32,32),dtype=dtype))
    cfg=TrainConfig(tile_size=32,stride=32,min_valid_fraction=0)
    with pytest.raises(ValueError,match='No usable tiles'):
        index_tiles(pair_rasters(im,ma),cfg,True)


def test_preprocessing_exact_finite_chain_and_nodata():
    import cv2
    from skimage import exposure
    image=np.random.default_rng(8).uniform(2,100,(64,64)).astype(np.float32)
    actual=enhance_band(image,np.ones_like(image,bool),.03,16)
    n=cv2.normalize(image,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    n=np.clip(n,0,1)
    h=exposure.equalize_adapthist(exposure.equalize_hist(n),clip_limit=.03,kernel_size=16)
    expected=cv2.normalize(h.astype(np.float32),None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    np.testing.assert_allclose(actual,expected,atol=1)
    valid=np.ones((64,64),bool); valid[:10]=False
    actual=enhance_band(image,valid,.03,16)
    assert (actual[~valid]==0).all()
    assert enhance_band(np.ones((32,32)),np.ones((32,32),bool)).min()==127


def test_multiclass_full_train_infer_validate(synthetic_data,tmp_path):
    images,masks=synthetic_data
    run=tmp_path/'run';pred=tmp_path/'pred';report=tmp_path/'metrics'
    cfg=TrainConfig(images_dir=str(images),masks_dir=str(masks),output_dir=str(run),
        architecture='tiny_unet',pretrained=False,in_channels=1,class_values=(0,100,255),
        class_names=('background','type_a','type_b'),ignore_values=(65535,),tile_size=64,stride=32,
        epochs=2,batch_size=2,num_workers=0,augment_level='heavy',val_fraction=.25,patience=0,
        min_valid_fraction=.01,device='cpu')
    train(cfg)
    assert (run/'best_checkpoint.pt').is_file()
    assert (run/'training_log.csv').is_file()
    model,schema,ckpt=load_model(run/'best_checkpoint.pt')
    assert schema.values==(0,100,255)
    assert ckpt['model_spec']['backend']=='native'
    main(['infer','--input',str(images),'--checkpoint',str(run/'best_checkpoint.pt'),'--output_dir',str(pred),
          '--tile_size','64','--stride','32','--batch_size','2','--tta','--save_prob','--save_raw_prob',
          '--smooth_sigma','1','--min_object_size','5','--min_hole_size','4','--device','cpu'])
    with rasterio.open(pred/'scene_0_pred.tif') as ps,rasterio.open(images/'scene_0.tif') as src:
        assert ps.crs==src.crs and ps.transform==src.transform
        assert ps.dtypes==('uint16',) and ps.nodata==65535
        assert set(np.unique(ps.read(1))).issubset({0,100,255,65535})
        assert (ps.read_masks(1)[:5]==0).all()
    with rasterio.open(pred/'scene_0_pred_prob.tif') as pp:
        probs=pp.read();valid=pp.read_masks(1)>0
        assert pp.count==3
        np.testing.assert_allclose(probs[:,valid].sum(0),1,atol=2e-6)
    main(['validate','--pred_dir',str(pred),'--mask_dir',str(masks),'--images_dir',str(images),
          '--output_dir',str(report),'--checkpoint',str(run/'best_checkpoint.pt')])
    metrics=json.loads((report/'metrics_summary.json').read_text())
    assert metrics['num_images']==4 and metrics['valid_pixels']>0
    assert metrics['missing_prediction_pixels']==0
    assert len(metrics['per_class'])==3
    assert (report/'confusion_matrix.csv').exists()
    # A complete run may be resumed safely; no unexpected epochs are started.
    cfg.resume=str(run/'last_checkpoint.pt')
    train(cfg)


def test_binary_output255_not_nodata(synthetic_data,tmp_path):
    images,_=synthetic_data
    class Constant(torch.nn.Module):
        def forward(self,x):
            z=torch.zeros(x.shape[0],2,*x.shape[2:],device=x.device)
            z[:,1]=4
            return z
    output=tmp_path/'pred.tif'
    schema=make_schema((0,255))
    infer_image(images/'scene_0.tif',output,Constant(),schema,
        {'normalize':'minmax','zero_is_nodata':True},torch.device('cpu'),64,32,2,save_prob=True)
    with rasterio.open(output) as src:
        values=src.read(1);valid=src.read_masks(1)>0
        assert (values[valid]==255).all() and (values[~valid]==65535).all()


def test_legacy_conversion_strict_roundtrip(tmp_path):
    model=build_model(ModelSpec('transunet',1,2),False).eval()
    old=tmp_path/'old.pt';new=tmp_path/'new.pt'
    torch.save({'model_state_dict':model.state_dict(),'args':{'architecture':'transunet',
                'in_channels':1,'num_classes':2,'mask_value':255,'tile_size':32,'normalize':'percentile'}},old)
    convert_legacy(old,new)
    restored,schema,ckpt=load_model(new)
    x=torch.randn(1,1,32,32)
    with torch.no_grad():
        torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
    assert ckpt['preprocessing']['legacy_normalization']
    assert schema.values==(0,255)


def test_foreground_nodata255_class_priority(tmp_path):
    from ocean_eddy.raster import read_labels
    from rasterio.transform import from_origin
    path=tmp_path/'labels.tif'
    array=np.zeros((16,16),np.uint8);array[5:10]=255
    with rasterio.open(path,'w',driver='GTiff',width=16,height=16,count=1,dtype='uint8',nodata=255,
                       crs='EPSG:3413',transform=from_origin(0,1,1,1)) as dst:
        dst.write(array,1)
    with rasterio.open(path) as src:
        y=read_labels(src,None,make_schema((0,255)),np.ones((16,16),bool))
    assert (y[5:10]==1).all()


def test_binary_training_and_metrics(synthetic_data,tmp_path):
    images,masks=synthetic_data
    for path in masks.glob('*.tif'):
        with rasterio.open(path,'r+') as ds:
            array=ds.read(1);array[array==100]=255;ds.write(array,1)
    cfg=TrainConfig(images_dir=str(images),masks_dir=str(masks),output_dir=str(tmp_path/'binary_run'),
                    architecture='tiny_unet',pretrained=False,tile_size=64,stride=64,epochs=1,
                    batch_size=2,num_workers=0,no_augment=True,ignore_values=(65535,),device='cpu')
    train(cfg)
    model,schema,ckpt=load_model(Path(cfg.output_dir)/'best_checkpoint.pt')
    assert schema.values==(0,255)
    assert len(ckpt['val_stats']['per_class'])==2
    assert 0<=ckpt['val_stats']['iou']<=1


def test_multiband_missing_band_and_preprocessed_valid_zero(tmp_path):
    from rasterio.transform import from_origin
    from ocean_eddy.raster import write_raster
    p=tmp_path/'image.tif'
    data=np.random.default_rng(0).random((4,32,32),dtype=np.float32)
    data[:,0:2]=0;data[1,5,5]=np.nan
    with rasterio.open(p,'w',driver='GTiff',width=32,height=32,count=4,dtype='float32',
                       crs='EPSG:3413',transform=from_origin(0,1,1,1)) as dst:
        dst.write(data)
    with rasterio.open(p) as src:
        x,valid=read_image(src,None,4,True)
        assert x.shape==(4,32,32)
        assert not valid[5,5] and not valid[:2].any()
        with pytest.raises(ValueError,match='no silent channel padding'):
            read_image(src,None,5)
        out=np.zeros((4,32,32),np.uint8)
        preserved=np.ones((32,32),bool);preserved[0]=False
        write_raster(tmp_path/'enhanced.tif',out,src,preserved,None)
    with rasterio.open(tmp_path/'enhanced.tif') as src:
        x,valid=read_image(src,None,4,False)
        assert valid[1:].all() and not valid[0].any()
        assert src.nodata is None


def test_validator_rejects_missing_coverage(synthetic_data,tmp_path):
    from ocean_eddy.raster import write_raster
    images,masks=synthetic_data
    pred=tmp_path/'pred'
    schema=make_schema((0,100,255),ignore_values=(65535,))
    for path in images.glob('*.tif'):
        with rasterio.open(path) as src:
            _,valid=read_image(src,None,1,True)
            valid[20:22,20:22]=False
            labels=np.zeros(valid.shape,np.int64)
            write_raster(pred/(path.stem+'_pred.tif'),schema.decode(labels,valid),src,valid,65535,
                         tags={'class_schema':schema.to_dict(),'preprocessing':{'zero_is_nodata':True}})
    with pytest.raises(ValueError,match='lack predictions'):
        main(['validate','--pred_dir',str(pred),'--mask_dir',str(masks),'--images_dir',str(images),
              '--output_dir',str(tmp_path/'metrics')])
    main(['validate','--pred_dir',str(pred),'--mask_dir',str(masks),'--images_dir',str(images),
          '--output_dir',str(tmp_path/'metrics'),'--missing_prediction_policy','ignore'])
    report=json.loads((tmp_path/'metrics/metrics_summary.json').read_text())
    assert report['missing_prediction_pixels']==16
