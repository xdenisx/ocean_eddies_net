from pathlib import Path
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin


def pytest_sessionstart(session):
    torch.set_num_threads(2)


@pytest.fixture
def synthetic_data(tmp_path):
    images=tmp_path/'images'; masks=tmp_path/'masks'
    images.mkdir(); masks.mkdir()
    for i in range(4):
        rng=np.random.default_rng(i)
        h,w=70,66
        image=rng.uniform(1,10,(1,h,w)).astype('float32')
        image[:,:3,:]=0; image[:,3:5,:]=np.nan
        mask=np.zeros((h,w),np.uint16)
        mask[12:30,10:25]=100
        mask[38:58,35:55]=255
        mask[5:8,5:8]=65535
        transform=from_origin(1000+i*10000,2000,20,20)
        with rasterio.open(images/f'scene_{i}.tif','w',driver='GTiff',width=w,height=h,count=1,
                           dtype='float32',crs='EPSG:3413',transform=transform,nodata=np.nan) as dst:
            dst.write(image)
        with rasterio.open(masks/f'scene_{i}.tif','w',driver='GTiff',width=w,height=h,count=1,
                           dtype='uint16',crs='EPSG:3413',transform=transform,nodata=65535) as dst:
            dst.write(mask,1)
    return images,masks
