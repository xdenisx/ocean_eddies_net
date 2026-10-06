#!/usr/bin/env python3
"""Generate small synthetic GeoTIFFs for an installation smoke test, not research."""
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='demo')
    args=p.parse_args()
    root=Path(args.output)
    for sub in ('images','masks'):
        (root/sub).mkdir(parents=True,exist_ok=True)
    for i in range(4):
        rng=np.random.default_rng(i)
        h,w=96,96
        yy,xx=np.mgrid[:h,:w]
        first=(xx-30)**2+(yy-32)**2<16**2
        second=(xx-68)**2+(yy-64)**2<13**2
        image=(rng.normal(30,2,(h,w))+first*10+second*20).astype(np.float32)
        image[:5]=0
        labels=np.zeros((h,w),np.uint16);labels[first]=100;labels[second]=255
        profile=dict(driver='GTiff',width=w,height=h,count=1,crs='EPSG:3413',
                     transform=from_origin(i*10000,10000,20,20))
        image_path=root/'images'/f'scene_{i}.tif';mask_path=root/'masks'/f'scene_{i}.tif'
        if image_path.exists() or mask_path.exists():
            raise FileExistsError('Demo data already exists; choose another --output.')
        with rasterio.open(image_path,'w',dtype='float32',nodata=0,**profile) as dst:
            dst.write(image,1)
        with rasterio.open(mask_path,'w',dtype='uint16',nodata=65535,**profile) as dst:
            dst.write(labels,1)
    print(f'Synthetic data written to {root}')


if __name__=='__main__':
    main()
