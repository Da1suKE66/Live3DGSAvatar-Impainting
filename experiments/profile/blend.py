"""Exploratory equal-RGB blend of two already prepared, co-registered label sets."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def blend(a,b,out):
    labels=sorted(p.name for p in a.glob('*.png'))
    if not labels or labels!=sorted(p.name for p in b.glob('*.png')):
        raise ValueError('Both teachers must have the same nonempty camera list')
    out.mkdir(exist_ok=False)
    for name in labels:
        x=np.array(Image.open(a/name).convert('RGBA'))
        y=np.array(Image.open(b/name).convert('RGBA'))
        if not np.array_equal(x[:,:,3],y[:,:,3]):
            raise ValueError(f'Masks differ for {name}')
        x[:,:,:3]=(x[:,:,:3].astype('uint16')+y[:,:,:3].astype('uint16')+1)//2
        Image.fromarray(x).save(out/name)
        print(name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('first','second','out'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    blend(a.first,a.second,a.out)
