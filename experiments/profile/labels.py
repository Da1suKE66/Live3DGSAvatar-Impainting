"""Prepare unsharpened, common-silhouette labels for the native repair loader."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(run, method, source):
    shared = run/'teachers/shared'
    target = run/'labels'/method
    target.mkdir(parents=True,exist_ok=False)
    views = sorted((shared/'views').glob('*.png'))
    if len(views) != 6:
        raise ValueError('Expected six registered camera views')
    for v in views:
        p = source/(v.name+'.png' if method=='edgar' else v.name)
        img = np.array(Image.open(p).convert('RGB'))
        mask = np.asarray(Image.open(shared/'render_alpha'/v.name).convert('L')) > 127
        if img.shape[:2] != mask.shape:
            raise ValueError(f'Teacher changed resolution for {v.name}')
        img[~mask] = 0
        Image.fromarray(np.dstack([img,mask.astype('uint8')*255])).save(target/v.name)
        print(method,v.name,int(mask.sum()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('method')
    p.add_argument('source',type=Path)
    a = p.parse_args()
    prepare(a.run,a.method,a.source)
