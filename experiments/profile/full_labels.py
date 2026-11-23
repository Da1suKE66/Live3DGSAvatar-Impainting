"""Compose a candidate teacher's RGB with exactly the labels admitted by native SyncFix."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageFilter


def stage(run,exp):
    native = run/'full-repo'
    shared = exp/'teachers/shared'
    shared.mkdir(parents=True,exist_ok=False)
    (exp/'datasets').symlink_to(native/'datasets')
    original = native/'output/repair/subject'
    (shared/'refs').symlink_to(original/'refs')
    (shared/'render_alpha').symlink_to(original/'render_alpha')
    views = shared/'views'
    views.mkdir()
    for p in sorted((native/'datasets/subject/repair/repairs').glob('*.png')):
        (views/p.name).symlink_to(original/'views'/p.name)
    print('admitted cameras',len(list(views.glob('*.png'))))


def compose(run,exp,method):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    from make_repairs import _fill_dark_holes
    native = run/'full-repo/datasets/subject/repair/repairs'
    out = exp/'labels'/method
    out.mkdir(parents=True,exist_ok=False)
    for p in sorted(native.glob('*.png')):
        name = p.name+'.png' if method=='edgar' else p.name
        source = Image.open(exp/'teachers'/method/name).convert('RGB')
        rgb = np.array(source,dtype='f4')
        blur = np.array(source.filter(ImageFilter.GaussianBlur(radius=1.0)),dtype='f4')
        rgb = np.clip(rgb+.35*(rgb-blur),0,255).astype('uint8')
        alpha = np.asarray(Image.open(p).convert('RGBA'))[...,3]
        rgb = _fill_dark_holes(rgb,alpha).copy()
        rgb[alpha==0] = 0
        Image.fromarray(np.dstack((rgb,alpha))).save(out/p.name)
        print(method,p.name,int((alpha>127).sum()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('experiment',type=Path)
    p.add_argument('stage_or_method')
    a=p.parse_args()
    run,exp = a.run.resolve(),a.experiment.resolve()
    if a.stage_or_method=='stage':
        stage(run,exp)
    else:
        compose(run,exp,a.stage_or_method)
