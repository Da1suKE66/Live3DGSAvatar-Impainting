"""Check the published source and this subject's required inference inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def git(repo,*args):
    return subprocess.check_output(['git','-C',repo,*args],text=True).strip()


def probe(method,repo,data):
    files=git(repo,'ls-files').splitlines()
    print('method',method,'upstream',git(repo,'remote','get-url','origin'),'commit',git(repo,'rev-parse','HEAD'))
    with np.load(data/'render_inputs.npz',allow_pickle=False) as z:
        print('subject mesh',z['mesh'].shape,'drive',z['blend_weight'].shape,'cameras',z['w2v'].shape)
    if method=='gaf':
        code=[p for p in files if p.endswith(('.py','.sh','.ckpt','.safetensors'))]
        print('tracked files',json.dumps(files))
        print('inference/training/weights candidates',code)
        print('release note',[s for s in (repo/'README.md').read_text().splitlines() if 'coming soon' in s.lower()])
        return bool(code)
    if method=='cap4d':
        needed=[repo/'data/assets/flame/flame2023_no_jaw.pkl',data/'fit.npz',data/'reference_images.json']
        sample=repo/'examples/input/lincoln/fit.npz'
        with np.load(sample,allow_pickle=False) as z:
            print('official example keys',sorted(z.files),'expression shape',z['expr'].shape)
        print('CAP4DInferenceDataset instantiates CAP4DFlameSkinner before decoding reference inputs')
    elif method=='diffusion-avatars':
        needed=[data/'nersemble',data/'rendering_data']
        print('published subject checkpoints',[line.split('|')[1].strip() for line in (repo/'README.md').read_text().splitlines() if '| [DA-' in line])
        print('author input requires matched per-identity rasterized canonical-coordinate, normal and depth sequences and camera calibration')
    elif method=='morphable-diffusion':
        needed=[repo/'assets/FLAME2020/generic_model.pkl',repo/'third_party/metrical-tracker/output/config/mesh/00001.ply']
        print('generate_face.sh first runs MICA and metrical-tracker on the expression frame')
        with (repo/'assets/head_template_mesh.obj').open() as f:
            print('template FLAME vertices',sum(line.startswith('v ') for line in f))
    else:
        raise ValueError(method)
    for p in needed:
        print('present' if p.exists() else 'missing',p.name)
    return all(p.exists() for p in needed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('method',choices=['gaf','cap4d','diffusion-avatars','morphable-diffusion'])
    p.add_argument('repo',type=Path)
    p.add_argument('data',type=Path)
    a=p.parse_args()
    ok=probe(a.method,a.repo,a.data)
    print('input check:', 'available for next stage' if ok else 'blocked before same-subject inference')
    sys.exit(0 if ok else 2)
