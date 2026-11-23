"""Prepare white-background views and run the released EDGAR six-expert refiner."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import shlex
import numpy as np
from PIL import Image


def main(run, repo, weights):
    shared = run/'teachers/shared'
    data = run/'teachers/edgar-input'
    data.mkdir(exist_ok=True)
    for path in sorted((shared/'views').glob('*.png')):
        # Native views already contain 0.8-gray premultiplied background.
        rgb = np.array(Image.open(path).convert('RGB'),dtype='float32')
        alpha = np.array(Image.open(shared/'render_alpha'/path.name),dtype='float32')/255
        white = np.clip(rgb+51*(1-alpha[...,None]),0,255)
        Image.fromarray(white.round().astype('uint8')).save(data/path.name)
    base = next((shared/'refs').glob('base_*.png')).stem.split('_')[-1]
    original = Image.open(run/'datasets/subject/images'/(base+'.png')).convert('RGBA')
    ref = Image.new('RGB',original.size,'white')
    ref.paste(original,mask=original.getchannel('A'))
    ref.save(data.parent/'edgar-reference.png')
    cmd = [sys.executable,'-m','edgar.infer','--input',data,'--reference',data.parent/'edgar-reference.png',
           '--checkpoint',weights/'edgar_exp6.pt','--sapiens',weights/'sapiens-pretrain-0.3b-torchscript/sapiens_0.3b_epoch_1600_torchscript.pt2',
           '--base-model',weights/'sd-turbo','--output',run/'teachers/edgar','--preprocessing','application','--seed','1701']
    env = dict(os.environ,PYTHONPATH=os.pathsep.join([str(repo/'src'),os.environ.get('PYTHONPATH','')]))
    print(shlex.join([str(x) for x in cmd]),flush=True)
    subprocess.run(cmd,env=env,check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run','repo','weights'):
        p.add_argument(key,type=Path)
    a = p.parse_args()
    main(a.run.resolve(),a.repo.resolve(),a.weights.resolve())
