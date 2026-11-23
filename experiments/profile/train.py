"""Run the fixed-budget native 3D control or a matched pseudo-label experiment."""
import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(run, method):
    with (run/'frames.csv').open() as f:
        splits = [r['split'] for r in csv.DictReader(f)]
    ratio = splits.count('validation')/(splits.count('train')+splits.count('validation'))
    cmd = [sys.executable,'-u','scripts/train.py','--subject','subject','--work-name',method,
           '--stage','base','--model-config','configs/model_rgb.yaml','--seed','1701',
           '--test-ratio',str(ratio),'--blend-start','0','--ssim-weight','0.1',
           '--lpips-weight','0','--prefetch-workers','0','--eval-frames','16',
           '--patience','10','--save-every','250','--min-delta','0.05']
    if method == 'base':
        cmd += ['--iterations','1200','--eval-every','300','--batch-size','8']
    else:
        cmd += ['--init-ply',str(run/'models/subject/base/model.ply'),
                '--iterations','250','--eval-every','250','--lr-scale','0.25']
        # Hold the six real observations per update fixed across all continuations.
        if method == 'real':
            cmd += ['--batch-size','6']
        else:
            cmd += ['--batch-size','8','--repair-per-batch','2','--repair-schedule','none',
                    '--repair-dir',str(run/'labels'/method)]
    print(shlex.join(cmd),flush=True)
    env = dict(os.environ, LIVE3DGS_SUBJECT='subject',PYTHONPATH=str(ROOT/'src'))
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('method')
    a = p.parse_args()
    main(a.run.resolve(),a.method)
