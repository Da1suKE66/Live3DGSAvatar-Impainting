"""Paired camera evaluation and a geometry-only diagnostic at virtual yaws."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import lpips
import numpy as np
import torch
import yaml
from PIL import Image

from live3dgsavatar.core.camera import orbit_w2c_matrix, rotation_y
from live3dgsavatar.core.io import load_ply
from live3dgsavatar.core.io.ply import read_embedded_config
from live3dgsavatar.core.render import SimpleRasterizer, render_frame
from live3dgsavatar.core.types import Mesh, Camera
from live3dgsavatar.data.pose import rot6d_to_matrix


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_csv(path,rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def score(run,method):
    out = run/'scores'/method
    out.mkdir(parents=True,exist_ok=False)
    ply = run/'models/subject'/method/'model.ply'
    model = load_ply(ply,config=read_embedded_config(ply),device='cuda')
    renderer = SimpleRasterizer()
    metric = lpips.LPIPS(net='vgg').to('cuda').eval()
    topology = Path(yaml.safe_load((run/'local.yaml').read_text())['paths']['mesh_topology'])
    with np.load(topology) as z:
        topo = {k:torch.tensor(z[k],device='cuda') for k in ('faces','uv_faces','uvs')}
    with (run/'frames.csv').open() as f:
        frames = list(csv.DictReader(f))
    bg = torch.zeros((1,3,1,1),device='cuda')
    rows = []
    virtual = []
    for split in ('heldout','subject'):
        data = run/'datasets'/split
        with np.load(data/'render_inputs.npz') as z:
            mesh,weight,k,view = [torch.tensor(z[key],dtype=torch.float32) for key in ('mesh','blend_weight','K','w2v')]
            cam = Camera(K=k[None],w2c=view[None],width=int(z['width']),height=int(z['height'])).to('cuda')
        if split == 'heldout':
            for row in (r for r in frames if r['dataset']==split):
                i = int(row['index'])
                name = f'{i:05d}.png'
                out3d = render_frame(model,renderer,Mesh(verts=mesh[i:i+1].cuda(),**topo),weight[i:i+1].cuda(),cam,bg)
                rgb = out3d.color.clamp(0,1)
                rgba = np.asarray(Image.open(data/'images'/name).convert('RGBA'))
                mask = torch.tensor(rgba[:,:,3]>127,device='cuda')
                gt = torch.tensor(rgba[:,:,:3],device='cuda').permute(2,0,1)[None]/255
                gt = gt*mask
                mse = ((gt-rgb).square()[0,:,mask]).mean().item()
                yaw = float(row['yaw'])
                psnr = -10*np.log10(max(1e-12,mse))
                perceptual = metric(rgb*2-1,gt*2-1).item()
                group = 'front' if abs(yaw)<=10 else ('negative' if yaw<0 else 'positive')
                rows.append(dict(source=row['source'],index=i,yaw=yaw,group=group,psnr=psnr,lpips=perceptual))
                Image.fromarray((rgb[0].permute(1,2,0)*255).round().byte().cpu().numpy()).save(out/name)
        else:
            shared = run/'teachers/shared'
            ref = next((shared/'refs').glob('base_*.png'))
            i = int(ref.stem.split('_')[-1])
            head = torch.from_numpy(rot6d_to_matrix(weight[i:i+1,100:106].numpy())[0]).float()
            for yaw in [-50,-35,-20,20,35,50]:
                orbit = orbit_w2c_matrix(view,mesh[i].mean(0),rotation_y(yaw)[:3,:3]@head.T)
                vcam = Camera(K=k[None],w2c=orbit[None],width=cam.width,height=cam.height).to('cuda')
                result = render_frame(model,renderer,Mesh(verts=mesh[i:i+1].cuda(),**topo),weight[i:i+1].cuda(),vcam,bg)
                silhouette = result.alpha[0,0].cpu().numpy()>.5
                label = f'yaw{yaw:+d}_pitch+0_roll+0_frame{i:05d}.png'
                surface = np.asarray(Image.open(shared/'flame_surface'/label))>127
                virtual.append(dict(yaw=yaw,hole=float((~silhouette&surface).sum()/surface.sum()),
                                    spill=float((silhouette&~surface).sum()/max(silhouette.sum(),1))))
                img = result.color[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
                Image.fromarray(img).save(out/label)
    write_csv(out/'frames.csv',rows)
    write_csv(out/'virtual.csv',virtual)
    groups = {}
    for group in ('negative','positive','side','front'):
        subset = [r for r in rows if r['group']==group or (group=='side' and r['group']!='front')]
        if not subset:
            raise ValueError(f'Missing {group} rows')
        groups[group] = dict(n=len(subset),**{m:float(np.mean([r[m] for r in subset])) for m in ('psnr','lpips')})
    receipt = dict(method=method,weights=sha(ply),frames_sha256=sha(run/'frames.csv'),groups=groups,
                   virtual={m:float(np.mean([r[m] for r in virtual])) for m in ('hole','spill')})
    (out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('method')
    args = p.parse_args()
    score(args.run.resolve(),args.method)
