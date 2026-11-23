"""Feed the released Morphable Diffusion sampler the captured face surface and native view orientations."""
import argparse
import json
from pathlib import Path
import os
import sys

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import yaml

from live3dgsavatar.core.camera import orbit_w2c_matrix, rotation_y
from live3dgsavatar.data.pose import rot6d_to_matrix

ANGLES = [-85,-70,-60,-50,-35,-20,-10,-4,4,10,20,35,50,60,70,85]
TARGETS = {-50,-35,-20,20,35,50}
PTS = [33,133,362,263,4,61,291,152,10]


def project(vertices,k,rt):
    xyz = (vertices@rt[:3,:3].T+rt[:3,3])@k[:3,:3].T
    return xyz[:,:2]/xyz[:,2:]


def inputs(run):
    shared = run/'teachers/shared'
    idx = json.loads((shared/'selected.json').read_text())['base_frame']
    with np.load(run/'datasets/subject/render_inputs.npz') as d:
        verts = d['mesh'][idx]
        view,k = torch.from_numpy(d['w2v']),d['K']
        head = torch.tensor(rot6d_to_matrix(d['blend_weight'][idx:idx+1,100:106])[0]).float()
    center = torch.from_numpy(verts.mean(0))
    local = ((verts-center.numpy())@head.numpy())*2.7
    kt = np.eye(4,dtype='f4')
    kt[0,0]=kt[1,1]=1545.2376
    kt[:2,2]=128.
    def registration(yaw):
        native = view if yaw is None else orbit_w2c_matrix(view,center,rotation_y(yaw)[:3,:3]@head.T)
        rt = np.column_stack(((native[:3,:3]@head).numpy(),[0.,0.,4.5])).astype('f4')
        src = project(verts[PTS],k,native.numpy())
        dst = project(local[PTS],kt,rt)
        warp,_ = cv2.estimateAffinePartial2D(src,dst,method=cv2.LMEDS)
        return rt,warp
    rgba = Image.open(run/'datasets/subject/images'/f'{idx:05d}.png').convert('RGBA')
    ref = Image.new('RGB',rgba.size,'white')
    ref.paste(rgba,mask=rgba.getchannel('A'))
    _,crop = registration(None)
    image = cv2.warpAffine(np.asarray(ref),crop,(256,256),borderValue=(255,255,255))
    img = torch.tensor(image).float()/127.5-1
    cameras,back = [],{}
    for angle in ANGLES:
        rt,warp = registration(angle)
        cameras.append(rt)
        back[angle] = cv2.invertAffineTransform(warp)
    with np.load(yaml.safe_load((run/'local.yaml').read_text())['paths']['mesh_topology']) as z:
        tri = local[z['faces']]
    # Use unique occupied surface voxels (the sampler pools features in sparse 3D voxels).
    points = np.concatenate([(tri[:,0]*i+tri[:,1]*j+tri[:,2]*(3-i-j))/3.
                             for i in range(4) for j in range(4-i)])
    dhw = points[:,[2,1,0]]
    bounds = np.stack([points.min(0),points.max(0)])
    coords,keep = np.unique(np.rint((dhw-dhw.min(0))/.005).astype('int32'),axis=0,return_index=True)
    shape = (np.ceil(np.ptp(dhw,axis=0)/.005).astype('int32')|3)+1
    batch = dict(input_image=img,target_image=img.expand(16,-1,-1,-1),input_elevation=torch.zeros(1),
        input_azimuth=torch.zeros(1),target_elevation=torch.zeros(16),target_azimuth=torch.zeros(16),
        target_K=torch.from_numpy(kt).expand(16,-1,-1),target_RT=torch.tensor(np.stack(cameras)),
        vertices=torch.tensor(points[keep]),out_sh=torch.tensor(shape),coord=torch.tensor(coords),bounds=torch.tensor(bounds))
    return {key:value[None].contiguous().cuda() for key,value in batch.items()},back,idx


@torch.no_grad()
def main(run,repo,weights,clip):
    sys.path.insert(0,str(repo))
    os.chdir(repo)
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.morphable_diffusion import SyncDDIMSampler
    torch.manual_seed(1701)
    cfg = OmegaConf.load('configs/facescape.yaml')
    cfg.model.params.clip_image_encoder_path = str(clip)
    model = instantiate_from_config(cfg.model)
    state = torch.load(weights,map_location='cpu',weights_only=False,mmap=True)
    missing,unexpected = model.load_state_dict(state['state_dict'],strict=False)
    print('state keys',len(missing),'missing',missing[:12],len(unexpected),'extra',unexpected[:12],flush=True)
    if any(name in dict(model.named_parameters()) for name in missing):
        raise ValueError('Checkpoint lacks trainable model parameters')
    del state
    model = model.eval().cuda()
    batch,back,idx = inputs(run)
    sampler = SyncDDIMSampler(model,50)
    result = model.sample(sampler,batch,2.0,2)
    images = ((result[0].clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).cpu().numpy()
    out = run/'teachers/morphable'
    out.mkdir(exist_ok=False)
    for angle,img in zip(ANGLES,images,strict=True):
        if angle in TARGETS:
            name=f'yaw{angle:+d}_pitch+0_roll+0_frame{idx:05d}.png'
            Image.fromarray(cv2.warpAffine(img,back[angle],(512,512),borderValue=(255,255,255))).save(out/name)
            print(name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','repo','weights','clip'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    main(*(getattr(a,k).resolve() for k in ('run','repo','weights','clip')))
