"""Invert one observed frontal image and register the EG3D turntable to native cameras."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import lpips
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from live3dgsavatar.core.camera import orbit_w2c_matrix, rotation_y
from live3dgsavatar.data.pose import rot6d_to_matrix

ANCHORS = [33,133,362,263,4,61,291,152,10]


def detect(detector,image):
    arr = np.ascontiguousarray(image.convert('RGB'))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=arr)).face_landmarks
    if not result:
        raise ValueError('Unable to register generated face landmarks')
    return np.array([(p.x*image.width,p.y*image.height) for p in result[0]],dtype='float32')


def similarity(source,target):
    matrix,_ = cv2.estimateAffinePartial2D(np.float32(source),np.float32(target),method=cv2.LMEDS)
    if matrix is None:
        raise ValueError('No similarity alignment')
    return matrix


def main(run,repo,model,assets):
    sys.path.insert(0,str(repo/'eg3d'))
    from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
    import legacy
    from torch_utils.ops import bias_act, upfirdn2d, filtered_lrelu
    # Use the authors' PyTorch reference operators on newer PyTorch/CUDA.
    for op in [bias_act,upfirdn2d,filtered_lrelu]:
        op._init = lambda: False

    from mediapipe.tasks.python import vision, BaseOptions
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(assets/'models/face_landmarker.task'))))
    with model.open('rb') as f:
        gen = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to('cuda')
    metric = lpips.LPIPS(net='vgg').requires_grad_(False).cuda().eval()
    torch.manual_seed(1701)
    pivot = torch.tensor(gen.rendering_kwargs.get('avg_camera_pivot',[0,0,0]),device='cuda')
    radius = gen.rendering_kwargs.get('avg_camera_radius',2.7)
    intr = FOV_to_intrinsics(18.837,device='cuda').reshape(1,9)
    def camera(angle):
        pose = LookAtPoseSampler.sample(np.pi/2+np.deg2rad(angle),np.pi/2,pivot,radius=radius,device='cuda')
        return torch.cat([pose.reshape(1,16),intr],dim=1)

    shared = run/'teachers/shared'
    idx = json.loads((shared/'selected.json').read_text())['base_frame']
    rgba = Image.open(run/'datasets/subject/images'/f'{idx:05d}.png').convert('RGBA')
    pts = detect(detector,rgba)
    anchors = [(pts[33]+pts[133])/2,(pts[263]+pts[362])/2,(pts[61]+pts[291])/2]
    mat = similarity(anchors,[[180,205],[332,205],[256,325]])
    aligned = cv2.warpAffine(np.array(rgba),mat,(512,512))
    target = torch.tensor(aligned[:,:,:3],device='cuda').permute(2,0,1)[None]/127.5-1
    mask = torch.tensor(aligned[:,:,3],device='cuda')[None,None]/255
    cam = camera(0.)
    with torch.no_grad():
        w_init = gen.mapping(torch.zeros((1,gen.z_dim),device='cuda'),cam,truncation_psi=0)
    w = w_init.clone().requires_grad_()
    optimizer = torch.optim.Adam([w],lr=.025)
    diagnostic = run/'teachers/eg3d-inversion'
    diagnostic.mkdir(exist_ok=True)
    for step in range(160):
        torch.manual_seed(1701)
        pred = gen.synthesis(w,cam,neural_rendering_resolution=64,noise_mode='const')['image']
        residual = (pred-target)*mask
        # VGG compares only the face oval; both backgrounds are identical.
        loss = residual.abs().sum()/(3*mask.sum())
        loss = loss + .4*metric(F.avg_pool2d(pred*mask,2),F.avg_pool2d(target*mask,2)).mean()
        loss = loss + .01*(w-w_init).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%20==0 or step==159:
            print('inversion',step,round(loss.item(),6),flush=True)
    torch.save(w.detach().cpu(),diagnostic/'w_plus.pt')
    angles = (-50,-35,-20,20,35,50)
    imgs, points = {}, {}
    with torch.no_grad():
        for angle in (*angles,0):
            torch.manual_seed(1701)
            res = gen.synthesis(w,camera(angle),neural_rendering_resolution=128,noise_mode='const')['image'][0]
            im = Image.fromarray(((res.clamp(-1,1)+1)*127.5).permute(1,2,0).byte().cpu().numpy())
            im.save(diagnostic/f'{angle:+d}.png')
            imgs[angle], points[angle] = im, detect(detector,im)[ANCHORS]
    detector.close()
    with np.load(run/'datasets/subject/render_inputs.npz') as d:
        verts = torch.tensor(d['mesh'][idx])
        head = torch.tensor(rot6d_to_matrix(d['blend_weight'][idx:idx+1,100:106])[0]).float()
        view, k = torch.tensor(d['w2v']), torch.tensor(d['K'])
    def target_points(angle):
        orbit = orbit_w2c_matrix(view,verts.mean(0),rotation_y(angle)[:3,:3]@head.T)
        xy = (verts[ANCHORS]@orbit[:3,:3].T+orbit[:3,3])@k.T
        return (xy[:,:2]/xy[:,2:]).numpy()
    # The two projects use different yaw conventions. Resolve the sign once across all views.
    errors, mats = {}, {}
    for sign in (1,-1):
        errors[sign] = []
        for angle in angles:
            pts = points[angle*sign]
            desired = target_points(angle)
            affine = similarity(pts,desired)
            mats[sign,angle] = affine
            residual = pts@affine[:,:2].T+affine[:,2]-desired
            errors[sign].append(float(np.sqrt(np.mean(residual**2))))
    sign = min(errors,key=lambda s:np.mean(errors[s]))
    print('landmark RMSE in pixels',errors,'chosen yaw sign',sign,flush=True)
    out = run/'teachers/eg3d'
    out.mkdir(exist_ok=False)
    for angle in angles:
        label = f'yaw{angle:+d}_pitch+0_roll+0_frame{idx:05d}.png'
        warp = cv2.warpAffine(np.array(imgs[sign*angle]),mats[sign,angle],rgba.size,
                              borderMode=cv2.BORDER_CONSTANT,borderValue=(204,204,204))
        Image.fromarray(warp).save(out/label)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run','repo','model','assets'):
        p.add_argument(key,type=Path)
    a = p.parse_args()
    main(a.run.resolve(),a.repo.resolve(),a.model.resolve(),a.assets.resolve())
