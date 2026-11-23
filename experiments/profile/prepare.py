"""Prepare a yaw-censored subject from a timestamp-partitioned MediaPipe capture."""
import argparse
import csv
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
import yaml
from PIL import Image

AXES = np.diag([1., -1., -1.])
STABLE = [4,5,6,8,10,33,46,70,93,105,107,127,133,168,195,197,234,263,276,300,323,334,336,356,362,454]


def select(capture, split):
    path = capture / 'datasets' / split
    tracks = json.loads((path / 'tracks.json').read_text())
    valid = [r for r in tracks if r['tracking_ok']]
    front = [r for r in valid if abs(r['yaw']) <= 10]
    if split == 'test':
        selected = sorted(front[::10] + [r for r in valid if abs(r['yaw']) >= 15], key=lambda r:r['local_id'])
    else:
        limit = 192 if split == 'train' else 32
        selected = [front[i] for i in np.linspace(0,len(front)-1,min(len(front),limit),dtype=int)]
    if not selected:
        raise ValueError(f"No eligible {split} frames")
    with np.load(path / 'render_inputs.npz') as manifest:
        ids = [r['local_id'] for r in selected]
        drive = manifest['blend_weight'][ids]
        camera = {k:manifest[k] for k in ('K','w2v','width','height')}
    return selected, drive, camera


def rotate(weight):
    rows = weight[:,52:58].reshape(-1,2,3)
    return np.concatenate([rows,np.cross(rows[:,:1],rows[:,1:2])],axis=1)


def fit(capture, assets, rows, drive, camera):
    # Estimate expression geometry from the selected training photos alone.
    from mediapipe.tasks.python import vision, BaseOptions
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(
        model_asset_path=str(assets/'models/face_landmarker.task')), num_faces=1)
    canonical = np.array([[float(x)/100 for x in line.split()[1:4]]
        for line in (assets/'mesh/canonical_face_model.obj').read_text().splitlines() if line.startswith('v ')])
    k = camera['K']
    rotations = rotate(drive)
    local = []
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for row, rot, weight in zip(rows,rotations,drive):
            file = capture/'datasets/train/images'/f"{row['local_id']:05d}.png"
            rgb = np.asarray(Image.open(file).convert('RGB'))
            h,w = rgb.shape[:2]
            pts = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks
            if not pts:
                raise ValueError(f"Landmarks missing in training frame {row['local_id']}")
            xyz = np.array([(p.x*w,p.y*h,p.z*w) for p in pts[0][:len(canonical)]])
            t = weight[-3:]
            z = ((canonical @ rot.T + t) @ AXES)[:,2]
            med = np.median(z[STABLE])
            z = .75*z + .25*(med + (xyz[:,2]-np.median(xyz[STABLE,2]))*med/k[0,0])
            rays = np.column_stack((xyz[:,:2],np.ones(len(xyz)))) @ np.linalg.inv(k).T
            local.append(((rays*z[:,None]) @ AXES - t) @ rot)
    x = drive[:,:52].astype('f8')
    mean, rest = x.mean(0), np.mean(local,axis=0)
    xc = x-mean
    basis = np.linalg.solve(xc.T@xc + 2*np.eye(x.shape[1]),xc.T@(local-rest).reshape(len(x),-1))
    return dict(mean=mean,rest=rest,basis=basis)


def encode(drive, rig):
    local = rig['rest']+((drive[:,:52]-rig['mean'])@rig['basis']).reshape(len(drive),-1,3)
    mesh = local@rotate(drive).transpose(0,2,1) + drive[:,None,-3:]
    # Native pose helpers expect the two rotation rows at offsets 100..105.
    weight = np.zeros((len(drive),129),dtype='f4')
    weight[:,:52],weight[:,100:106],weight[:,106:109] = drive[:,:52],drive[:,52:58],drive[:,58:61]
    return mesh.astype('f4'),weight


def main(capture, assets, run, syncfix):
    run.mkdir(parents=True,exist_ok=True)
    if (run/"rig.npz").exists() or (run/"local.yaml").exists():
        raise FileExistsError(f"Use a fresh run directory: {run}")
    train, drives, cam = select(capture,'train')
    rig = fit(capture,assets,train,drives,cam)
    np.savez(run/'rig.npz',**rig)
    records = []
    for target, splits in [('subject',['train','validation']),('heldout',['test'])]:
        dataset = run/'datasets'/target
        (dataset/'images').mkdir(parents=True,exist_ok=False)
        blocks, weights, names = [],[],[]
        for split in splits:
            rows, w, fields = select(capture,split)
            if not all(np.array_equal(fields[k],cam[k]) for k in cam):
                raise ValueError('Camera changed between partitions')
            mesh, encoded = encode(w,rig)
            blocks.append(mesh)
            weights.append(encoded)
            for r in rows:
                name = f'{len(names):05d}'
                original = f"{r['local_id']:05d}.png"
                (dataset/'images'/(name+'.png')).symlink_to((capture/'datasets'/split/'images'/original).resolve())
                records.append(dict(dataset=target,index=len(names),split=split,source=r['source_index'],
                                    local_id=r['local_id'],seconds=r['seconds'],yaw=r['yaw']))
                names.append(name)
        np.savez(dataset/'render_inputs.npz',mesh=np.concatenate(blocks),blend_weight=np.concatenate(weights),
                 names=np.array(names),**cam)
    with (run/'frames.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    topo = capture/'topology.npz'
    cfg = {'paths':dict(data_root=str(run/'datasets'),output_dir=str(run/'output'),
                        models_dir=str(run/'models'),mesh_topology=str(topo))}
    nval = sum(r['split']=='validation' for r in records)
    cfg['train'] = {'pipeline': {'test_ratio': nval/(len(train)+nval)}}
    if syncfix:
        cfg['syncfix'] = syncfix
    print(yaml.safe_dump(cfg))
    (run/'local.yaml').write_text(yaml.safe_dump(cfg))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('capture','assets','run'):
        p.add_argument('--'+arg,type=Path,required=True)
    for arg in ('repo','python','model'):
        p.add_argument('--syncfix-'+arg,type=Path)
    args = p.parse_args()
    values = {k:getattr(args,'syncfix_'+k) for k in ('repo','python','model')}
    if any(values.values()) and not all(values.values()):
        p.error('Supply all three SyncFix paths together')
    syncfix = {k:str(v.absolute()) for k,v in values.items() if v}
    main(args.capture.resolve(),args.assets.resolve(),args.run.resolve(),syncfix)
