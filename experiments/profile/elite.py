"""ELITE's independent HuFix generative adaptation prior on the existing geometry."""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor,to_pil_image


def main(run,repo,checkpoint):
    sys.path[:0] = [str(repo),str(repo/'hufix/src')]
    from hufix.src.model import Difix
    torch.manual_seed(42)
    model = Difix(pretrained_path=str(checkpoint),timestep=199,mv_unet=True)
    model.set_eval()
    shared = run/'teachers/shared'
    # Form the white-background views and reference from the same render alpha and training image.
    paths = sorted((shared/'views').glob('*.png'))
    images = []
    for p in paths:
        alpha = to_tensor(Image.open(shared/'render_alpha'/p.name))
        images.append((to_tensor(Image.open(p).convert('RGB')) + .2*(1-alpha)).clamp(0,1))
    idx = next((shared/'refs').glob('base_*.png')).stem.split('_')[-1]
    rgba = Image.open(run/'datasets/subject/images'/(idx+'.png')).convert('RGBA')
    ref = Image.new('RGB',rgba.size,'white')
    ref.paste(rgba,mask=rgba.getchannel('A'))
    with torch.no_grad():
        output = model.sample_batch_multi_tensor(image=torch.stack(images),ref_image=to_tensor(ref)[None],batch_size=1)
    out = run/'teachers/elite'
    out.mkdir(exist_ok=False)
    for name,tensor in zip(paths,output,strict=True):
        to_pil_image(tensor.cpu().clamp(0,1)).save(out/name.name)
        print(name.name,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run','repo','checkpoint'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    main(a.run.resolve(),a.repo.resolve(),a.checkpoint.resolve())
