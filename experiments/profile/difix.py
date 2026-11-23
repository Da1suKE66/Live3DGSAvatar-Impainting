"""Run NVIDIA's reference-conditioned Difix pipeline on registered profile views."""
import argparse
from pathlib import Path
import sys

from PIL import Image
import torch


def main(run, repo, model):
    sys.path.insert(0,str(repo/'src'))
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained(str(model),trust_remote_code=True,local_files_only=True).to('cuda')
    shared = run/'teachers/shared'
    ref = Image.open(next((shared/'refs').glob('base_*.png'))).convert('RGB')
    out = run/'teachers/difix'
    out.mkdir(exist_ok=False)
    for path in sorted((shared/'views').glob('*.png')):
        with torch.inference_mode():
            image = pipe('remove degradation',image=Image.open(path).convert('RGB'),ref_image=ref,
                height=512,width=512,num_inference_steps=1,timesteps=[199],guidance_scale=0.,
                generator=torch.Generator('cuda').manual_seed(1701)).images[0]
        image.save(out/path.name)
        print(path.name,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run','repo','model'):
        p.add_argument(key,type=Path)
    a = p.parse_args()
    main(a.run.resolve(),a.repo.resolve(),a.model.resolve())
