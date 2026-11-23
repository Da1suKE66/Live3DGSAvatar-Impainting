"""Isolate the FFHQ prior used by FATE before its full-head GAN inversion."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tf
# BasicSR 1.4.2 uses the old torchvision module name for rgb_to_grayscale.
sys.modules.setdefault('torchvision.transforms.functional_tensor',tf)
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean


@torch.no_grad()
def main(run,model_path,landmarker):
    from mediapipe.tasks.python import vision, BaseOptions
    import mediapipe as mp
    net = GFPGANv1Clean(out_size=512,num_style_feat=512,channel_multiplier=2,
        decoder_load_path=None,fix_decoder=False,num_mlp=8,input_is_latent=True,
        different_w=True,narrow=1,sft_half=True)
    net.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True)['params_ema'],strict=True)
    net.eval().cuda()
    torch.manual_seed(1701)
    shared = run/'teachers/shared'
    out = run/'teachers/fate-prior'
    out.mkdir(exist_ok=False)
    canonical = np.float32([[192.98138,239.94708],[318.90277,240.1936],[256.63416,314.01935],
                            [201.26117,371.41043],[313.08905,371.15118]])
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(landmarker)))
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for path in sorted((shared/'views').glob('*.png')):
            rgb = np.array(Image.open(path).convert('RGB'))
            points = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)).face_landmarks
            if not points:
                raise ValueError('Cannot align '+path.name)
            pts = np.float32([(p.x*rgb.shape[1],p.y*rgb.shape[0]) for p in points[0]])
            key = np.float32([(pts[33]+pts[133])/2,(pts[263]+pts[362])/2,pts[4],pts[61],pts[291]])
            mat,_ = cv2.estimateAffinePartial2D(key,canonical,method=cv2.LMEDS)
            aligned = cv2.warpAffine(rgb,mat,(512,512),borderValue=(204,204,204))
            pred = net(tf.to_tensor(aligned)[None].cuda()*2-1,return_rgb=False,weight=.5)[0]
            restored = np.asarray(tf.to_pil_image(((pred[0]+1)/2).clamp(0,1).cpu()))
            back = cv2.invertAffineTransform(mat)
            # Outside the aligned crop keep the input unchanged.
            blended = cv2.warpAffine(restored,back,(rgb.shape[1],rgb.shape[0]),dst=rgb.copy(),
                                     borderMode=cv2.BORDER_TRANSPARENT)
            Image.fromarray(blended).save(out/path.name)
            print(path.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','model','landmarker'):
        p.add_argument(k,type=Path)
    args=p.parse_args()
    main(args.run.resolve(),args.model.resolve(),args.landmarker.resolve())
