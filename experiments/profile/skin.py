"""Check sensitivity to the original profile matte using an independent skin segmentation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def prepare(wide, model):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    out = wide / "skin"
    masks = out / "masks"
    masks.mkdir(parents=True, exist_ok=False)
    coverage = []
    options = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                          output_category_mask=True, output_confidence_masks=False)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        labels = list(segmenter.labels)
        category = labels.index("face-skin")
        for row in read(wide / "frames.csv"):
            filename = f"{int(row['index']):05d}.png"
            src = wide / "datasets/heldout/images" / filename
            with Image.open(src) as photo:
                input_rgb = np.array(photo.convert("RGB"))
                oval = np.asarray(photo.getchannel("A")) > 127
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_rgb)
            result = segmenter.segment(frame)
            skin = result.category_mask.numpy_view().reshape(oval.shape) == category
            if not skin.any():
                raise ValueError(f"No semantic face pixels in {filename}")
            overlap = int((skin & oval).sum())
            path = masks / filename
            Image.fromarray(skin.astype("uint8") * 255).save(path)
            coverage.append(dict(index=row["index"], source=row["source"], yaw=row["yaw"],
                                 oval_pixels=int(oval.sum()), skin_pixels=int(skin.sum()),
                                 overlap_pixels=overlap, skin_covered_by_oval=overlap / int(skin.sum()),
                                 mask_sha256=digest(path)))
    write(out / "coverage.csv", coverage)
    receipt = dict(model_sha256=digest(model), categories=labels, target=category,
                   frames_sha256=digest(wide / "frames.csv"), coverage_sha256=digest(out / "coverage.csv"))
    (out / "segmenter.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def score(wide, methods):
    import lpips
    import torch
    metric = lpips.LPIPS(net="vgg").cuda().eval()
    coverage = read(wide / "skin/coverage.csv")
    with torch.inference_mode():
        for method in methods:
            model_out = wide / "scores" / method
            receipt = json.loads((model_out / "summary.json").read_text())
            path = model_out / "skin-frames.csv"
            if path.exists():
                raise FileExistsError(path)
            rows = []
            for row in coverage:
                filename = f"{int(row['index']):05d}.png"
                with Image.open(wide / "skin/masks" / filename) as image:
                    mask = torch.from_numpy(np.array(image) > 127).cuda()
                assert int(mask.sum().item()) == int(row["skin_pixels"])
                rgb = []
                for src in (wide / "datasets/heldout/images" / filename, model_out / filename):
                    with Image.open(src) as image:
                        array = np.array(image.convert("RGB"))
                    rgb.append(torch.from_numpy(array).cuda().permute(2, 0, 1)[None].float() / 255)
                gt, pred = rgb
                mse = ((pred - gt).square()[0, :, mask]).mean().item()
                lp = metric((pred * mask) * 2 - 1, (gt * mask) * 2 - 1).item()
                yaw = float(row["yaw"])
                rows.append(dict(index=row["index"], source=row["source"], yaw=row["yaw"],
                                 group="negative" if yaw < 0 else "positive",
                                 psnr=-10 * np.log10(max(1e-12, mse)), lpips=lp))
            write(path, rows)
            print(method, receipt["weights"], len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("run", type=Path)
    prep.add_argument("model", type=Path)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("methods", nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run / "wide", args.model)
    else:
        score(args.run / "wide", args.methods)
