"""Compare the observed extreme poses and virtual profiles using a common crop per row."""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

METHODS = [
    ("full-native", "原生 SyncFix"), ("matched-elite", "ELITE HuFix"),
    ("matched-edgar", "EDGAR"), ("matched-blend", "各半混合"),
]


def make(run, out, font):
    with (run / "frames.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    poses = [select(rows, key=lambda r: float(r["yaw"])) for select in (min, max)]
    cell, pad, sidebar = 240, 16, 164
    image = Image.new("RGB", (sidebar + 5 * (cell + pad) + pad, 77 + 4 * (cell + pad) + 20), "#f5f5f2")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(str(font), 23)
    small = ImageFont.truetype(str(font), 19)
    for col, label in enumerate(["实拍（脸部掩模）"] + [label for _, label in METHODS]):
        draw.text((sidebar + col * (cell + pad) + cell / 2, 39), label, font=title, fill="#202827", anchor="mm")

    base_ref = next((run / "teachers/shared/refs").glob("base_*.png"))
    base_index = base_ref.stem.split("_")[-1]
    for i in range(4):
        if i < len(poses):
            row = poses[i]
            name = f"{int(row['index']):05d}.png"
            rgba = Image.open(run / "datasets/heldout/images" / name).convert("RGBA")
            mask = rgba.getchannel("A")
            photo = Image.new("RGB", rgba.size)
            photo.paste(rgba, mask=mask)
            angle, label = float(row["yaw"]), "实拍测试"
        else:
            angle, label = (-50, 50)[i - len(poses)], "虚拟视角"
            name = f"yaw{angle:+d}_pitch+0_roll+0_frame{base_index}.png"
            mask = Image.open(run / "teachers/shared/flame_surface" / name).convert("L")
            photo = None
        sources = [photo] + [Image.open(run / "scores" / key / name).convert("RGB") for key, _ in METHODS]

        x0, y0, x1, y1 = mask.getbbox()
        side = int(max(x1 - x0, y1 - y0) * 1.14)
        x, y = (x0 + x1 - side) // 2, (y0 + y1 - side) // 2
        top = 77 + i * (cell + pad)
        draw.text((20, top + cell / 2 - 15), label, font=small, fill="#34423f")
        draw.text((20, top + cell / 2 + 14), f"yaw {angle:+.1f}°", font=small, fill="#34423f")
        for col, source in enumerate(sources):
            left = sidebar + col * (cell + pad)
            if source is None:
                draw.text((left + cell / 2, top + cell / 2), "无同步实拍", font=small, anchor="mm", fill="#727878")
            else:
                tile = source.crop((x, y, x + side, y + side)).resize((cell, cell), Image.Resampling.LANCZOS)
                image.paste(tile, (left, top))
    image.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    make(args.run, args.out, args.font)
