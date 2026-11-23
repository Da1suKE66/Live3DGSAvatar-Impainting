"""Prepare the unused high-yaw photos with the original training-only rig."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from prepare import encode


def prepare_wide(capture, run):
    with (run / "frames.csv").open() as stream:
        used = list(csv.DictReader(stream))
    used_ids = {int(row["source"]) for row in used}
    used_times = [float(row["seconds"]) for row in used]
    with np.load(run / "rig.npz") as archive:
        rig = dict(archive)
    with np.load(run / "datasets/subject/render_inputs.npz") as archive:
        camera = {k: archive[k] for k in ("K", "w2v", "width", "height")}

    rows, meshes, weights, photos = [], [], [], []
    for part in ("train", "validation"):
        folder = capture / "datasets" / part
        tracks = json.loads((folder / "tracks.json").read_text())
        indices = []
        for track in tracks:
            yaw, source, time = track["yaw"], track["source_index"], track["seconds"]
            if not track["tracking_ok"] or abs(yaw) < 30 or source in used_ids:
                continue
            gap = min(abs(time - seen) for seen in used_times)
            if gap < 1:
                continue
            i = track["local_id"]
            indices.append(i)
            photos.append(folder / "images" / f"{i:05d}.png")
            rows.append(dict(dataset="heldout", index=len(rows), split=part, source=source, local_id=i,
                             seconds=time, yaw=yaw, nearest_used_seconds=gap))
        with np.load(folder / "render_inputs.npz") as archive:
            if not all(np.array_equal(archive[k], value) for k, value in camera.items()):
                raise ValueError(f"Camera changed in {part}")
            if indices:
                mesh, drive = encode(archive["blend_weight"][indices], rig)
                meshes.append(mesh)
                weights.append(drive)

    if not rows or len({r["source"] for r in rows}) != len(rows):
        raise ValueError("Expected distinct, eligible source frames")
    if not all(p.is_file() for p in photos):
        raise FileNotFoundError("Some selected photos are missing")
    wide = run / "wide"
    wide.mkdir(exist_ok=False)
    data = wide / "datasets/heldout"
    (data / "images").mkdir(parents=True)
    for i, photo in enumerate(photos):
        (data / "images" / f"{i:05d}.png").symlink_to(photo)
    np.savez(data / "render_inputs.npz", mesh=np.concatenate(meshes), blend_weight=np.concatenate(weights), **camera)
    with (wide / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} unused real photos with >= 30 degrees yaw and >= 1 second separation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    prepare_wide(args.capture.resolve(), args.run.resolve())
