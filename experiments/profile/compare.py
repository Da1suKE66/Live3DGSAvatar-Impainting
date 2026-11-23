"""Audit the published per-frame measurements and regenerate paired comparison tables."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
SHORT = [
    "base", "real", "raw", "native-syncfix", "syncfix", "difix", "edgar",
    "elite", "eg3d", "fate-prior", "morphable",
]
FULL = ["full-base", "full-real", "full-native", "full-elite", "full-edgar", "full-blend"]
MATCHED = ["matched-real", "matched-elite", "matched-edgar", "matched-blend"]
METRICS = ("psnr", "lpips")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def audit_split(path):
    frames = rows(path)
    partitions = {k: [r for r in frames if r["split"] == k] for k in ("train", "validation", "test")}
    assert [len(p) for p in partitions.values()] == [192, 32, 87]
    assert len({r["source"] for r in frames}) == len(frames)
    for earlier, later in (("train", "validation"), ("validation", "test")):
        assert max(float(r["seconds"]) for r in partitions[earlier]) < min(float(r["seconds"]) for r in partitions[later])
    assert all(abs(float(r["yaw"])) <= 10 for r in partitions["train"] + partitions["validation"])
    return partitions["test"], hashlib.sha256(path.read_bytes()).hexdigest()


def pose_group(yaw):
    if abs(yaw) <= 10:
        return "front"
    assert abs(yaw) >= 15
    return "negative" if yaw < 0 else "positive"


def audit_method(method, expected, split_hash):
    folder = ROOT / "logs" / method
    receipt = json.loads((folder / "summary.json").read_text())
    assert receipt["frames_sha256"] == split_hash and receipt["method"] == method
    samples = rows(folder / "frames.csv")
    assert len(samples) == len(expected)
    for sample, frame in zip(samples, expected, strict=True):
        assert all(sample[k] == frame[k] for k in ("index", "source", "yaw"))
        assert sample["group"] == pose_group(float(frame["yaw"]))
        assert all(math.isfinite(float(sample[m])) for m in METRICS)

    out = {"method": method}
    for label, count in (("side", 49), ("negative", 26), ("positive", 23), ("front", 38)):
        group = [r for r in samples if r["group"] == label or (label == "side" and r["group"] != "front")]
        assert len(group) == receipt["groups"][label]["n"] == count
        for metric in METRICS:
            value = mean(float(r[metric]) for r in group)
            assert math.isclose(value, receipt["groups"][label][metric], abs_tol=1e-9)
            out[f"{label}_{metric}"] = round(value, 7)

    virtual = rows(folder / "virtual.csv")
    assert sorted(int(r["yaw"]) for r in virtual) == [-50, -35, -20, 20, 35, 50]
    for metric in ("hole", "spill"):
        value = mean(float(r[metric]) for r in virtual)
        assert math.isclose(value, receipt["virtual"][metric], abs_tol=1e-9)
        out[f"virtual_{metric}"] = round(value, 7)

    train = json.loads((folder / "native-training.json").read_text())
    out.update(trained_steps=train["iterations"], selected_step=train["best_iteration"] + 1)
    return out, samples


def main():
    expected, split_hash = audit_split(ROOT / "logs" / "split.csv")
    summaries, comparisons, data = [], [], {}
    for method in SHORT + FULL + MATCHED:
        summary, data[method] = audit_method(method, expected, split_hash)
        summaries.append(summary)
    for method, samples in data.items():
        reference = "full-native" if method in FULL + MATCHED else "real"
        pairs = [(r, b) for r, b in zip(samples, data[reference], strict=True) if r["group"] != "front"]
        comparison = {"method": method, "reference": reference, "side_frames": len(pairs)}
        for metric in METRICS:
            diffs = [float(r[metric]) - float(b[metric]) for r, b in pairs]
            comparison[f"delta_{metric}"] = round(mean(diffs), 7)
            comparison[f"wins_{metric}"] = sum(d > 0 if metric == "psnr" else d < 0 for d in diffs)
        comparisons.append(comparison)
    write(ROOT / "summary.csv", summaries)
    write(ROOT / "paired.csv", comparisons)
    print(f"{len(summaries)} frozen models, {len(expected)} matched frames each, six identical sweep cameras per model.")


if __name__ == "__main__":
    main()
