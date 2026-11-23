"""Recalculate metrics for the registered, real wide-profile photo supplement."""
import hashlib
import json
import math
from statistics import mean

from compare import METRICS, ROOT, paired_scores, repeated_pairs, rows, write


def angle_band(yaw):
    return "30_to_45" if abs(yaw) < 45 else "45_and_up"


def select_group(samples, group):
    if group == "side":
        return samples
    if group in ("negative", "positive"):
        return [r for r in samples if r["group"] == group]
    return [r for r in samples if angle_band(float(r["yaw"])) == group]


def main():
    logs = ROOT / "logs"
    wide = logs / "wide"
    original = rows(logs / "split.csv")
    used_sources = {r["source"] for r in original}
    used_times = [float(r["seconds"]) for r in original]
    split = wide / "split.csv"
    expected = rows(split)
    assert len(expected) == len({r["source"] for r in expected}) == 55
    assert used_sources.isdisjoint(r["source"] for r in expected)
    for frame in expected:
        gap = min(abs(float(frame["seconds"]) - t) for t in used_times)
        assert gap >= 1.0 and math.isclose(gap, float(frame["nearest_used_seconds"]))
        assert frame["split"] in ("train", "validation") and abs(float(frame["yaw"])) >= 30
    fingerprint = hashlib.sha256(split.read_bytes()).hexdigest()
    coverage_path = wide / "skin-coverage.csv"
    segmenter = json.loads((wide / "segmenter.json").read_text())
    assert segmenter["categories"][segmenter["target"]] == "face-skin"
    assert segmenter["frames_sha256"] == fingerprint
    assert segmenter["coverage_sha256"] == hashlib.sha256(coverage_path.read_bytes()).hexdigest()
    for a, b in zip(rows(coverage_path), expected, strict=True):
        assert all(a[k] == b[k] for k in ("index", "source", "yaw"))
        assert math.isclose(int(a["overlap_pixels"]) / int(a["skin_pixels"]), float(a["skin_covered_by_oval"]))
    metrics, comparisons = [], []
    group_counts = {"side": 55, "negative": 22, "positive": 33, "30_to_45": 25, "45_and_up": 30}
    pairs = [(1701, "full-native", "matched-difix")]
    pairs.extend(repeated_pairs())
    models = [row["method"] for row in rows(ROOT / "summary.csv")]
    for mask, filename in (("original_oval", "frames.csv"), ("face_skin", "skin-frames.csv")):
        data = {}
        for method in models:
            receipt = json.loads((wide / method / "summary.json").read_text())
            old = json.loads((logs / method / "summary.json").read_text())
            assert receipt["method"] == method and receipt["weights"] == old["weights"]
            assert receipt["frames_sha256"] == fingerprint
            samples = rows(wide / method / filename)
            for r, e in zip(samples, expected, strict=True):
                assert all(r[k] == e[k] for k in ("source", "index", "yaw"))
                assert r["group"] == ("negative" if float(e["yaw"]) < 0 else "positive")
                assert all(math.isfinite(float(r[k])) for k in METRICS)
            data[method] = samples
            for group, count in group_counts.items():
                selected = select_group(samples, group)
                assert len(selected) == count
                row = {"method": method, "mask": mask, "group": group, "n": count}
                for k in METRICS:
                    value = mean(float(r[k]) for r in selected)
                    if mask == "original_oval" and group in receipt["groups"]:
                        assert receipt["groups"][group]["n"] == count
                        assert math.isclose(value, receipt["groups"][group][k], abs_tol=1e-9)
                    row[k] = round(value, 7)
                metrics.append(row)
        for seed, native, candidate in pairs:
            for group in group_counts:
                a, b = (select_group(data[k], group) for k in (native, candidate))
                row = {"seed": seed, "reference": native, "candidate": candidate, "mask": mask, "group": group}
                row.update(paired_scores(a, b, "side"))
                comparisons.append(row)
    write(ROOT / "wide-summary.csv", metrics)
    write(ROOT / "wide-pairs.csv", comparisons)
    print(f"{len(models)} frozen checkpoints scored against {len(expected)} previously unused real photos.")


if __name__ == "__main__":
    main()
