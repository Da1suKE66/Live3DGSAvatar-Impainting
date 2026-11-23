#!/usr/bin/env bash
# Run a single condition and keep the original process logs.
set -euo pipefail
: "${RUN:?absolute prepared run}" "${PYTHON_BIN:?absolute GPU Python}" "${UPSTREAM:?author checkouts}" "${ASSETS:?local asset directory}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
scripts="$repo/experiments/profile"
model_dir="$ASSETS/models/profile"
method=$1
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ASSETS/torch"
export PYTHONPATH="$repo/src:${PYTHONPATH:-}"
train() {
    "$PYTHON_BIN" "$scripts/train.py" "$RUN" "$method" > "$RUN/$method.log" 2>&1
}
labels() {
    "$PYTHON_BIN" "$scripts/labels.py" "$RUN" "$method" "$1" > "$RUN/$method-labels.log"
}
case "$method" in
 base)
    train
    "$PYTHON_BIN" "$repo/scripts/cache_surface_masks.py" --data-dir "$RUN/datasets/subject" > "$RUN/masks.log" 2>&1
    # Views are generated here so even the no-teacher controls use the same sweep.
    PYTHONPATH="${SYNCFIX_SITE:-}:$UPSTREAM/SyncFix/src:$PYTHONPATH" "$PYTHON_BIN" -u "$repo/scripts/make_repairs.py" \
      --subject subject --work-name base --yaws=-50,-35,-20,20,35,50 --out "$RUN/teachers/shared" \
      --out-dir "$RUN/labels/native-syncfix" > "$RUN/syncfix-teacher.log" 2>&1
    ;;
 real|native-syncfix) train ;;
 raw|syncfix)
    folder=$method
    if [[ $method == raw ]]; then folder=views; fi
    labels "$RUN/teachers/shared/$folder"
    train ;;
 difix|edgar|elite|eg3d|fate-prior|morphable)
    case "$method" in
      difix) args=("$scripts/difix.py" "$RUN" "$UPSTREAM/Difix3D" "$model_dir/difix_ref"); site="${LEGACY_SITE:-}" ;;
      edgar) args=("$scripts/edgar.py" "$RUN" "$UPSTREAM/edgar-code" "$model_dir"); site="${LEGACY_SITE:-}" ;;
      elite) args=("$scripts/elite.py" "$RUN" "$UPSTREAM/ELITE" "$model_dir/2d_prior.pth"); site="${LEGACY_SITE:-}" ;;
      eg3d) args=("$scripts/eg3d.py" "$RUN" "$UPSTREAM/eg3d" "$model_dir/eg3d.pkl" "$ASSETS"); site="" ;;
      fate-prior) args=("$scripts/fate_prior.py" "$RUN" "$model_dir/GFPGANv1.3.pth" "$ASSETS/models/face_landmarker.task"); site="${FATE_SITE:-}" ;;
      morphable) args=("$scripts/morphable.py" "$RUN" "$UPSTREAM/morphablediffusion" "$model_dir/morphable-second.ckpt" "$model_dir/morphable.ckpt"); site="${MORPHABLE_SITE:-}:${LEGACY_SITE:-}" ;;
    esac
    env PYTHONPATH="$site:$PYTHONPATH" "$PYTHON_BIN" -u "${args[@]}" > "$RUN/$method-teacher.log" 2>&1
    labels "$RUN/teachers/$method"
    train ;;
 *) echo "unknown method $method" >&2; exit 2 ;;
esac
"$PYTHON_BIN" "$scripts/score.py" "$RUN" "$method" > "$RUN/$method-score.log" 2>&1
