#!/usr/bin/env bash
# After native-recipe.sh, run a full-recipe ablation using the same native initialization.
set -euo pipefail
: "${RUN:?set absolute RUN directory}" "${PYTHON_BIN:?set absolute Python path}"
method=$1
export PATH="$RUN/launcher:$PATH" PYTHONPATH="$RUN/full-repo/src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "$RUN/full-repo"
if [[ "$method" == real ]]; then
    bash train_avatar.sh --subject subject --work-name full-real --no-repairs -- --seed 1701
else
    labels="$RUN/full-replacement/labels/$method"
    [[ -d "$labels" ]]
    bash train_avatar.sh --subject subject --work-name "full-$method" -- --seed 1701 --repair-dir "$labels"
fi
ln -s "$RUN/full-repo/models/subject/full-$method" "$RUN/models/subject/full-$method"
