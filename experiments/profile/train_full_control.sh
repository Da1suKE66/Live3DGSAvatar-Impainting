#!/usr/bin/env bash
# Replay a native avatar recipe, optionally using the exact native initialization file.
set -euo pipefail
: "${RUN:?set absolute RUN directory}" "${PYTHON_BIN:?set absolute Python path}"
method=${1:?expected real, elite, edgar or blend}
case "$method" in real|elite|edgar|blend) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
name="full-$method"
extra=(--seed 1701)
case "${2:-}" in
    --shared-init)
        name="matched-$method"
        init="$RUN/full-repo/tmp/subject_avatar_unified.ply"
        [[ -s "$init" ]]
        sha256sum "$init"
        extra+=(--init-ply "$init")
        ;;
    "") ;;
    *) echo "Unknown option: $2" >&2; exit 2 ;;
esac
export PATH="$RUN/launcher:$PATH" PYTHONPATH="$RUN/full-repo/src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "$RUN/full-repo"
options=(--subject subject --work-name "$name")
if [[ "$method" == real ]]; then
    options+=(--no-repairs)
else
    labels="$RUN/full-replacement/labels/$method"
    [[ -d "$labels" ]]
    extra+=(--repair-dir "$labels")
fi
bash train_avatar.sh "${options[@]}" -- "${extra[@]}"
ln -s "$RUN/full-repo/models/subject/$name" "$RUN/models/subject/$name"
