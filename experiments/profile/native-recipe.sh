#!/usr/bin/env bash
# Run upstream's standard two-stage recipe in an isolated clone of this branch.
set -euo pipefail
: "${RUN:?set RUN to the absolute prepared experiment directory}"
: "${PYTHON_BIN:?set PYTHON_BIN to the absolute GPU Python executable}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
full="$RUN/full-repo"
git clone --quiet --local "$repo" "$full"
cd "$full"
mkdir -p datasets/subject "$RUN/launcher"
for item in images render_inputs.npz surface_masks.npz; do
    ln -s "$RUN/datasets/subject/$item" "datasets/subject/$item"
done
"$PYTHON_BIN" - "$RUN/local.yaml" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
for k in ('data_root','models_dir','output_dir'):
    del cfg['paths'][k]
Path('configs/local.yaml').write_text(yaml.safe_dump(cfg))
PY
# The upstream shell entry points use conda solely to select a Python interpreter.
cat > "$RUN/launcher/conda" <<'WRAPPER'
#!/usr/bin/env bash
set -eu
[[ "$1" == run && "$2" == -n && "$4" == --no-capture-output && "$5" == python ]]
shift 5
exec "$PYTHON_BIN" "$@"
WRAPPER
chmod +x "$RUN/launcher/conda"
export PATH="$RUN/launcher:$PATH" PYTHONPATH="$full/src:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
bash train_base.sh --subject subject -- --seed 1701 > "$RUN/full-base.log" 2>&1
bash train_avatar.sh --subject subject -- --seed 1701 > "$RUN/full-native.log" 2>&1
