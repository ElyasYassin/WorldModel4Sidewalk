#!/usr/bin/env bash
# One-time setup of a training environment on a Linux GPU server (no Isaac Sim: training/eval/feature
# extraction only). Creates a venv in the data drive and installs the repo in editable mode.
#   bash scripts/server/setup_env.sh [/home/storage/$USER/go2wm]
set -euo pipefail
ROOT="${1:-/home/storage/$USER/go2wm}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT"/{data,outputs,logs,venv}
python3 -m venv "$ROOT/venv"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
pip install --upgrade pip wheel
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install -e "$REPO[dev,vision]" imageio-ffmpeg
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
cat <<EOF
Done. In every session:
  source $ROOT/venv/bin/activate
  export GO2WM_ROOT=$ROOT
Then: cd $REPO && python -m pytest -q
EOF
