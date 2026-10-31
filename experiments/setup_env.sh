#!/usr/bin/env bash
# Create the two pinned forecasting environments used by the experiments (requires uv and CUDA 12.8).
#   .venv        Chronos-2 and TimesFM 2.5
#   .venv-tirex  TiRex
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH=(--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0+cu128")

uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install "${TORCH[@]}"
VIRTUAL_ENV=.venv uv pip install -r experiments/requirements.txt
if [ ! -d third_party/timesfm ]; then
  git clone --quiet --depth 1 --branch v2.0.2 https://github.com/google-research/timesfm.git third_party/timesfm
fi
VIRTUAL_ENV=.venv uv pip install --no-deps -e third_party/timesfm
VIRTUAL_ENV=.venv uv pip install --no-deps -e .

uv venv .venv-tirex --python 3.12
VIRTUAL_ENV=.venv-tirex uv pip install "${TORCH[@]}"
VIRTUAL_ENV=.venv-tirex uv pip install -r experiments/requirements-tirex.txt
VIRTUAL_ENV=.venv-tirex uv pip install --no-deps -e .

.venv/bin/python -c "import torch, chronos, timesfm; print('forecasting env ok', torch.__version__, torch.cuda.is_available())"
.venv-tirex/bin/python -c "import torch, tirex; print('tirex env ok', torch.__version__, torch.cuda.is_available())"
