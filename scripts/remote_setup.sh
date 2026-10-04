#!/usr/bin/env bash
# Run ON the instance: install uv if needed, sync deps, sanity-check the GPU, run the tests.
set -euo pipefail
cd ~/ladder-witnesses
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv sync --group dev
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
uv run pytest -q
