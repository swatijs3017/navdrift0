#!/usr/bin/env bash
set -e

# Make navdrift0 package importable from the repo root
export PYTHONPATH="/opt/render/project/src:${PYTHONPATH}"

MODEL_DIR="./checkpoints/onnx"
NORM_DIR="./checkpoints/drift_former"
mkdir -p "$MODEL_DIR" "$NORM_DIR"

if [ ! -f "$MODEL_DIR/drift_former_int8.onnx" ]; then
    python - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
repo_id = os.environ.get("HF_REPO_ID", "")
if not repo_id:
    print("HF_REPO_ID not set — running in demo mode.")
else:
    token = os.environ.get("HF_TOKEN", None)
    hf_hub_download(repo_id=repo_id, filename="drift_former_int8.onnx",
                    local_dir="./checkpoints/onnx", token=token)
    hf_hub_download(repo_id=repo_id, filename="norm_stats.npz",
                    local_dir="./checkpoints/drift_former", token=token)
    print("Model downloaded.")
PYEOF
fi

exec uvicorn navdrift0.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
