#!/usr/bin/env bash
set -e

# Project root is on the path — no package install needed
export PYTHONPATH="/opt/render/project/src:${PYTHONPATH}"

mkdir -p ./checkpoints/onnx ./checkpoints/drift_former

python - <<'PYEOF'
import os
repo_id = os.environ.get("HF_REPO_ID", "")
if not repo_id:
    print("HF_REPO_ID not set — running in demo mode.")
else:
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=repo_id, filename="drift_former_int8.onnx", local_dir="./checkpoints/onnx")
    hf_hub_download(repo_id=repo_id, filename="norm_stats.npz", local_dir="./checkpoints/drift_former")
    print("Model downloaded.")
PYEOF

exec python -m uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
