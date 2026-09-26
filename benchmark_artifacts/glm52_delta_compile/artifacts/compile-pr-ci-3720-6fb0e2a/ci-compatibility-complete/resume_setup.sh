#!/usr/bin/env bash
set -euo pipefail
CI_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export UV_CACHE_DIR="$CI_ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$CI_ROOT/managed-python" UV_MANAGED_PYTHON=1 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=2
ulimit -c 0
exec > >(tee "$CI_ROOT/setup-logs/setup-resume.log") 2>&1
"$CI_ROOT/uv-bootstrap/bin/uv" --version
uv_run() { "$CI_ROOT/uv-bootstrap/bin/uv" "$@"; }
CI_PYTHON="$CI_ROOT/venv/bin/python"
# Exact shallow source stores are self-contained and preserve original commit IDs.
# The diagnostic worktrees and their commonGit stores stay inside this directory.
git -C "$CI_ROOT/miles-source.git" worktree add --detach "$CI_ROOT/miles" 5093563d01e81e6a71a1425470f32e359338f37f
git -C "$CI_ROOT/sglang-source.git" worktree add --detach "$CI_ROOT/sglang" 106ef6d1262a8c1bf61cabcb9053102ff80f71f9
mkdir "$CI_ROOT/Megatron-LM"
tar -xzf "$CI_ROOT/megatron-f148a32-source.tar.gz" -C "$CI_ROOT/Megatron-LM"
printf '%s\n' f148a32b4385b758b66a77c9c3ad1641f1295d4b > "$CI_ROOT/megatron-source-revision.txt"

# Use the same pinnedTorch2.11 PyPI CUDA13 wheel as hostedCI. Tests force all GPUs
# invisible. All packages are isolated from the activeMiles image's site-packages.
export SGLANG_SOURCE_ROOT="$CI_ROOT/sglang/python"
export MEGATRON_SOURCE_ROOT="$CI_ROOT/Megatron-LM"
export PYTHONPATH="$CI_ROOT/miles:$SGLANG_SOURCE_ROOT:$MEGATRON_SOURCE_ROOT"
export CUDA_VISIBLE_DEVICES=''
cd "$CI_ROOT/miles"
timeout --kill-after=10 900 "$CI_ROOT/uv-bootstrap/bin/uv" pip install --python "$CI_PYTHON" -r requirements.txt -r tests/ci/requirements-ci-cpu.txt -c "$CI_ROOT/hosted-installed-constraints.txt"
uv_run pip freeze --python "$CI_PYTHON" > "$CI_ROOT/setup-logs/installed-packages.txt"
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CI_PYTHON" tests/ci/verify_source_resolution.py

# Hosted suite usesHelm4.2.3; retain the binary in this task's bin directory.
timeout --kill-after=10 120 curl -fL --retry 2 https://get.helm.sh/helm-v4.2.3-linux-amd64.tar.gz -o "$CI_ROOT/helm-v4.2.3-linux-amd64.tar.gz"
tar -xzf "$CI_ROOT/helm-v4.2.3-linux-amd64.tar.gz" -C "$CI_ROOT" linux-amd64/helm
cp "$CI_ROOT/linux-amd64/helm" "$CI_ROOT/bin/helm"
"$CI_ROOT/bin/helm" version

env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CI_PYTHON" - <<'PY' > "$CI_ROOT/setup-logs/runtime.json"
import json, os, platform, sys
import numpy, torch
print(json.dumps({'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                  'torch_git': torch.version.git_version, 'numpy': numpy.__version__,
                  'cuda_available': torch.cuda.is_available(), 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                  'intra_threads': torch.get_num_threads(), 'inter_threads': torch.get_num_interop_threads(),
                  'affinity': sorted(os.sched_getaffinity(0)), 'omp_num_threads': os.environ.get('OMP_NUM_THREADS')}, indent=2))
PY
c++ --version > "$CI_ROOT/setup-logs/compiler.txt"
lscpu > "$CI_ROOT/setup-logs/lscpu.txt"
cat "$CI_ROOT/setup-logs/runtime.json"
git status --porcelain
echo 'Setup complete. Run: venv/bin/python run_reproduction.py from the ci-torch211-repro directory.'
