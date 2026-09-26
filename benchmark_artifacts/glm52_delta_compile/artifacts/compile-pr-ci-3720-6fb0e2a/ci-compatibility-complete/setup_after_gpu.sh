#!/usr/bin/env bash
# Run only after root releases the replacement node's CPU slot.
set -euo pipefail
if [[ "${1:-}" != --after-gpu-release ]]; then
  echo 'Run with --after-gpu-release only after the parent task releases the CPU slot.' >&2
  exit 2
fi
CI_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EFFORT_ROOT=$(dirname "$CI_ROOT")
mkdir -p "$CI_ROOT/setup-logs" "$CI_ROOT/bin"
exec > >(tee -a "$CI_ROOT/setup-logs/setup.log") 2>&1
ulimit -c 0
export UV_CACHE_DIR="$CI_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$CI_ROOT/managed-python"
export UV_MANAGED_PYTHON=1
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_BUILDS=2

df -h "$CI_ROOT"
CI_AVAILABLE_BYTES=$(df -PB1 "$CI_ROOT" | awk 'NR==2 {print $4}')
if (( CI_AVAILABLE_BYTES < 24000000000 )); then
  echo 'At least24GB free workspace space is required for isolated CI dependencies and compiler artifacts.' >&2
  exit 2
fi

timeout --kill-after=10 120 /opt/sglang/bin/python -m pip install --no-deps --target "$CI_ROOT/uv-bootstrap" uv==0.12.19
uv_run() {
  "$CI_ROOT/uv-bootstrap/bin/uv" "$@"
}
uv_run --version
if uv_run python install 3.11.16; then
  CI_PYTHON_VERSION=3.11.16
else
  echo 'Exact managedPython3.11.16 unavailable; recording a3.11 fallback explicitly.'
  uv_run python install 3.11
  CI_PYTHON_VERSION=3.11
fi
uv_run venv --python "$CI_PYTHON_VERSION" "$CI_ROOT/venv"
CI_PYTHON="$CI_ROOT/venv/bin/python"
printf '%s\n' "$CI_PYTHON" > "$CI_ROOT/python-path.txt"
"$CI_PYTHON" --version

# Exact shallow source stores are self-contained and preserve original commit IDs.
# The diagnostic worktrees and their commonGit stores stay inside this directory.
git -C "$CI_ROOT/miles-source.git" worktree add --detach "$CI_ROOT/miles" 5093563d01e81e6a71a1425470f32e359338f37f
git -C "$CI_ROOT/sglang-source.git" worktree add --detach "$CI_ROOT/sglang" 106ef6d1262a8c1bf61cabcb9053102ff80f71f9
mkdir "$CI_ROOT/Megatron-LM"
tar -xzf "$CI_ROOT/megatron-f148a32-source.tar.gz" -C "$CI_ROOT/Megatron-LM"
printf '%s\n' f148a32b4385b758b66a77c9c3ad1641f1295d4b > "$CI_ROOT/megatron-source-revision.txt"

# Use the same pinnedTorch2.11 PyPI CUDA13 wheel as hostedCI. Tests force all GPUs
# invisible. All packages are isolated from the activeMiles image's site-packages.
export GITHUB_WORKSPACE="$CI_ROOT/miles"
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
