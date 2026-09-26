#!/usr/bin/env bash
set -euo pipefail
cd /hai-workspace/glm52-delta
tar -xzf artifacts/torch-compile-word-two-phase-sources.tar.gz
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 PYTHONPATH=miles-torch-compile TORCHINDUCTOR_CACHE_DIR=/hai-workspace/glm52-delta/artifacts/torch-compile-word-two-phase-unit-cache timeout --kill-after=5 240 /opt/sglang/bin/python - <<'PYCHECK' > artifacts/torch-compile-word-two-phase-unit.log 2>&1
import importlib.util, sys, pytest
name = "miles.utils.delta_preparation"
spec = importlib.util.spec_from_file_location(name, "delta_preparation_word_two_phase_variant.py")
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
raise SystemExit(pytest.main(["test_delta_preparation_word_two_phase_candidate.py", "-q"]))
PYCHECK
set +e
CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=1 timeout --kill-after=10 1200 /opt/sglang/bin/python benchmark_delta_preparation.py --miles-path miles --preparation-module delta_preparation_word_two_phase_variant.py --checkpoint models/GLM-5.2_5layer-NVFP4 --synthetic-source artifacts/synthetic-balanced-disk-delta-01/delta-publication --expected-tensors 4690 --block-bytes 4096 --bucket-mib 128 --baseline-workers 32 --stage-workers 32 --kernel-threads 1 --warmups 2 --repeats 5 --output artifacts/torch-compile-word-two-phase-32x1-replay.jsonl > artifacts/torch-compile-word-two-phase-32x1-replay.log 2>&1
result=$?
echo "$result" > artifacts/torch-compile-word-two-phase-32x1-replay.exit
tail -5 artifacts/torch-compile-word-two-phase-unit.log
tail -4 artifacts/torch-compile-word-two-phase-32x1-replay.log
exit "$result"
