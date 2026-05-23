import os

import miles.utils.external_utils.command_utils as U
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h100",
    labels=["megatron"],
    disabled="Requires Blackwell/B200 CI runner for NVFP4.",
)


COMMAND = (
    "MILES_FP4_DIRECT_WEIGHT_UPDATE=1 "
    "python scripts/run_qwen3_30b_a3b.py "
    "--mode debug_minimal "
    "--no-enable-eval "
    "--hardware B200 "
    "--num-gpus-per-node 8 "
    "--actor-num-gpus-per-node 4 "
    "--rollout-num-gpus 4 "
    "--rollout-nvfp4 "
    "--train-nvfp4 "
    "--megatron-path /root/TransformerEngine:/root/Megatron-LM "
    '--extra-args "'
    "--use-rollout-routing-replay "
    "--use-miles-router "
    "--first-last-layers-bf16 "
    "--num-layers-at-start-in-bf16 0 "
    "--num-layers-at-end-in-bf16 8 "
    "--sglang-disable-shared-experts-fusion "
    "--num-rollout 3"
    '"'
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    U.exec_command(COMMAND)
