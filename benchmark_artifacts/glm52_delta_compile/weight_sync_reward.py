"""Explicit opt-in synthetic reward for changed-weight synchronization validation.

The native E2E never calls this module. Its eight consecutive sample indices
per prompt receive four zeros and four ones. This is not a math-task reward.
"""


def _one_reward(sample):
    if sample.index is None or isinstance(sample.index, bool):
        raise ValueError("Balanced synthetic reward requires an integer sample.index")
    index = int(sample.index)
    if index != sample.index or index < 0:
        raise ValueError(
            f"Invalid sample.index for balanced synthetic reward: {sample.index!r}"
        )
    return float(index % 2)


async def balanced_index_reward(args, sample, **kwargs):
    """Support both per-sample and batched async RM dispatch without mutating samples."""
    if args.n_samples_per_prompt <= 0 or args.n_samples_per_prompt % 2:
        raise ValueError(
            "Balanced synthetic reward requires a positive even n_samples_per_prompt"
        )
    if isinstance(sample, list):
        return [_one_reward(item) for item in sample]
    return _one_reward(sample)
