import numpy as np


def usage_coeff_schedule(
    sae_step: int,
    base: float = 1.5,
    warmup_steps: int = 6000,
    period_steps: int = 60000,
    floor: float = 0.05,
) -> float:
    """
    Calculate the usage (KL) coefficient based on the current SAE training step.

    Implements a linear warmup phase followed by a post-warmup cosine schedule.

    Args:
        sae_step: Current step in SAE training
        base: Maximum coefficient value during warmup (default: 1.5)
        warmup_steps: Number of steps for linear warmup (default: 6000)
        period_steps: Period of cosine oscillation after warmup (default: 60000)
        floor: Minimum coefficient value (default: 0.05)

    Returns:
        float: Usage coefficient for the current step
    """
    if sae_step < warmup_steps:
        return floor + (base - floor) * sae_step / warmup_steps

    # cosine from `base` down to `floor` and back every `period_steps`
    phase = (sae_step - warmup_steps) % period_steps
    cos_term = 0.5 * (1 + np.cos(2 * np.pi * phase / period_steps))
    return floor + (base - floor) * cos_term


def l1_coeff_schedule(
    sae_step: int,
    base: float = 1e-3,
    warmup_steps: int = 6000,
    start: float = 0.0,
) -> float:
    """
    Calculate the L1 coefficient based on the current SAE training step.

    Implements a linear warmup phase from start to base value.

    Args:
        sae_step: Current step in SAE training
        base: Target L1 coefficient value after warmup (default: 1e-3)
        warmup_steps: Number of steps for linear warmup (default: 6000)
        start: Initial L1 coefficient value (default: 0.0)

    Returns:
        float: L1 coefficient for the current step
    """
    if sae_step < warmup_steps:
        return start + (base - start) * sae_step / warmup_steps

    return base
