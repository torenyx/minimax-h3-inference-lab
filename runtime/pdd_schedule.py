# Adapted from VideoX-Fun videox_fun/utils/pdd_utils.py (Apache-2.0).
from typing import Sequence
import torch

def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    r"""The exponential sigma shift of a rectified-flow schedule, `sigma' = s*sigma / (1 + (s-1)*sigma)`."""
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    r"""
    The PDD time discretization `0 = t_0 < ... < t_N = 1` of a rectified-flow schedule with an exponential sigma shift.

    The paper's shift reparameterization (eq. 16), `t_n = shift_s(n/N)` with `shift_s(t) = (t/s) / (1 + (1/s - 1) t)`,
    is algebraically the same grid as `t = 1 - sigma'` over a uniform sigma grid — the convention of MiniMax-H3's
    scheduler, where `t = 1` is clean. A consequence worth relying on when the model's scheduler is a plain Euler
    rectified-flow one: the block boundaries of this grid, taken every `L` indices, are exactly the grid
    `set_timesteps(N / L + 1)` builds, so PDD generation reuses such a released scheduler unchanged.

    Args:
        shift (`float`): The exponential shift of the schedule (`12.0` video / `3.0` audio for MiniMax-H3; `1.0` is
            a uniform grid).
        num_steps (`int`): The grid size `N`.

    Returns:
        `torch.Tensor` of shape `(num_steps + 1,)`, float64: the grid, ascending from `0` to `1`.
    """
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(shift, sigma)


def pdd_sampling_plan(step_sizes: torch.Tensor, start: int, block_size: int) -> torch.Tensor:
    r"""
    The single direction a PDD generation step needs: the *mean* velocity of the whole block.

    Normalizing the fused displacement by the block span turns it into the block's average velocity, which is what an
    ordinary Euler step over the block boundaries consumes — so a plain rectified-flow scheduler drives PDD
    generation unchanged.

    Args:
        step_sizes (`torch.Tensor` of shape `(N,)`): The grid step sizes `h_l = t_{l+1} - t_l`.
        start (`int`): The block start `n`.
        block_size (`int`): The block size `L`.

    Returns:
        `torch.Tensor` of shape `(1, N)`: the plan.
    """
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype, device=step_sizes.device)
    span = step_sizes[start : start + block_size].sum()
    plan[0, start : start + block_size] = step_sizes[start : start + block_size] / span
    return plan

