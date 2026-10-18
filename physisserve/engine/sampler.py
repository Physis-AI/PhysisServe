"""The fixed four-pass noisy sampler used by every PhysisServe checkpoint.

At noise level ``s`` the model predicts ``v = eps - x0``.  Each pass recovers
``x0 = x - s v`` and, unless it is the final pass, draws fresh noise for the
next level ``s'``: ``x' = (1 - s') x0 + s' eps``.  Checkpoints differ only in
their sigma/timestep ladder.  Arithmetic is float32 and each result is rounded
back to the latent dtype.
"""
from dataclasses import dataclass

import torch

DENOISING_STEPS = 4


@dataclass(frozen=True, init=False)
class Sampler:
    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...]

    def __init__(self, sigmas, timesteps=None):
        """sigmas: the noise level of every pass, then 0; timesteps: what the model is told at each pass
        (0-1000 scale), 1000 * sigma unless the checkpoint was trained otherwise."""
        sigmas = tuple(float(s) for s in sigmas)
        if len(sigmas) != DENOISING_STEPS + 1 or sigmas[-1] != 0.0:
            raise ValueError(f"sigmas must hold {DENOISING_STEPS} pass noise levels followed by 0")
        timesteps = tuple(1000.0 * s for s in sigmas[:-1]) if timesteps is None else tuple(float(t) for t in timesteps)
        if len(timesteps) != DENOISING_STEPS:
            raise ValueError(f"timesteps must hold one value per pass ({DENOISING_STEPS})")
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "timesteps", timesteps)

    @classmethod
    def grid(cls, shift: float = 1.0) -> "Sampler":
        """The levels of diffusers' FlowMatchEulerDiscreteScheduler(shift=...): four points spaced evenly
        over [1, shift(1/1000)], then shifted once more as a whole; they agree to float32 rounding (within
        two ulps, exactly for most shifts)."""
        if not shift > 0:
            raise ValueError("shift must be positive")

        def warp(sigma):
            return shift * sigma / (1 + (shift - 1) * sigma)

        shifted = warp(torch.linspace(1.0, warp(1.0 / 1000), DENOISING_STEPS, dtype=torch.float64))
        return cls([*shifted.float().tolist(), 0.0])

    def advance(self, prediction: torch.Tensor, sample: torch.Tensor, step: int,
                generator: torch.Generator) -> torch.Tensor:
        """Latent after pass ``step`` given its prediction; the last pass returns the clean latent."""
        level, following = self.sigmas[step], self.sigmas[step + 1]
        clean = (sample.float() - level * prediction.float()).to(sample.dtype)
        if following == 0.0:
            return clean
        noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=torch.float32)
        return ((1.0 - following) * clean.float() + following * noise).to(sample.dtype)
