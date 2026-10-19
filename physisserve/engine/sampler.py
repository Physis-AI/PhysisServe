# Required Notice: Copyright 2026 Physis AI, Inc. and/or its affiliates
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://polyformproject.org/licenses/noncommercial/1.0.0
#
# This software is provided "AS IS", without warranties or conditions.
# See the License for permissions, limitations, and liability terms.

"""The fixed four-pass noisy sampler used by every PhysisServe checkpoint.

At noise level ``s`` the model predicts ``v = eps - x0``.  Each pass recovers
``x0 = x - s v`` and, unless it is the final pass, draws fresh noise for the
next level ``s'``: ``x' = (1 - s') x0 + s' eps``.  Checkpoints differ only in
their sigma/timestep ladder.  Arithmetic is float32 and each result is rounded
back to the latent dtype.
"""
import torch

DENOISING_STEPS = 4


class Sampler:
    def __init__(self, sigmas, timesteps):
        """Model-owned noise levels (four passes, then 0) and native per-pass timesteps."""
        self.sigmas, self.timesteps = sigmas, timesteps

    def advance(self, prediction: torch.Tensor, sample: torch.Tensor, step: int,
                generator: torch.Generator) -> torch.Tensor:
        """Latent after pass ``step`` given its prediction; the last pass returns the clean latent."""
        level, following = self.sigmas[step], self.sigmas[step + 1]
        clean = (sample.float() - level * prediction.float()).to(sample.dtype)
        if following == 0.0:
            return clean
        noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=torch.float32)
        return ((1.0 - following) * clean.float() + following * noise).to(sample.dtype)
