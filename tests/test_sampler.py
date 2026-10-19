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

import pytest
import torch

from physisserve.engine.sampler import Sampler


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_each_pass_renoises_the_clean_estimate_with_fresh_noise(dtype):
    sampler = Sampler((1.0, 0.75, 0.5, 0.25, 0.0), (1000.0, 750.0, 500.0, 250.0))
    x, v = (torch.randn(1, 4, 3, 2, 2).to(dtype) for _ in range(2))
    generator = torch.Generator().manual_seed(1)
    clean = (x.float() - v.float()).to(dtype)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float32)
    torch.testing.assert_close(sampler.advance(v, x, 0, generator),
                               (0.25 * clean.float() + 0.75 * noise).to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(sampler.advance(v, x, 3, generator),
                               (x.float() - 0.25 * v.float()).to(dtype), rtol=0, atol=0)
