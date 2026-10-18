from dataclasses import FrozenInstanceError

import pytest
import torch

from physisserve.engine.sampler import Sampler


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_grid_matches_diffusers_flow_match_euler(shift):
    diffusers = pytest.importorskip("diffusers")
    reference = diffusers.FlowMatchEulerDiscreteScheduler(shift=shift)
    reference.set_timesteps(4)
    sampler = Sampler.grid(shift)
    torch.testing.assert_close(torch.tensor(sampler.timesteps), reference.timesteps.float(), rtol=1e-6, atol=1e-6)


def test_each_pass_renoises_the_clean_estimate_with_fresh_noise():
    sampler = Sampler((1.0, 0.75, 0.5, 0.25, 0.0))
    x, v = torch.randn(1, 4, 3, 2, 2), torch.randn(1, 4, 3, 2, 2)
    generator = torch.Generator().manual_seed(1)
    clean = x - 1.0 * v
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float32)
    torch.testing.assert_close(sampler.advance(v, x, 0, generator), 0.25 * clean + 0.75 * noise)
    torch.testing.assert_close(sampler.advance(v, x, 3, generator), x - 0.25 * v)  # final pass returns x0


def test_sampler_rejects_any_non_four_step_ladder():
    for bad in [((1.0,),), ((1.0, 0.5, 0.0),),
                ((1.0, 0.75, 0.5, 0.25, 0.1, 0.0),),
                ((1.0, 0.75, 0.5, 0.25, 0.0), (1000.0, 750.0))]:
        with pytest.raises(ValueError):
            Sampler(*bad)
    for shift in (0.0, -1.0):
        with pytest.raises(ValueError):
            Sampler.grid(shift)


def test_sampler_owns_immutable_ladders():
    sigmas, timesteps = [1.0, 0.75, 0.5, 0.25, 0.0], [1000.0, 750.0, 500.0, 250.0]
    sampler = Sampler(sigmas, timesteps)
    sigmas[0] = timesteps[0] = -1.0
    assert sampler.sigmas == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert sampler.timesteps == (1000.0, 750.0, 500.0, 250.0)
    with pytest.raises(FrozenInstanceError):
        sampler.sigmas = ()
