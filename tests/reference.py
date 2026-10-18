"""Serial reference: chunk by chunk, pass by pass, block by block; each pass keeps its own per-block KV history.

This is the plainest form of Same Step, with no ticks, ranks, plan or transport. Every Same Step
topology of the engine must match it bit for bit.
"""
import torch

from physisserve.contract import DENOISING_STEPS
from physisserve.engine.chunks import Chunks
from physisserve.engine.kv_cache import Cache, history_chunks
from physisserve.engine.model_runner import generator


@torch.inference_mode()
def run(dit, sampler, condition, shape, chunks, seed=0, noise=None, device="cpu", dtype=torch.float32):
    blocks = dit.num_blocks
    dit.load(range(blocks), device, dtype)
    dit.prepare(condition, shape, chunks)
    first_frames, sink_sigma = dit.first_frames, dit.sink_sigma
    geometry = Chunks(shape, chunks, first_frames)
    pages, sink = {}, {}
    outputs = []
    for chunk in range(chunks):
        chunk_shape = geometry.shape_of(chunk)
        sample = (noise[chunk].to(device, dtype).clone() if noise is not None else
                  torch.randn(chunk_shape, generator=generator(device, seed, chunk, -1), device=device, dtype=dtype))
        for step, timestep in enumerate(sampler.timesteps):
            timestep = torch.tensor(timestep, dtype=torch.float32)  # CPU scalar, as the runner passes it
            hidden = dit.embed(sample, timestep, chunk)
            for block in range(blocks):
                store = pages.setdefault((step, block), [])
                read_chunks = history_chunks(chunk)
                cache = Cache([sink[block] if read == 0 else store[read] for read in read_chunks])
                hidden = dit.block(block, hidden, timestep, chunk, cache)
                store.append(cache.page)
            prediction = dit.unembed(hidden, timestep, chunk)
            sample = sampler.advance(prediction, sample, step, generator(device, seed, chunk, step))
        outputs.append(sample.clone())
        if chunk == 0 and chunks > 1:
            if sink_sigma:
                clean_noise = torch.randn(
                    chunk_shape, generator=generator(device, seed, 0, DENOISING_STEPS),
                    device=device, dtype=torch.float32,
                )
                sample = ((1.0 - sink_sigma) * sample.float() + sink_sigma * clean_noise).to(dtype)
            timestep = torch.tensor(dit.timestep_for_sigma(sink_sigma), dtype=torch.float32)
            hidden = dit.embed(sample, timestep, 0)
            for block in range(blocks):
                cache = Cache([])
                hidden = dit.block(block, hidden, timestep, 0, cache)
                sink[block] = cache.page
    return torch.cat(outputs, dim=2)
