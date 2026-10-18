"""Non-configurable invariants of PhysisServe inference."""
from typing import Final

LATENTS_PER_CHUNK: Final = 1
ATTENTION_SINK_CHUNKS: Final = 1
DENOISING_STEPS: Final = 4
HISTORY_CHUNKS: Final = 18


def require_one_latent_per_chunk(shape) -> None:
    """Reject a shape that does not contain exactly one temporal latent per chunk."""
    if len(shape) != 5 or shape[2] != LATENTS_PER_CHUNK:
        raise ValueError("PhysisServe requires exactly one temporal latent per chunk (shape[2] must be 1)")
