"""Chunk geometry of one request: every chunk has the request's shape, except that the first may span a different
number of latent frames (``first_frames``).

Some model adapters keep the first chunk at 3 latent frames while every later chunk contains 1.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunks:
    shape: tuple[int, ...]           # (B, C, T, H, W) of every chunk after the first
    count: int
    first_frames: int                # latent frames of chunk 0

    def frames_of(self, chunk: int) -> int:
        return self.first_frames if chunk == 0 else self.shape[2]

    def shape_of(self, chunk: int) -> tuple[int, ...]:
        b, c, _, h, w = self.shape
        return (b, c, self.frames_of(chunk), h, w)

    def start(self, chunk: int) -> int:
        """Index of the chunk's first latent frame in the video (``start(count)`` is the total)."""
        return 0 if chunk == 0 else self.first_frames + (chunk - 1) * self.shape[2]

    @property
    def total(self) -> int:
        """Latent frames of the whole request."""
        return self.start(self.count)

    def history_frames(self, pages: int) -> int:
        """Latent frames held by ``pages`` history pages. Chunk 0 is always among them when there is any history:
        it is the fixed sink, or the start of a window that has not moved yet."""
        return self.start(pages)
