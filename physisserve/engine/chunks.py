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

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if len(self.shape) != 5 or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("shape must be five positive integers (B, C, T, H, W)")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive integer")
        if type(self.first_frames) is not int or self.first_frames < 1:
            raise ValueError("first_frames must be a positive integer")

    @property
    def frames(self) -> int:
        """Latent frames of every chunk after the first."""
        return self.shape[2]

    @property
    def first(self) -> int:
        return self.first_frames

    def frames_of(self, chunk: int) -> int:
        self._check(chunk)
        return self.first if chunk == 0 else self.frames

    def shape_of(self, chunk: int) -> tuple[int, ...]:
        b, c, _, h, w = self.shape
        return (b, c, self.frames_of(chunk), h, w)

    def start(self, chunk: int) -> int:
        """Index of the chunk's first latent frame in the video (``start(count)`` is the total)."""
        if not 0 <= chunk <= self.count:
            raise ValueError(f"chunk {chunk} is outside 0..{self.count}")
        return 0 if chunk == 0 else self.first + (chunk - 1) * self.frames

    @property
    def total(self) -> int:
        """Latent frames of the whole request."""
        return self.start(self.count)

    def history_frames(self, pages: int) -> int:
        """Latent frames held by ``pages`` history pages. Chunk 0 is always among them when there is any history:
        it is the fixed sink, or the start of a window that has not moved yet."""
        if type(pages) is not int or pages < 0:
            raise ValueError("pages must be a nonnegative integer")
        return 0 if pages == 0 else self.first + (pages - 1) * self.frames

    def _check(self, chunk: int):
        if not 0 <= chunk < self.count:
            raise ValueError(f"chunk {chunk} is outside 0..{self.count - 1}")
