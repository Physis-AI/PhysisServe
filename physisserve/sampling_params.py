from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    num_chunks: int | None = None
    height: int | None = None
    width: int | None = None
    fps: int | None = None
    seed: int = 0

    def __post_init__(self):
        for name in ("num_chunks", "height", "width", "fps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")

    def request(self, prompt, config):
        if not isinstance(prompt, str):
            raise TypeError("each prompt must be a string")
        height, width = self.height or config.height, self.width or config.width
        for size, patch in zip((height, width), config.patch_size[-2:]):
            if size % (config.spatial_scale * patch):
                raise ValueError("height and width must be divisible by the model's spatial patch size")
        shape = (1, config.channels, 1, height // config.spatial_scale, width // config.spatial_scale)
        return Request(prompt, shape, self.num_chunks or config.chunks, self.seed, self.fps or config.fps)


@dataclass(frozen=True)
class Request:
    prompt: str
    shape: tuple[int, ...]
    chunks: int
    seed: int
    fps: int

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if (len(self.shape) != 5 or any(type(n) is not int or n < 1 for n in self.shape)
                or self.shape[0] != 1 or self.shape[2] != 1):
            raise ValueError("shape must be (1, channels, 1, height, width)")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        SamplingParams(num_chunks=self.chunks, fps=self.fps, seed=self.seed)
