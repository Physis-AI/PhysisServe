from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SamplingParams:
    """``dit_only`` returns latents; ``deferred_decode`` decodes after all streams.

    ``stream_decode`` decodes each completed chunk while denoising continues.
    ``deferred_decode`` saves latents under ``latent_dir`` (default: ./latents)
    before decoding. Each generation gets a separate subdirectory. Other modes
    save latents only when ``latent_dir`` is specified.
    """

    num_chunks: int | None = None
    height: int | None = None
    width: int | None = None
    fps: int | None = None
    seed: int = 0
    mode: str = "deferred_decode"
    latent_dir: str | Path | None = None

    def __post_init__(self):
        for name in ("num_chunks", "height", "width", "fps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if self.mode not in ("dit_only", "deferred_decode", "stream_decode"):
            raise ValueError("mode must be dit_only, deferred_decode, or stream_decode")
        if self.latent_dir is not None:
            if not isinstance(self.latent_dir, (str, Path)) or not str(self.latent_dir).strip():
                raise ValueError("latent_dir must be a nonempty directory path")
            path = Path(self.latent_dir)
            if path.exists() and not path.is_dir():
                raise ValueError("latent_dir must be a directory")
            object.__setattr__(self, "latent_dir", path)

    def request(self, prompt, config):
        if not isinstance(prompt, str):
            raise TypeError("each prompt must be a string")
        height, width = self.height or config.height, self.width or config.width
        for size, patch in zip((height, width), config.patch_size[-2:]):
            if size % (config.spatial_scale * patch):
                raise ValueError("height and width must be divisible by the model's spatial patch size")
        shape = (1, config.channels, 1, height // config.spatial_scale, width // config.spatial_scale)
        return Request(prompt, shape, self.num_chunks or config.chunks, self.seed,
                       self.fps or config.fps, self.mode, self.latent_dir)


@dataclass(frozen=True)
class Request:
    prompt: str
    shape: tuple[int, ...]
    chunks: int
    seed: int
    fps: int
    mode: str = "dit_only"
    latent_dir: str | Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if (len(self.shape) != 5 or any(type(n) is not int or n < 1 for n in self.shape)
                or self.shape[0] != 1 or self.shape[2] != 1):
            raise ValueError("shape must be (1, channels, 1, height, width)")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        params = SamplingParams(num_chunks=self.chunks, fps=self.fps, seed=self.seed,
                                mode=self.mode, latent_dir=self.latent_dir)
        object.__setattr__(self, "latent_dir", params.latent_dir)
