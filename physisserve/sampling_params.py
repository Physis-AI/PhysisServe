from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SamplingParams:
    """``dit_only`` returns latents; ``offline_decode`` decodes after all streams.

    ``online_decode`` decodes each completed chunk while denoising continues.
    ``offline_decode`` saves latents under ``latent_dir`` (default: ./latents)
    before decoding. Each generation gets a separate subdirectory. Other modes
    save latents only when ``latent_dir`` is specified.
    """

    num_chunks: int | None = None
    height: int | None = None
    width: int | None = None
    fps: int | None = None
    seed: int = 0
    mode: str = "offline_decode"
    latent_dir: str | Path | None = None

    def __post_init__(self):
        for name in ("num_chunks", "height", "width", "fps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if self.mode not in ("dit_only", "offline_decode", "online_decode"):
            raise ValueError("mode must be dit_only, offline_decode, or online_decode")
        if self.latent_dir is not None:
            if not isinstance(self.latent_dir, (str, Path)) or not str(self.latent_dir).strip():
                raise ValueError("latent_dir must be a nonempty directory path")
            path = Path(self.latent_dir).expanduser()
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
    """Internal request built from validated sampling parameters and model defaults."""

    prompt: str
    shape: tuple[int, ...]
    chunks: int
    seed: int
    fps: int
    mode: str = "dit_only"
    latent_dir: str | Path | None = None
