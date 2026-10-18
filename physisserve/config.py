import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path


def resolve_model(model):
    root = Path(model).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(str(model)))
    if not (root / "model_index.json").is_file():
        raise ValueError("model must be a complete Diffusers directory or Hugging Face repository")
    return str(root.resolve())


@dataclass
class Config:
    model: str
    ranks: int = 1
    strategy: str = "virtual_rank"
    max_batch: int = 4
    device: str = "cuda"
    encoder_device: str = "cpu"
    rope_backend: str = "native"
    batch_costs: tuple[tuple[float, ...], ...] | None = None
    admission_window: int | None = None
    backend: str = field(init=False)
    dtype: str = field(init=False, default="bfloat16")
    compute_dtype: str | None = field(init=False, default=None)
    cache_text_kv: bool = field(init=False, default=False)

    def __post_init__(self):
        if type(self.ranks) is not int or self.ranks < 1:
            raise ValueError("pipeline_parallel_size must be a positive integer")
        if self.strategy not in ("stage_equal", "virtual_rank"):
            raise ValueError("strategy must be stage_equal or virtual_rank")
        if type(self.max_batch) is not int or not 1 <= self.max_batch <= 4:
            raise ValueError("max_batch_size must be between 1 and 4")
        if self.rope_backend not in ("native", "triton"):
            raise ValueError("rope_backend must be native or triton")
        if self.device not in ("cpu", "cuda") or self.encoder_device not in ("cpu", "cuda"):
            raise ValueError("device and encoder_device must be cpu or cuda")
        if self.strategy == "stage_equal":
            self.max_batch = 1
            if self.batch_costs is not None or self.admission_window is not None:
                raise ValueError("batch_costs and admission_window require virtual_rank")
        self.model = resolve_model(self.model)
        root = Path(self.model)
        index = json.loads((root / "model_index.json").read_text())
        cfg = json.loads((root / "transformer" / "config.json").read_text())
        name = cfg.get("_class_name", index.get("transformer", [None, None])[1])
        self.num_blocks = cfg["num_layers"]
        self.channels = cfg["in_channels"]
        patch = cfg.get("patch_size", (1, 2, 2))
        self.patch_size = (1, patch, patch) if isinstance(patch, int) else tuple(patch)
        self.sigmas = (1.0, 0.9097071290016174, 0.7173174619674683, 0.0244140625, 0.0)
        self.fps, self.text_length = 24, None
        if name == "WanTransformer3DModel":
            self.backend, self.cache_text_kv, self.text_length = "wan", True, 512
            if self.channels == 48:
                self.height, self.width, self.spatial_scale, self.chunks = 704, 1280, 16, 29
            elif self.channels == 16 and self.num_blocks in (30, 40):
                self.height, self.width, self.spatial_scale, self.chunks = 480, 832, 8, 19
                self.fps = 16
                if self.num_blocks == 30:
                    self.sigmas = (1.0, 0.750257670879364, 0.499718636274338, 0.250000029802322, 0.0)
                    self.timesteps = (1000.0, 750.0, 500.0, 250.0)
                else:
                    self.sigmas = (0.700232386589050, 0.499718636274338, 0.400259941816330, 0.198347106575966, 0.0)
                    self.timesteps = (700.0, 500.0, 400.0, 200.0)
            else:
                raise ValueError("supported Wan models are 2.1 1.3B, 2.1 14B and 2.2 5B")
        elif name == "LTX2VideoTransformer3DModel":
            self.backend, self.dtype, self.compute_dtype = "ltx2", "float32", "bfloat16"
            self.height, self.width, self.spatial_scale, self.chunks = 544, 960, 32, 14
            self.patch_size, self.text_length = (1, 1, 1), 1024
            self.sigmas = (1.0, 0.8920687437057495, 0.6825016736984253, 0.10000002384185791, 0.0)
        elif name == "MiniMaxH3Transformer3DModel":
            self.backend = "h3"
            self.height, self.width, self.spatial_scale, self.chunks = 768, 1344, 16, 48
            self.timesteps = tuple(1.0 - value for value in self.sigmas[:-1])
        else:
            raise ValueError(f"unsupported Diffusers transformer: {name}")
        if not hasattr(self, "timesteps"):
            self.timesteps = tuple(1000.0 * value for value in self.sigmas[:-1])
        if self.ranks > 4 * self.num_blocks:
            raise ValueError("pipeline_parallel_size exceeds the number of denoising stages")
        for component in ("vae", "text_encoder", "tokenizer"):
            if not (root / component).is_dir():
                raise ValueError(f"Diffusers model is missing {component}/")
        self.make_layout()

    def make_layout(self):
        from .engine.strategies import StageEqual, VirtualRank

        if self.strategy == "stage_equal":
            return StageEqual(self.ranks)
        return VirtualRank(self.ranks, self.max_batch, self.batch_costs, self.admission_window)

    def make_backend(self, root=None):
        import torch

        options = dict(text_length=self.text_length, rope_backend=self.rope_backend)
        if self.backend == "wan":
            options["cache_text_kv"] = self.cache_text_kv
        if self.compute_dtype is not None:
            options["compute_dtype"] = getattr(torch, self.compute_dtype)
        return importlib.import_module(f"physisserve.models.{self.backend}").backend(root or self.model, **options)

    def make_sampler(self):
        from .engine.sampler import Sampler

        return Sampler(self.sigmas, self.timesteps)
