import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .engine.sampler import DENOISING_STEPS


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
    max_batch: int | None = None
    device: str = "cuda"
    encoder_device: str = "cpu"
    batch_costs: tuple[tuple[float, ...], ...] | None = None
    admission_window: int | None = None
    backend: str = field(init=False)
    dtype: str = field(init=False)
    compute_dtype: str | None = field(init=False, default=None)
    supports_online_decode: bool = field(init=False, default=True)
    min_decode_chunks: int = field(init=False, default=1)

    def __post_init__(self):
        if not isinstance(self.model, (str, Path)) or not str(self.model).strip():
            raise ValueError("model must be a nonempty path or Hugging Face model ID")
        if self.ranks is None:
            raise ValueError("pipeline_parallel_size must be a positive integer")
        if self.strategy not in ("stage_equal", "virtual_rank"):
            raise ValueError("strategy must be stage_equal or virtual_rank")
        if self.max_batch is None:
            self.max_batch = 1 if self.strategy == "stage_equal" else DENOISING_STEPS
        if self.device not in ("cpu", "cuda") or self.encoder_device not in ("cpu", "cuda"):
            raise ValueError("device and encoder_device must be cpu or cuda")
        if self.strategy == "stage_equal":
            if type(self.max_batch) is not int or self.max_batch != 1:
                raise ValueError("stage_equal requires max_batch_size=1")
            if self.batch_costs is not None or self.admission_window is not None:
                raise ValueError("batch_costs and admission_window require virtual_rank")
        layout = self.make_layout()
        self.model = resolve_model(self.model)
        root = Path(self.model)
        index = json.loads((root / "model_index.json").read_text())
        cfg = json.loads((root / "transformer" / "config.json").read_text())
        name = cfg.get("_class_name", index.get("transformer", [None, None])[1])
        self.num_blocks = cfg["num_layers"]
        self.channels = cfg["in_channels"]
        if any(type(value) is not int or value < 1 for value in (self.num_blocks, self.channels)):
            raise ValueError("model num_layers and in_channels must be positive integers")
        patch = cfg.get("patch_size", (1, 2, 2))
        self.patch_size = (1, patch, patch) if isinstance(patch, int) else tuple(patch)
        if len(self.patch_size) != 3 or any(type(value) is not int or value < 1 for value in self.patch_size):
            raise ValueError("model patch_size must contain three positive integers")
        self.backend = {"WanTransformer3DModel": "wan", "LTX2VideoTransformer3DModel": "ltx2",
                        "MiniMaxH3Transformer3DModel": "h3"}.get(name)
        if self.backend is None:
            raise ValueError(f"unsupported Diffusers transformer: {name}")
        importlib.import_module(f"physisserve.models.{self.backend}").configure(self)
        if self.ranks > DENOISING_STEPS * self.num_blocks:
            raise ValueError("pipeline_parallel_size exceeds the number of denoising stages")
        layout.num_ranks(self.num_blocks)
        for component in ("vae", "text_encoder", "tokenizer"):
            if not (root / component).is_dir():
                raise ValueError(f"Diffusers model is missing {component}/")

    def make_layout(self):
        from .engine.strategies import StageEqual, VirtualRank

        if self.strategy == "stage_equal":
            return StageEqual(self.ranks)
        return VirtualRank(self.ranks, self.max_batch, self.batch_costs, self.admission_window)

    def make_backend(self, root=None):
        return importlib.import_module(f"physisserve.models.{self.backend}").backend(root or self.model)

    def make_sampler(self):
        from .engine.sampler import Sampler

        return Sampler(self.sigmas, self.timesteps)
