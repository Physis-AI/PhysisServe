import importlib
import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from .engine.sampler import DENOISING_STEPS


def resolve_model(model):
    root = Path(model).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(str(model)))
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
        if type(self.ranks) is not int or self.ranks < 1:
            raise ValueError("pipeline_parallel_size must be a positive integer")
        if self.strategy not in ("stage_equal", "virtual_rank"):
            raise ValueError("strategy must be stage_equal or virtual_rank")
        if self.max_batch is None:
            self.max_batch = 1 if self.strategy == "stage_equal" else DENOISING_STEPS
        if type(self.max_batch) is not int or not 1 <= self.max_batch <= DENOISING_STEPS:
            raise ValueError("max_batch_size must be an integer from 1 to 4")
        if self.admission_window is not None and (type(self.admission_window) is not int or self.admission_window < 1):
            raise ValueError("admission_window must be a positive integer")
        if self.device not in ("cpu", "cuda") or self.encoder_device not in ("cpu", "cuda"):
            raise ValueError("device and encoder_device must be cpu or cuda")
        if self.strategy == "stage_equal":
            if self.max_batch != 1:
                raise ValueError("stage_equal requires max_batch_size=1")
            if self.batch_costs is not None or self.admission_window is not None:
                raise ValueError("batch_costs and admission_window require virtual_rank")
        if self.batch_costs is not None:
            rows = tuple(tuple(row) for row in self.batch_costs)
            if (not rows or any(len(row) < self.max_batch or any(
                    type(value) not in (int, float) or not isfinite(value) or value <= 0 for value in row)
                    for row in rows)):
                raise ValueError("batch_costs must contain finite positive costs for every batch size")
            self.batch_costs = tuple(row[:self.max_batch] for row in rows)
            if not isfinite(sum(map(sum, self.batch_costs))) or not isfinite(4 * sum(row[0] for row in rows)):
                raise ValueError("total batch_costs must be finite")
        self.model = resolve_model(self.model)
        root = Path(self.model)
        index = json.loads((root / "model_index.json").read_text())
        cfg = json.loads((root / "transformer" / "config.json").read_text())
        name = cfg.get("_class_name", index.get("transformer", [None, None])[1])
        self.num_blocks = cfg["num_layers"]
        self.channels = cfg["in_channels"]
        patch = cfg.get("patch_size", (1, 2, 2))
        self.patch_size = (1, patch, patch) if isinstance(patch, int) else tuple(patch)
        self.backend = {"WanTransformer3DModel": "wan", "LTX2VideoTransformer3DModel": "ltx2",
                        "MiniMaxH3Transformer3DModel": "h3"}.get(name)
        if self.backend is None:
            raise ValueError(f"unsupported Diffusers transformer: {name}")
        importlib.import_module(f"physisserve.models.{self.backend}").configure(self)
        stages = DENOISING_STEPS * self.num_blocks
        if self.ranks > stages:
            raise ValueError("pipeline_parallel_size exceeds the number of denoising stages")
        if self.batch_costs is not None and len(self.batch_costs) != self.num_blocks:
            raise ValueError("batch_costs must contain one row per block")
        if self.admission_window is not None and self.admission_window < stages:
            raise ValueError("admission_window must be at least the stage count")

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
