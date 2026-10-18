"""Shared native-batch validation and independent pre-RoPE KV publication."""

import torch

from ..engine.kv_cache import Cache


def validate_batch(dit, index, hidden_list, timestep_list, chunk_list, cache_list, blocks):
    """Validate members of one prepared stream and return their common history length."""
    if dit.model is None or not hasattr(dit, "shape"):
        raise RuntimeError(f"load and prepare {type(dit).__name__} before calling block_batch")
    if type(index) is not int or str(index) not in blocks:
        raise ValueError("block_batch index must be an owned block")
    members = len(hidden_list)
    if not members or any(len(items) != members for items in (timestep_list, chunk_list, cache_list)):
        raise ValueError("block_batch requires nonempty, equally sized hidden/timestep/chunk/cache lists")
    expected = dit.hidden_shape(dit.shape)
    if any(not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != expected
           or hidden.device != dit.text.device or hidden.dtype != dit.dtype for hidden in hidden_list):
        raise ValueError("block_batch hidden members must match the prepared shape, device and dtype")
    if any(not isinstance(timestep, torch.Tensor) or timestep.ndim != 0
           or timestep.device.type != "cpu" or timestep.dtype != torch.float32
           or not torch.isfinite(timestep).item() for timestep in timestep_list):
        raise ValueError("block_batch timesteps must be finite float32 CPU scalar tensors")
    if any(type(chunk) is not int or not 0 <= chunk < dit.chunks for chunk in chunk_list):
        raise ValueError("block_batch chunks must be integers in the prepared stream")
    if any(not isinstance(cache, Cache) for cache in cache_list) \
            or len({id(cache) for cache in cache_list}) != members:
        raise ValueError("block_batch requires a distinct Cache for every member")
    if any(cache.page is not None for cache in cache_list):
        raise ValueError("block_batch requires fresh caches that have not already written a page")
    lengths = [len(cache.history) for cache in cache_list]
    if len(set(lengths)) != 1:
        raise ValueError("block_batch requires equal history lengths")
    if any(length > chunk for length, chunk in zip(lengths, chunk_list)):
        raise ValueError("block_batch history cannot contain more pages than preceding chunks")
    page_shape = dit.kv_shape(dit.shape)
    first_shape = (*dit.shape[:2], dit.first_frames, *dit.shape[3:])
    sink_shape = dit.kv_shape(first_shape)
    for cache in cache_list:
        for position, page in enumerate(cache.history):
            if not isinstance(page, (tuple, list)) or len(page) != 2 or any(
                not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
                or tensor.device != dit.text.device or tensor.dtype != dit.dtype for tensor in page
            ) or page[0].shape != page[1].shape:
                raise ValueError(
                    "block_batch history pages must be matching KV tensors on the prepared device and dtype"
                )
            expected = sink_shape if position == 0 else page_shape
            if tuple(page[0].shape) != expected:
                raise ValueError("block_batch history must be the fixed sink page followed by steady chunk pages")
    return lengths[0]


class BatchCache:
    """Stack equal-length windows, retaining only each member's own pre-RoPE pages."""

    def __init__(self, caches, batch_size):
        self.caches, self.batch_size = caches, batch_size

    def __call__(self, key, value):
        caches, batch = self.caches, self.batch_size
        direct = (
            not torch.is_grad_enabled() and type(batch) is int and batch > 0
            and type(key) is torch.Tensor and type(value) is torch.Tensor
            and key.ndim == 4 and key.shape == value.shape and key.numel() > 0
            and bool(caches) and key.shape[0] == len(caches) * batch
            and len({id(cache) for cache in caches}) == len(caches)
            and all(type(cache) is Cache and cache.page is None for cache in caches)
            and len({len(cache.history) for cache in caches}) == 1
            and bool(caches[0].history)
        )
        if direct:
            shapes = []
            for page in caches[0].history:
                if (not isinstance(page, (tuple, list)) or len(page) != 2
                        or any(type(tensor) is not torch.Tensor or tensor.ndim != 4 for tensor in page)
                        or page[0].shape != page[1].shape):
                    direct = False
                    break
                shape = tuple(page[0].shape)
                if shape[0] != batch or shape[1] != key.shape[1] or shape[2] < 1 or shape[3] != key.shape[3]:
                    direct = False
                    break
                shapes.append(shape)
            direct = direct and key.layout == value.layout == torch.strided and all(
                isinstance(page, (tuple, list)) and len(page) == 2
                and all(type(tensor) is torch.Tensor and tuple(tensor.shape) == shapes[position]
                        and tensor.dtype == template.dtype and tensor.device == template.device
                        and tensor.layout == torch.strided
                        for tensor, template in zip(page, (key, value)))
                for cache in caches for position, page in enumerate(cache.history)
            )
        if direct:
            window_shape = (*key.shape[:2], key.shape[2] + sum(shape[2] for shape in shapes), key.shape[3])
            packed = (key.new_empty(window_shape), value.new_empty(window_shape))
            for member, (cache, k, v) in enumerate(zip(caches, key.split(batch), value.split(batch))):
                # Keep independent pages; concatenate directly into the final batch window.
                cache.page = (k.clone(), v.clone())
                parts = [*cache.history, cache.page]
                for component in (0, 1):
                    torch.cat([page[component] for page in parts], dim=2,
                              out=packed[component][member * batch:(member + 1) * batch])
            return packed

        # Preserve callbacks, autograd and empty-history behavior.
        pages = []
        for cache, k, v in zip(self.caches, key.split(self.batch_size), value.split(self.batch_size)):
            # A sink can outlive its batch peers; do not keep the packed storage alive.
            pages.append(cache(k.clone(), v.clone()))
        return tuple(torch.cat([page[i] for page in pages], dim=0) for i in (0, 1))
