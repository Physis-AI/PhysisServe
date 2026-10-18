"""Pack independent pre-RoPE KV pages into a native batch window."""

import torch


class BatchCache:
    """Stack equal-length windows, retaining only each member's own pre-RoPE pages."""

    def __init__(self, caches, batch_size):
        self.caches, self.batch_size = caches, batch_size

    def __call__(self, key, value):
        caches, batch = self.caches, self.batch_size
        history_tokens = sum(page[0].shape[2] for page in caches[0].history)
        window_shape = (*key.shape[:2], history_tokens + key.shape[2], key.shape[3])
        packed = (key.new_empty(window_shape), value.new_empty(window_shape)) if history_tokens else (key, value)
        for member, (cache, k, v) in enumerate(zip(caches, key.split(batch), value.split(batch))):
            # A sink can outlive its batch peers; retain no views into the packed storage.
            cache.page = (k.clone(), v.clone())
            if history_tokens:
                parts = [*cache.history, cache.page]
                for component in (0, 1):
                    torch.cat([page[component] for page in parts], dim=2,
                              out=packed[component][member * batch:(member + 1) * batch])
        return packed
