# Required Notice: Copyright 2026 Physis AI, Inc. and/or its affiliates
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://polyformproject.org/licenses/noncommercial/1.0.0
#
# This software is provided "AS IS", without warranties or conditions.
# See the License for permissions, limitations, and liability terms.

import atexit

from .config import Config
from .sampling_params import SamplingParams


class PhysisServe:
    """One prompt or a list of concurrent streams, on one model and GPU pool.

    Multi-GPU execution starts its own processes; run scripts under a
    ``if __name__ == '__main__':`` guard. Parallelism is over denoising stages.
    For multiple nodes, use ``torchrun ... --module physisserve.launch script.py``;
    the launcher supplies the GPU pool size without changing the script.
    """

    def __init__(self, model, *, pipeline_parallel_size=1, strategy="virtual_rank",
                 max_batch_size=None, device="cuda", encoder_device="cpu",
                 batch_costs=None, admission_window=None):
        from . import launch

        self.config = Config(model, ranks=launch.ranks(pipeline_parallel_size), strategy=strategy,
                             max_batch=max_batch_size, device=device, encoder_device=encoder_device,
                             batch_costs=batch_costs, admission_window=admission_window)
        self._workers = launch.load(self.config, model)
        self._closed = False
        atexit.register(self.close)

    def _call(self, operation, *args, **kwargs):
        if self._closed:
            raise RuntimeError("PhysisServe is closed")
        try:
            return self._workers.call(operation, *args, **kwargs)
        except BaseException:
            self.close()
            raise

    def _inputs(self, prompts, sampling_params):
        if not isinstance(prompts, (str, list, tuple)):
            raise TypeError("prompts must be a string or a list/tuple of strings")
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("each prompt must be a string")
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, (SamplingParams, list, tuple)):
            raise TypeError("sampling_params must be SamplingParams or a list/tuple of SamplingParams")
        params = ([sampling_params] * len(prompts) if isinstance(sampling_params, SamplingParams)
                  else list(sampling_params))
        if len(params) != len(prompts) or any(not isinstance(item, SamplingParams) for item in params):
            raise ValueError("provide one SamplingParams or one per prompt")
        return prompts, [param.resolve(self.config) for param in params]

    def generate(self, prompts, sampling_params=None):
        """Return ordered CPU latents and, for decoding modes, uint8 THWC videos.

        Every prompt has its own KV cache and seed.
        ``SamplingParams.mode`` selects DiT only, decoding after all streams finish,
        or decoding completed chunks during denoising. Offline decoding first
        saves each stream's latents to disk; ``latent_path`` identifies the file.
        """
        prompts, params = self._inputs(prompts, sampling_params)
        if not prompts:
            return []
        self._call("prepare", prompts, params)
        outputs = self._call("run")[0]
        offline = [index for index, param in enumerate(params)
                   if param.mode == "offline_decode" and outputs[index].get("status") != "rejected"]
        if offline:
            decoded = self.decode([outputs[index] for index in offline])
            for index, output in zip(offline, decoded):
                outputs[index] = output
        return outputs

    def decode(self, outputs):
        """Decode previously generated latents after inference or benchmarking."""
        return self._call("decode", outputs)[0]

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        workers, self._workers = self._workers, None
        workers.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
