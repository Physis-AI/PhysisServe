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

import imageio.v3 as iio

from physisserve import PhysisServe, SamplingParams


def main():
    model = PhysisServe("Physis-AI/physisserve-wan2p1_1p3b-same_step-sdv2", pipeline_parallel_size=1)
    sampling_params = SamplingParams(mode="offline_decode")
    prompts = [
        "A corgi running along the beach at sunset.",
        "A paper boat floating down a quiet forest stream.",
    ]
    outputs = model.generate(prompts, sampling_params)

    for index, output in enumerate(outputs):
        iio.imwrite(f"video-{index}.mp4", output["video"].numpy(), fps=output["fps"],
                    codec="libx264", pixelformat="yuv420p", macro_block_size=None,
                    output_params=["-movflags", "+faststart"])


if __name__ == "__main__":
    main()
