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
