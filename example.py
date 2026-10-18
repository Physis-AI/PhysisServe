import argparse

import imageio.v3 as iio

from physisserve import PhysisServe, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Diffusers directory or Hugging Face model ID")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--prompt", nargs="+", default=["A corgi running along the beach at sunset."])
    parser.add_argument("--chunks", type=int, default=None)
    args = parser.parse_args()

    with PhysisServe(args.model, pipeline_parallel_size=args.gpus) as model:
        outputs = model.generate(args.prompt, SamplingParams(num_chunks=args.chunks, seed=0))
        for index, output in enumerate(outputs):
            iio.imwrite(f"video-{index}.mp4", output["video"].numpy(), fps=output["fps"],
                        codec="libx264", pixelformat="yuv420p", macro_block_size=None,
                        output_params=["-movflags", "+faststart"])


if __name__ == "__main__":
    main()
