import argparse

import imageio.v3 as iio

from physisserve import PhysisServe, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Diffusers directory or Hugging Face model ID")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--prompt", nargs="+", default=["A corgi running along the beach at sunset."])
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--mode", choices=("dit_only", "deferred_decode", "stream_decode"),
                        default="deferred_decode")
    parser.add_argument("--latent-dir", help="optionally save denoised latents in this directory")
    args = parser.parse_args()

    with PhysisServe(args.model, pipeline_parallel_size=args.gpus) as model:
        outputs = model.generate(args.prompt, SamplingParams(num_chunks=args.chunks, seed=0,
                                                            mode=args.mode, latent_dir=args.latent_dir))
        for index, output in enumerate(outputs):
            if output.get("status") == "rejected":
                print(f"stream {index} rejected: {output['reason']}")
                continue
            if "video" not in output:
                print(f"stream {index}: latents {tuple(output['latents'].shape)}"
                      + (f", saved to {output['latent_path']}" if "latent_path" in output else ""))
                continue
            iio.imwrite(f"video-{index}.mp4", output["video"].numpy(), fps=output["fps"],
                        codec="libx264", pixelformat="yuv420p", macro_block_size=None,
                        output_params=["-movflags", "+faststart"])


if __name__ == "__main__":
    main()
