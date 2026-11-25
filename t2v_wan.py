"""Text-to-video reference clips with Wan 2.1 T2V 1.3B (diffusers), for the video -> ARDY pipeline.

Run with venv_t2v:
    venv_t2v\\Scripts\\python.exe t2v_wan.py --name pirate --prompt "..." [--seed 0]

The prompt is wrapped in a framing that asks for what GVHMR needs: one person, whole body
visible head to toe, locked-off camera. Output: clips/<name>.mp4 (16 fps, 832x480, 81 frames).
"""

import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
from diffusers import AutoencoderKLWan, WanPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

FRAMING = (
    "Static locked-off camera, wide shot, the whole body of a single man is visible from head to feet, "
    "standing in the middle of the frame on flat ground, realistic live-action film, natural light. "
)
NEGATIVE = (
    "camera movement, zoom, pan, close-up, cropped body, feet out of frame, multiple people, crowd, "
    "blurry, low quality, distorted limbs, extra limbs, deformed hands, text, watermark, cartoon, anime, "
    "static image, still frame"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = ROOT / "clips"
    out_dir.mkdir(exist_ok=True)

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
    # The ARDY demo shares this GPU: keep only the active module on it.
    pipe.enable_model_cpu_offload()

    prompt = FRAMING + args.prompt
    t0 = time.time()
    frames = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        height=480,
        width=832,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).frames[0]
    path = out_dir / f"{args.name}.mp4"
    export_to_video(frames, str(path), fps=16)
    (out_dir / f"{args.name}.txt").write_text(f"seed={args.seed}\naction={args.prompt}\nprompt={prompt}\n", encoding="utf-8")
    print(f"saved {path} in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
