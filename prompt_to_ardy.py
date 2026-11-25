"""End to end: rich text prompt -> Wan 2.1 clip -> GVHMR -> ARDY Mode A/B + critic -> comparison video.

    py -3.11 prompt_to_ardy.py --name drunk --prompt "A drunk sailor staggers ..." [--seed 0] [--video file.mp4]

Each stage runs in its own environment (venv_t2v, venv_gvhmr, ARDY's venv). With --video the T2V
stage is skipped and the given clip is used. Outputs: clips/<name>.mp4, extracted/<name>.npz,
results/<name>/{metrics.json,bundle.npz,compare.mp4}.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY_T2V = ROOT / "venv_t2v" / "Scripts" / "python.exe"
PY_GVHMR = ROOT / "venv_gvhmr" / "Scripts" / "python.exe"
PY_ARDY = Path(os.environ.get("ARDY_PYTHON", ROOT.parent / "ardy" / "venv" / "Scripts" / "python.exe"))


def run(py, *args):
    print(">>", Path(py).parent.parent.name, *args, flush=True)
    subprocess.run([str(py), *map(str, args)], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", help="action to film and to give the text-only baseline")
    ap.add_argument("--video", help="use this clip instead of generating one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strengths", nargs="+", default=["0.5"])  # Mode A + Mode B 0.5 by default
    args = ap.parse_args()
    clip = ROOT / "clips" / f"{args.name}.mp4"
    if args.video:
        clip.parent.mkdir(exist_ok=True)
        shutil.copy(args.video, clip)
        if args.prompt:
            (ROOT / "clips" / f"{args.name}.txt").write_text(f"action={args.prompt}\nsource={args.video}\n", encoding="utf-8")
    else:
        assert args.prompt, "--prompt is needed to generate the clip"
        run(PY_T2V, "t2v_wan.py", "--name", args.name, "--prompt", args.prompt, "--seed", args.seed)
    run(PY_GVHMR, "video_extractor.py", clip)
    run(PY_ARDY, "run_video_test.py", "--clips", args.name, "--strengths", *args.strengths)
    res = ROOT / "results" / args.name
    run(PY_GVHMR, "render_compare.py", res / "bundle.npz", res / "compare.mp4")
    print("done:", res / "compare.mp4")


if __name__ == "__main__":
    main()
