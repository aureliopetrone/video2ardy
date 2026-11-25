# video2ardy — video-driven motion synthesis for NVIDIA ARDY

Text-to-motion models only know the actions their text encoder can name. This experiment turns a
**video** (a real clip, or a clip generated from a rich prompt by a text-to-video model) into an
animation produced by [ARDY](https://github.com/nv-tlabs/ardy). ARDY is used here as a
physics-plausible motion prior and in-betweener, not as a blind text-to-motion generator. It is
guided by sparse key-constraints taken from the video and by SDEdit-style initialisation from the
video's own latent tokens, and a spatio-temporal critic cleans what the video estimator gets wrong.

![noble clip: video, direct retarget, text only, Mode A, Mode B](results/preview_noble.gif)

*Reference video (Wan 2.1) · direct GVHMR retarget · ARDY text only · Mode A · Mode B 0.3/0.5/0.8 ·
Mode B sparse · critic cleanup only. The full side-by-side for five clips is in
[`results/comparison_5clips.mp4`](results/comparison_5clips.mp4).*

## Pipeline

```
rich text prompt ──(Wan 2.1 T2V 1.3B, optional)──► reference video (.mp4)
        │
        ▼  1. video_extractor.py      GVHMR → world-grounded SMPL-X motion, contact and 2D-confidence per frame
        ▼  2. video_to_ardy_adapter.py
        │     retarget SMPL-X → ARDY SOMA30 skeleton (30 fps)
        │     critic cleanup of the estimate (NDF prior + joint-range/jerk + foot contact), planted feet kept
        │     unreliable frames flagged (low confidence, or bodies no person can make: T2V hallucinations)
        │     sparse constraints: smoothed root path + heading, hands, adaptive full-body anchors (K = 10–25)
        │     reference latent tokens z_ref through ARDY's own autoencoder
        ▼  3. ardy_video_projector.py
              Mode A  constraints only, from pure noise, empty text prompt
              Mode B  constraints + SDEdit: z_ref noised to strength·T replaces the window's tokens
              both through the critic with silent rollback; unreliable stretches are left to ARDY
```

`run_video_test.py` runs the comparison and `render_compare.py` renders it. `prompt_to_ardy.py`
chains everything from a prompt (or a video file) to the comparison video.

## Results

Five clips generated locally with Wan 2.1 T2V 1.3B (5 s, 16 fps, resampled to 30 fps): an exhausted
pirate captain scanning the horizon, an arrogant nobleman's dismissive gesture, a nervous thief
checking corners and crouching, a captain giving orders, and a drunk sailor. Prompts are in
`clips/*.txt`. Per-clip tables are in [`results/summary.md`](results/summary.md). The table below
gives medians over the five clips.

| condition | foot skate cm/s | NDF | bio stress | pos. jerk | PA-MPJPE cm | hands err. cm | pose var. | asym. | seam start | seam end |
|---|---|---|---|---|---|---|---|---|---|---|
| direct retarget (GVHMR) | 1.03 | 3.33 | 10.66 | 85.9 | 0 | 0 | 1.00 | 1.00 | 3.87 | 10.36 |
| retarget + critic cleanup only | 3.29 | 0.00 | 0.00 | 87.4 | 1.82 | 0.78 | 1.02 | 1.01 | – | – |
| ARDY, text prompt only | 1.21 | 0.12 | 0.00 | 28.6 | 18.14 | 37.30 | 0.34 | 0.27 | – | – |
| **Mode A** + critic | 2.90 | 0.37 | 0.10 | 52.6 | 2.75 | 2.81 | 1.04 | 0.96 | 0.60 | 1.50 |
| Mode B s=0.3 + critic | 3.93 | 0.48 | 0.16 | 59.8 | 2.60 | 2.76 | 1.08 | 0.96 | 0.49 | 1.35 |
| **Mode B s=0.5** + critic | 3.20 | 0.54 | 0.63 | 56.5 | 2.64 | 2.81 | 1.05 | 0.96 | 0.56 | 1.19 |
| Mode B s=0.8 + critic | 2.57 | 0.59 | 0.47 | 60.5 | 2.73 | 2.89 | 1.05 | 0.94 | 0.58 | 1.21 |
| Mode A sparse + critic | 2.09 | 0.23 | 0.00 | 39.5 | 3.33 | 5.93 | 0.94 | 0.84 | 0.67 | 1.23 |
| Mode B s=0.5 sparse + critic | 2.54 | 0.28 | 0.03 | 56.1 | 2.73 | 4.10 | 0.91 | 0.90 | 0.52 | 1.13 |

How to read it:

- **NDF and bio stress** are the critic's scores: distance from the manifold of clean motion, and
  joint-range/jerk violations. Lower is better.
- **PA-MPJPE and hands error** are the distance from the video estimate, measured only on frames it
  can be trusted on. **Pose variance and asymmetry** are ratios to the video.
- **Seam** is the joint acceleration around the start and end of the video segment inside a longer
  ARDY sequence, divided by the sequence's 90th percentile. About 1 means no visible seam. For the
  direct retarget, the clip is pasted between ARDY frames.
- Sparse means a root constraint every 10 frames, hands only on the anchors, and anchors 15–30
  frames apart.

What the numbers say:

- **Text only does not perform the scene.** ARDY moves cleanly but is about 37 cm away at the
  hands: the nobleman never makes the gesture and the thief never crouches.
- **The direct retarget is faithful but off the manifold.** It scores NDF 3.3 and bio stress 10.7,
  and it pops when pasted into an ARDY sequence (seam 4–10).
- **Mode A and Mode B keep the acting.** Hands stay about 3 cm from the video, with near-clean
  critic scores and seamless stitching.
- **With sparse constraints, Mode B keeps more of the video than Mode A.** Hands error is 4.1 cm
  against 5.9, and pose variance is closer to the video. This is the style that SDEdit carries and
  sparse constraints lose.
- **The T2V output is not always usable.** In the drunk-sailor clip, Wan turns the actor upside down
  for about a second. Those 51 frames are flagged (critic energy > 300 or motion prior > 9), get no
  constraints and no SDEdit tokens, and ARDY stumbles through them upright.

Lessons measured on the way:

- Joints that SMPL-X lacks (finger ends, eyes, jaw) must take ARDY's typical local rotation, not
  identity. Straight fingers alone pushed the bio stress to 125.
- With dense video constraints, the in-line critic finds no free frame to place corrective keys on.
  That is why the cleanup happens once on the whole clip, before constraints and z_ref are taken.
  The in-line critic adds little on top of it (compare `modeB_0.5_nc`, without critic, in
  `summary.md`).
- ARDY's own `post_process_motion` doubled foot skate on a fast clip (tennis), so it is off.
- Every ARDY output, text-only included, carries a small 4-frame (one-token) flicker. It dominates
  positional jerk, which is why the seam metric uses acceleration.

Limitations:

- On these mostly standing clips, ARDY adds a little foot sliding (2–4 cm/s against 1 cm/s for the
  video). On a fast real clip (tennis, not published here because it is third-party footage), the
  pipeline cut sliding from 35 to 12–15 cm/s.
- The renders are stick figures. Nothing has been tested in a game engine yet.
- Tested only on Windows 11 with an RTX 5070 Ti.

## Setup (Windows, as tested)

Folders sit side by side (all paths can be overridden with the environment variables in brackets):

```
Tools\
  ardy\          clone of nv-tlabs/ardy, installed with its venv            [ARDY_ROOT, ARDY_PYTHON]
  ardy_critic\   the critic package (see below)                             [ARDY_CRITIC_ROOT]
  video2ardy\    this repository
```

1. **ARDY.** Follow its README: SOMA-RP-30FPS model, text encoder server on 127.0.0.1:9550. This repo
   uses ARDY's venv for the ARDY steps and installs nothing into it.
2. **The critic.** `ardy_critic` is a separate package: a spatio-temporal NDF plus a biomechanical
   and contact critic that wraps `Ardy.__call__` with silent rollback. It is not part of this
   repository. The ARDY steps import it from `ARDY_CRITIC_ROOT`.
3. **GVHMR**, in its own venv:
   ```
   git clone --depth 1 https://github.com/zju3dv/GVHMR
   py -3.11 -m venv venv_gvhmr
   venv_gvhmr\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   venv_gvhmr\Scripts\python -m pip install timm==0.9.12 lightning hydra-core==1.3.2 hydra-zen hydra_colorlog rich "numpy<2" matplotlib tensorboardX opencv-python ffmpeg-python scikit-image termcolor einops imageio av joblib trimesh smplx ultralytics==8.2.42 lapx yacs pycolmap wis3d imageio-ffmpeg
   venv_gvhmr\Scripts\python -m pip install --no-deps -e GVHMR
   git clone --depth 1 https://github.com/facebookresearch/pytorch3d third_party_pytorch3d
   venv_gvhmr\Scripts\python tools\make_pytorch3d_shim.py third_party_pytorch3d
   ```
   - Put GVHMR's checkpoints (gvhmr, hmr2, vitpose, yolo) in `GVHMR/inputs/checkpoints/` as its
     INSTALL.md says.
   - Copy an `ffmpeg.exe` to `bin/`.
   - **No SMPL/SMPL-X model files are needed.** GVHMR predicts SMPL-X parameters, and the only
     thing it needs a body model for at inference is the skeleton. `video_extractor.py` gives it a
     skeleton-only stand-in built from the neutral SMPL-X rest joints that NVIDIA Kimodo ships
     (`kimodo/assets/skeletons/smplx22/joints.p`) [`SMPLX22_JOINTS`]. Body shape is ignored, which
     is harmless because the motion is retargeted by rotations.
4. **Wan 2.1** (optional, for prompt → video):
   ```
   py -3.11 -m venv venv_t2v
   venv_t2v\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   venv_t2v\Scripts\python -m pip install diffusers transformers accelerate ftfy imageio imageio-ffmpeg sentencepiece
   ```
   Weights download on first use (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`). One clip takes about 10 min
   on a shared RTX 5070 Ti, with a peak of 10.7 GB.

## Usage

```
py -3.11 prompt_to_ardy.py --name noble --prompt "An arrogant nobleman ... waves one hand dismissively ..."
py -3.11 prompt_to_ardy.py --name myclip --video path\to\clip.mp4 --prompt "text for the text-only baseline"
```

Or step by step:

```
venv_t2v\Scripts\python t2v_wan.py --name noble --prompt "..."
venv_gvhmr\Scripts\python video_extractor.py clips\noble.mp4
..\ardy\venv\Scripts\python run_video_test.py --clips noble [--strengths 0.3 0.5 0.8]
venv_gvhmr\Scripts\python render_compare.py results\noble\bundle.npz results\noble\compare.mp4
```

By default the benchmark runs Mode A and Mode B at strength 0.5.

## Related work

The pieces exist separately:

- T2V clips as motion data for training a new text-to-motion model:
  [ViMoGen](https://arxiv.org/abs/2510.26794).
- Generated videos → 3D motion → humanoid robots through RL tracking:
  [GenMimic](https://arxiv.org/abs/2512.05094) and
  [synthetic video scenarios](https://www.alphaxiv.org/abs/2607.21648).
- SDS from a video model on an SMPL-X body:
  [Towards motion from video diffusion models](https://arxiv.org/abs/2411.12831).
- SDEdit on motion retrieved from a library: [RMD](https://arxiv.org/abs/2412.04343).
- Diffusion priors inside video motion estimation:
  [COIN](https://arxiv.org/abs/2408.16426) and
  [Motion Diffusion-Guided HMR](https://arxiv.org/abs/2411.10582).

We did not find the combination tried here: projecting a T2V clip into an off-the-shelf
autoregressive motion model, without any training, through SDEdit in its latent space, its native
kinematic constraints and a critic cleanup.

## Credits and licences

- This repository's code: Apache-2.0 (see `LICENSE`).
- [ARDY](https://github.com/nv-tlabs/ardy) (NVIDIA, Apache-2.0 code; model weights under the NVIDIA
  Open Model Agreement).
- [GVHMR](https://github.com/zju3dv/GVHMR) (Zhejiang University; see its licence, research use).
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1) (Apache-2.0); the clips in `clips/` were generated
  with it.
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) transforms (BSD).
- SMPL-X rest joints come from NVIDIA Kimodo's assets and are not redistributed here.
