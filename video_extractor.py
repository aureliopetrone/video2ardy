"""Module 1 of the video -> ARDY pipeline: world-grounded 3D human motion from a single-person video.

Runs GVHMR (zju3dv/GVHMR, cloned in ./GVHMR) without the SMPL / SMPL-X body model files, which are
licence-gated: GVHMR's network predicts SMPL-X parameters directly, and the only place the body model
is needed at inference is the skeleton for forward kinematics (foot-lock post-processing and our
output joints). We give it a skeleton-only stand-in built from the neutral SMPL-X rest joints that
Kimodo ships (kimodo/assets/skeletons/smplx22/joints.p). Body shape (betas) is therefore ignored for
FK; that is harmless here because the motion is retargeted to ARDY's SOMA skeleton by rotations.

Run with venv_gvhmr (cwd anywhere):
    venv_gvhmr\\Scripts\\python.exe video_extractor.py clips\\pirate.mp4 [--moving_cam] [--fps30]

Output: extracted/<name>.npz with
    fps                  frames per second of the extracted motion (30)
    body_pose  (T,21,3)  SMPL-X body pose, axis-angle (local)
    global_orient (T,3)  root orientation in the gravity-aligned world (y up), axis-angle
    transl     (T,3)     SMPL-X translation in that world (after GVHMR's foot-lock post-processing)
    transl_raw (T,3)     translation before post-processing
    betas      (T,10)
    joints     (T,22,3)  world joints from FK on the neutral skeleton (y up, metres)
    rest_joints (22,3)   the neutral rest joints used
    static_conf (T,6)    GVHMR static (contact) probability for [L_ankle, L_foot, R_ankle, R_foot, L_wrist, R_wrist]
    kp2d_conf  (T,17)    ViTPose COCO-17 keypoint confidence per frame
    joint_conf (T,22)    per-SMPL-joint confidence mapped from the 2D keypoints (occlusion / hallucination proxy)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GVHMR = ROOT / "GVHMR"
# Neutral SMPL-X rest joints (22 body joints) as shipped by NVIDIA Kimodo (not redistributed here):
# point SMPLX22_JOINTS at kimodo/assets/skeletons/smplx22/joints.p of a Kimodo checkout.
KIMODO_SMPLX22 = Path(os.environ.get(
    "SMPLX22_JOINTS", ROOT.parent / "kimodo_locale" / "kimodo" / "kimodo" / "assets" / "skeletons" / "smplx22" / "joints.p"))

os.environ["PATH"] = str(ROOT / "bin") + os.pathsep + os.environ["PATH"]  # ffmpeg.exe
sys.path.insert(0, str(GVHMR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SMPLX_PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# COCO-17 keypoints -> SMPL-X body joints they observe (for a per-joint confidence)
# COCO: 0 nose 1 leye 2 reye 3 lear 4 rear 5 lsho 6 rsho 7 lelb 8 relb 9 lwri 10 rwri
#       11 lhip 12 rhip 13 lknee 14 rknee 15 lank 16 rank
COCO_FOR_JOINT = {
    0: [11, 12], 1: [11], 2: [12], 3: [11, 12, 5, 6], 4: [13], 5: [14], 6: [5, 6, 11, 12],
    7: [15], 8: [16], 9: [5, 6], 10: [15], 11: [16], 12: [5, 6, 0], 13: [5], 14: [6],
    15: [0, 1, 2, 3, 4], 16: [5], 17: [6], 18: [7], 19: [8], 20: [9], 21: [10],
}


class SkeletonOnlySMPLX(torch.nn.Module):
    """Stand-in for GVHMR's SMPL-X body model: parents + rest skeleton, no mesh."""

    def __init__(self):
        super().__init__()
        rest = torch.load(KIMODO_SMPLX22, weights_only=True).reshape(22, 3).float()
        self.register_buffer("rest", rest, persistent=False)
        self.parents = torch.tensor(SMPLX_PARENTS_22 + [15] * 33)  # only [:22] is used

    def get_skeleton(self, betas):
        return self.rest.to(betas.device).expand(*betas.shape[:-1], 22, 3)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Mesh output needs the SMPL-X model files, which this setup does not use.")


def install_stub():
    # torch >= 2.6 loads with weights_only=True: right for the mirrored GVHMR/HMR2/ViTPose weights
    # (plain tensors), but ultralytics pickles whole modules. Only the YOLO file, taken from the
    # official ultralytics release, is allowed a full unpickle.
    _orig_load = torch.load

    def _load(f, *a, **k):
        if str(f).replace("\\", "/").endswith("checkpoints/yolo/yolov8x.pt"):
            k["weights_only"] = False
        return _orig_load(f, *a, **k)

    torch.load = _load

    import hmr4d.utils.smplx_utils as su

    def make_smplx(type="neu_fullpose", **kwargs):
        return SkeletonOnlySMPLX()

    su.make_smplx = make_smplx
    import hmr4d.model.gvhmr.utils.endecoder as ed

    ed.make_smplx = make_smplx


def ffmpeg_to_30fps(src: Path, dst: Path):
    """Motion-compensated resample to 30 fps (GVHMR is trained on 30 fps); copies if already 30."""
    import imageio.v3 as iio

    meta = iio.immeta(src, plugin="pyav")
    fps = float(meta.get("fps", 30))
    if abs(fps - 30) < 0.01:
        shutil.copy(src, dst)
        return fps
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(dst),
    ]
    subprocess.run(cmd, check=True)
    return fps


@torch.no_grad()
def extract(video: Path, out_dir: Path, static_cam: bool, f_mm=None):
    os.chdir(GVHMR)  # GVHMR loads its checkpoints by relative path
    install_stub()

    import hydra
    from hydra import compose, initialize_config_module

    import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401  (registers model/gvhmr/gvhmr_pl_demo)
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor, estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
    from hmr4d.utils.video_io_utils import get_video_lwh

    name = video.stem
    work = out_dir / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    v30 = work / "input_30fps.mp4"
    src_fps = ffmpeg_to_30fps(video, v30)
    length, width, height = get_video_lwh(v30)
    print(f"[extract] {video.name}: source {src_fps:.1f} fps -> 30 fps, {length} frames, {width}x{height}")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=[f"video_name={name}", f"static_cam={static_cam}"])

    bbx_p, vitpose_p, feat_p = work / "bbx.pt", work / "vitpose.pt", work / "vit_features.pt"
    if not bbx_p.exists():
        bbx_xyxy = Tracker().get_one_track(str(v30)).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, bbx_p)
    bbx = torch.load(bbx_p)
    bbx_xys = bbx["bbx_xys"]
    if not vitpose_p.exists():
        torch.save(VitPoseExtractor().extract(str(v30), bbx_xys), vitpose_p)
    vitpose = torch.load(vitpose_p)
    if not feat_p.exists():
        torch.save(Extractor().extract_video_features(str(v30), bbx_xys), feat_p)
    feats = torch.load(feat_p)

    if static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        vo = SimpleVO(str(v30), scale=0.5, step=8, method="sift", f_mm=f_mm).compute()
        R_w2c = torch.from_numpy(vo[:, :3, :3])
    K = (create_camera_sensor(width, height, f_mm)[2] if f_mm else estimate_K(width, height)).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": vitpose,
        "K_fullimg": K,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": feats,
    }
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()

    # raw (no post-processing) and post-processed translation
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    batch = {
        "length": data["length"][None],
        "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
        "bbx_xys": data["bbx_xys"][None],
        "K_fullimg": data["K_fullimg"][None],
        "cam_angvel": data["cam_angvel"][None],
        "f_imgseq": data["f_imgseq"][None],
    }
    batch = {k: v.cuda() for k, v in batch.items()}
    raw = detach_to_cpu(model.pipeline.forward(batch, train=False, postproc=False, static_cam=static_cam))
    post = detach_to_cpu(model.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam))
    g = post["pred_smpl_params_global"]
    g_raw = raw["pred_smpl_params_global"]

    # FK on the neutral skeleton (same convention as GVHMR's EnDecoder.fk_v2)
    endecoder = model.pipeline.endecoder.cpu()
    joints = endecoder.fk_v2(g["body_pose"], g["betas"], g["global_orient"], g["transl"])[0]

    kp_conf = vitpose[..., 2].float()  # (T, 17)
    joint_conf = torch.stack([kp_conf[:, COCO_FOR_JOINT[j]].mean(-1) for j in range(22)], dim=-1)

    out = {
        "fps": 30.0,
        "source_fps": src_fps,
        "body_pose": g["body_pose"][0].reshape(-1, 21, 3).numpy(),
        "global_orient": g["global_orient"][0].numpy(),
        "transl": g["transl"][0].numpy(),
        "transl_raw": g_raw["transl"][0].numpy(),
        "body_pose_raw": g_raw["body_pose"][0].reshape(-1, 21, 3).numpy(),
        "betas": g["betas"][0].numpy(),
        "joints": joints.numpy(),
        "rest_joints": endecoder.smplx_model.rest.numpy(),
        "static_conf": torch.sigmoid(post["static_conf_logits"][0].float()).numpy(),
        "kp2d": vitpose.numpy(),
        "kp2d_conf": kp_conf.numpy(),
        "joint_conf": joint_conf.numpy(),
        "bbx_xyxy": bbx["bbx_xyxy"].numpy(),
    }
    out_path = out_dir / f"{name}.npz"
    np.savez(out_path, **out)
    print(f"[extract] saved {out_path}: {length} frames, mean 2D conf {kp_conf.mean():.2f}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--moving_cam", action="store_true", help="estimate camera rotation (SimpleVO)")
    ap.add_argument("--f_mm", type=int, default=None)
    args = ap.parse_args()
    video = Path(args.video).resolve()
    extract(video, ROOT / "extracted", static_cam=not args.moving_cam, f_mm=args.f_mm)


if __name__ == "__main__":
    main()
