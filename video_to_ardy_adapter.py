"""Module 2 of the video -> ARDY pipeline: SMPL-X (GVHMR) -> ARDY SOMA30 at 30 fps, plus sparse
expressive key-constraints and the reference latent tokens for SDEdit.

Runs in ARDY's venv (Tools\\ardy\\venv) without installing anything there.

Retargeting. SMPL-X and SOMA30 share the rest convention: T-pose, y up, facing +Z, every joint
frame aligned with the world at rest (checked on the SOMA30 rest skeleton: arms along +-X, identity
rotations). So the global rotation of each SOMA joint is copied from the SMPL-X joint that plays
the same role, and SOMA local rotations follow from the SOMA hierarchy. Joints SMPL-X lacks (second
neck, jaw, eyes, hand ends) inherit their parent's global rotation (local identity). The root path
is the SMPL-X pelvis scaled by the leg-length ratio, canonicalized to start at the origin facing
+Z, with the ground put where ARDY's feet stand.

Constraints (ARDY's native API, ardy.constraints):
  * Root2DConstraintSet: smoothed planar root path + heading every ``root_every`` frames
    (depth jitter of monocular video is filtered by a Gaussian on the path, sigma in frames).
  * Left/RightHandConstraintSet: hand position + rotation at the salient frames and between them
    (acting gestures), skipped where the 2D detector is not confident about the wrist.
  * FullBodyConstraintSet: full-body anchors at adaptive intervals (K = 10..25 frames): denser where
    the "acting signature" (hand/head acceleration, postural asymmetry) is high, sparser in
    transitions, never on low-confidence frames.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ARDY_ROOT = Path(os.environ.get("ARDY_ROOT", Path(__file__).resolve().parent.parent / "ardy"))
CRITIC_ROOT = Path(os.environ.get("ARDY_CRITIC_ROOT", Path(__file__).resolve().parent.parent / "ardy_critic"))
for p in (ARDY_ROOT, CRITIC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ardy.geometry import axis_angle_to_matrix  # noqa: E402
from ardy.motion_rep.tools import compute_heading_angle  # noqa: E402

# SOMA30 joint -> SMPL-X joint whose global rotation it takes (None = same as its SOMA parent)
SOMA_FROM_SMPLX = {
    "Hips": 0, "Spine1": 3, "Spine2": 6, "Chest": 9, "Neck1": 12, "Neck2": None, "Head": 15,
    "Jaw": None, "LeftEye": None, "RightEye": None,
    "LeftShoulder": 13, "LeftArm": 16, "LeftForeArm": 18, "LeftHand": 20,
    "LeftHandThumbEnd": None, "LeftHandMiddleEnd": None,
    "RightShoulder": 14, "RightArm": 17, "RightForeArm": 19, "RightHand": 21,
    "RightHandThumbEnd": None, "RightHandMiddleEnd": None,
    "LeftLeg": 1, "LeftShin": 4, "LeftFoot": 7, "LeftToeBase": 10,
    "RightLeg": 2, "RightShin": 5, "RightFoot": 8, "RightToeBase": 11,
}
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPLX_L_WRIST, SMPLX_R_WRIST, SMPLX_HEAD = 20, 21, 15


@dataclass
class AdapterConfig:
    root_every: int = 3            # Root2D constraint spacing (frames)
    root_sigma: float = 3.0        # Gaussian smoothing of the planar root path (frames)
    k_min: int = 10                # full-body anchor spacing bounds (frames)
    k_max: int = 25
    hand_every: int = 5            # hand end-effector spacing between anchors (frames)
    conf_min: float = 0.45         # 2D confidence under which a frame/joint is not used as a constraint
    ground_toe_y: float = 0.017    # toe-joint height on the floor in ARDY's own output (measured: median 1.7 cm)


@dataclass
class Reference:
    """A reference motion on ARDY's skeleton, plus everything derived from the video."""

    name: str
    fps: float
    local_rots: torch.Tensor      # [T, J, 3, 3] SOMA30 local rotations
    root_pos: torch.Tensor        # [T, 3] hips position, y up, ground at 0
    joints: torch.Tensor          # [T, J, 3] FK
    conf: torch.Tensor            # [T] frame confidence (mean 2D keypoint confidence)
    hand_conf: torch.Tensor       # [T, 2] left/right wrist confidence
    contact: torch.Tensor         # [T, 4] GVHMR static prob [L_ankle, L_foot, R_ankle, R_foot]
    salience: torch.Tensor = None # [T] acting-signature score
    anchors: list = field(default_factory=list)
    reliable: torch.Tensor = None # [T] bool: frame usable as a constraint / SDEdit source


def _smplx_global_rots(global_orient: np.ndarray, body_pose: np.ndarray) -> torch.Tensor:
    aa = torch.cat([torch.as_tensor(global_orient)[:, None], torch.as_tensor(body_pose)], dim=1).float()
    R = axis_angle_to_matrix(aa)  # [T, 22, 3, 3]
    G = torch.empty_like(R)
    for j, p in enumerate(SMPLX_PARENTS):
        G[:, j] = R[:, j] if p < 0 else G[:, p] @ R[:, j]
    return G


def _gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian smoothing along dim 0 with edge padding."""
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    flat = x.reshape(x.shape[0], -1).T[:, None]  # [C, 1, T]
    pad = torch.nn.functional.pad(flat, (r, r), mode="replicate")
    out = torch.nn.functional.conv1d(pad, k[None, None])
    return out[:, 0].T.reshape(x.shape)


def retarget(npz_path: Path, skeleton, cfg: AdapterConfig = AdapterConfig(), use_postproc: bool = True,
             rest_local: torch.Tensor | None = None) -> Reference:
    """``rest_local`` [J, 3, 3]: typical local rotation per joint in ARDY's data (the critic's
    ``rot_mean``). Joints SMPL-X lacks (second neck, jaw, eyes, finger ends) take it, instead of
    identity: ARDY's relaxed hands are not straight, and identity there reads as an odd pose."""
    d = np.load(npz_path)
    fps = float(d["fps"])
    T = d["transl"].shape[0]
    body_pose = d["body_pose"] if use_postproc else d["body_pose_raw"]
    transl = d["transl"] if use_postproc else d["transl_raw"]
    G = _smplx_global_rots(d["global_orient"], body_pose)  # [T, 22, 3, 3]

    names = skeleton.bone_order_names
    parents = skeleton.joint_parents.tolist()
    J = len(names)
    Gs = torch.empty(T, J, 3, 3)
    for j, n in enumerate(names):
        src = SOMA_FROM_SMPLX[n]
        Gs[:, j] = G[:, src] if src is not None else Gs[:, parents[j]]
    if rest_local is not None:
        rest_local = rest_local.detach().float().cpu()
        for j, n in enumerate(names):  # parents come first in SOMA order
            if SOMA_FROM_SMPLX[n] is None:
                Gs[:, j] = Gs[:, parents[j]] @ rest_local[j]
    Ls = torch.empty_like(Gs)
    for j, p in enumerate(parents):
        Ls[:, j] = Gs[:, j] if p < 0 else Gs[:, p].transpose(-1, -2) @ Gs[:, j]

    # root path: SMPL-X pelvis = rest pelvis + transl, scaled by the leg ratio
    rest = torch.as_tensor(d["rest_joints"]).float()
    smplx_leg = float(rest[0, 1] - rest[7:9, 1].mean())
    nj = skeleton.neutral_joints.cpu()
    i_lf, i_rf = skeleton.bone_index["LeftFoot"], skeleton.bone_index["RightFoot"]
    soma_leg = float(nj[0, 1] - nj[[i_lf, i_rf], 1].mean())
    scale = soma_leg / smplx_leg
    pelvis = torch.as_tensor(transl).float() + rest[0]
    root = pelvis * scale

    # canonicalize: first frame at the origin, facing +Z
    _, joints0, _ = skeleton.fk(Ls[None].to(skeleton.device), root[None].to(skeleton.device))
    joints0 = joints0[0].cpu()
    h0 = float(compute_heading_angle(joints0[None, :1], skeleton)[0, 0])
    c, s = np.cos(-h0), np.sin(-h0)
    Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)  # rotation about y by -h0
    root = (root - root[:1] * torch.tensor([1.0, 0, 1.0])) @ Ry.T
    Ls[:, 0] = Ry @ Ls[:, 0]

    _, joints, _ = skeleton.fk(Ls[None].to(skeleton.device), root[None].to(skeleton.device))
    joints = joints[0].cpu()
    h_check = float(compute_heading_angle(joints[None, :1], skeleton)[0, 0])
    assert abs(h_check) < 1e-3, f"canonical heading not zero ({h_check})"

    # ground: toes of the supporting foot at ARDY's toe height
    fidx = list(skeleton.foot_joint_idx)  # [LFoot, LToe, RFoot, RToe]
    toe_y = joints[:, [fidx[1], fidx[3]], 1]  # [T, 2]
    contact = torch.as_tensor(d["static_conf"][:, :4]).float()
    support = torch.minimum(toe_y[:, 0], toe_y[:, 1])
    in_contact = (contact.max(-1).values > 0.5)
    floor = torch.quantile(support[in_contact] if in_contact.sum() > 5 else support, 0.2)
    target = cfg.ground_toe_y
    root = root + torch.tensor([0.0, float(target - floor), 0.0])
    joints = joints + torch.tensor([0.0, float(target - floor), 0.0])

    kp = torch.as_tensor(d["kp2d_conf"]).float()  # COCO17
    conf = kp.mean(-1)
    hand_conf = kp[:, [9, 10]]  # left/right wrist
    ref = Reference(npz_path.stem, fps, Ls, root, joints, conf, hand_conf, contact)
    ref.salience = acting_salience(ref, skeleton)
    return ref


def acting_salience(ref: Reference, skeleton) -> torch.Tensor:
    """Per-frame "acting signature": hand/head acceleration relative to the root, plus postural
    asymmetry (left vs right hand height and reach), weighted by detector confidence."""
    j = ref.joints
    names = skeleton.bone_index
    ids = [names["LeftHand"], names["RightHand"], names["Head"]]
    rel = j[:, ids] - j[:, :1]  # root-relative
    rel = _gaussian_smooth(rel, 1.0)
    vel = torch.zeros_like(rel)
    vel[1:-1] = (rel[2:] - rel[:-2]) * ref.fps / 2
    acc = torch.zeros_like(rel)
    acc[1:-1] = (rel[2:] - 2 * rel[1:-1] + rel[:-2]) * ref.fps**2
    acc_n = acc.norm(dim=-1).sum(-1)
    asym = (rel[:, 0, 1] - rel[:, 1, 1]).abs() + ((rel[:, 0] - rel[:, 1]).norm(dim=-1) - 0.6).abs()
    speed = vel.norm(dim=-1).sum(-1)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    # acting beats are where a fast gesture comes to rest (speed minimum after acceleration peak)
    sal = z(acc_n) + 0.5 * z(asym) - 0.3 * z(speed)
    return _gaussian_smooth(sal, 1.5) * ref.conf.clamp(0, 1)


def mark_unreliable(ref: Reference, critic, cfg: AdapterConfig = AdapterConfig(), e_max: float = 300.0,
                    ndf_max: float = 9.0, dilate: int = 4) -> torch.Tensor:
    """Frames the video estimate cannot be trusted on: low 2D confidence, or a body no person can
    make (the critic's smoothed energy above ``e_max``, or its motion prior above ``ndf_max``), e.g.
    a T2V clip that turns the actor upside down for a second. Measured: legitimate clips stay under
    ~260 (tennis p99) and a prior <= 7.2 (deep crouch of the thief), the hallucinated stretch of the
    drunk clip reaches energies of 400-18000 and a prior up to 13.6.
    Sets and returns ``ref.reliable``; the mask is dilated by ``dilate`` frames."""
    dev = critic.rot_mean.device
    e = critic.evaluate(ref.local_rots[None].to(dev), ref.root_pos[None].to(dev))
    tot = critic.smooth(e["total"])[0].cpu()
    prior = e["prior"][0].cpu()
    bad = (tot > e_max) | (prior > ndf_max) | (ref.conf < cfg.conf_min)
    if dilate > 0:
        bad = torch.nn.functional.max_pool1d(bad.float()[None, None], 2 * dilate + 1, 1, dilate)[0, 0] > 0
    ref.reliable = ~bad
    return ref.reliable


def _reliable(ref: Reference) -> torch.Tensor:
    T = ref.joints.shape[0]
    return ref.reliable if ref.reliable is not None else torch.ones(T, dtype=torch.bool)


def select_anchors(ref: Reference, cfg: AdapterConfig) -> list[int]:
    """Adaptive full-body anchors: first/last reliable frame, the reliable frames that border an
    unreliable stretch (so ARDY in-fills between trusted poses), salience peaks at least k_min apart,
    gaps filled so no two anchors are more than k_max apart; unreliable frames are never used."""
    T = ref.joints.shape[0]
    sal = ref.salience.clone()
    ok = (ref.conf >= cfg.conf_min) & _reliable(ref)
    sal[~ok] = -1e9
    good = [t for t in range(T) if bool(ok[t])]
    if not good:
        return []
    peaks = [t for t in range(1, T - 1) if sal[t] >= sal[t - 1] and sal[t] >= sal[t + 1] and sal[t] > 0]
    peaks.sort(key=lambda t: -float(sal[t]))
    chosen = sorted({good[0], good[-1]} | {t for t in good if (t > 0 and not ok[t - 1]) or (t < T - 1 and not ok[t + 1])})
    for t in peaks:
        if all(abs(t - c) >= cfg.k_min for c in chosen):
            chosen.append(t)
    chosen.sort()
    # fill long gaps at the most salient admissible frame
    changed = True
    while changed:
        changed = False
        for a, b in zip(chosen[:-1], chosen[1:]):
            if b - a > cfg.k_max and bool(ok[a + 1:b].any()):
                lo, hi = a + cfg.k_min // 2, b - cfg.k_min // 2
                cand = sal[lo:hi + 1]
                if hi < lo or float(cand.max()) <= -1e8:
                    continue  # the gap is an unreliable stretch: ARDY fills it
                t = lo + int(torch.argmax(cand))
                chosen.append(t)
                chosen.sort()
                changed = True
                break
    return chosen


def build_constraints(ref: Reference, skeleton, cfg: AdapterConfig = AdapterConfig(), kinds=("root", "hands", "fullbody")):
    from ardy.constraints import (
        FullBodyConstraintSet,
        LeftHandConstraintSet,
        RightHandConstraintSet,
        Root2DConstraintSet,
    )

    dev = skeleton.device
    T = ref.joints.shape[0]
    Gr, joints, _ = skeleton.fk(ref.local_rots[None].to(dev), ref.root_pos[None].to(dev))
    Gr, joints = Gr[0], joints[0]
    out = []
    anchors = select_anchors(ref, cfg)
    ref.anchors = anchors

    if "root" in kinds:
        root2d = _gaussian_smooth(ref.root_pos[:, [0, 2]], cfg.root_sigma).to(dev)
        heading = compute_heading_angle(joints[None], skeleton)[0]
        hs = torch.atan2(_gaussian_smooth(torch.sin(heading).cpu()[:, None], 2.0)[:, 0],
                         _gaussian_smooth(torch.cos(heading).cpu()[:, None], 2.0)[:, 0]).to(dev)
        rel = _reliable(ref)
        idx = torch.tensor([t for t in range(0, T, cfg.root_every) if bool(rel[t])], dtype=torch.long)
        out.append(Root2DConstraintSet(skeleton, idx, root2d[idx], hs[idx]))

    if "fullbody" in kinds:
        idx = torch.tensor(anchors)
        out.append(FullBodyConstraintSet(skeleton, idx, joints[idx], Gr[idx]))

    if "hands" in kinds:
        frames = set()
        for a, b in zip(anchors[:-1], anchors[1:]):
            frames.update(range(a, b, cfg.hand_every))
        frames.update(anchors)
        frames = sorted(f for f in frames if f < T)
        for k, cls in ((0, LeftHandConstraintSet), (1, RightHandConstraintSet)):
            rel = _reliable(ref)
            f = [t for t in frames if float(ref.hand_conf[t, k]) >= cfg.conf_min and bool(rel[t])]
            if f:
                idx = torch.tensor(f)
                out.append(cls(skeleton, idx, joints[idx], Gr[idx], None))
    return out


def to_features(model, ref: Reference) -> torch.Tensor:
    """Normalized ARDY features [1, T, D] of the reference."""
    dev = model.device
    return model.motion_rep(ref.local_rots[None].to(dev), ref.root_pos[None].to(dev), to_normalize=True)


def encode_reference(model, feats: torch.Tensor) -> torch.Tensor:
    """Reference hybrid tokens z_ref (explicit root + latent body), the space ARDY diffuses in.
    The clip is padded to a whole number of tokens by repeating its last frame."""
    nfpt = model.num_frames_per_token
    T = feats.shape[1]
    Tp = -(-T // nfpt) * nfpt
    if Tp > T:
        feats = torch.cat([feats, feats[:, -1:].expand(-1, Tp - T, -1)], dim=1)
    pad = torch.ones(1, Tp, dtype=torch.bool, device=feats.device)
    with torch.no_grad():
        z, _ = model.hybrid.get_hybrid_motion_from_explicit(
            motion=feats, motion_len=torch.tensor([Tp], device=feats.device), motion_pad_mask=pad
        )
    return z


def critic_clean(ref: Reference, critic, skeleton, steps: int = 150, knot: int = 4, lr_rot: float = 0.01,
                 lr_root: float = 0.004, w_fid_rot: float = 2.0, w_fid_root: float = 20.0, w_ee: float = 30.0,
                 w_plant: float = 1000.0,
                 log=None) -> Reference:
    """Whole-clip cleanup of the video reference with the spatio-temporal critic as the energy
    (NDF prior + joint-range/jerk barriers + foot skate/penetration, GVHMR's contact probabilities
    as the contact hint), staying close to the video: fidelity on every rotation change and on the
    root, stronger on hands and head (the acting). The correction lives on knots every ``knot``
    frames, linearly interpolated, so the optimizer cannot add frame-to-frame jitter; the iterate
    with the lowest objective is kept. Constraints and z_ref are then taken from the clean copy."""
    from ardy.geometry import cont6d_to_matrix, matrix_to_cont6d

    dev = skeleton.device
    lr0 = ref.local_rots[None].to(dev)
    root0 = ref.root_pos[None].to(dev)
    T = lr0.shape[1]
    fc = (ref.contact[None].to(dev) > 0.5)
    base6 = matrix_to_cont6d(lr0)
    nk = (T - 1) // knot + 2
    k6 = torch.zeros(1, nk, *base6.shape[2:], device=dev, requires_grad=True)
    kr = torch.zeros(1, nk, 3, device=dev, requires_grad=True)
    pos = torch.arange(T, device=dev, dtype=torch.float32) / knot
    i0 = pos.floor().long()
    w = (pos - i0)[None, :]

    sig = knot / 2.0
    r = int(3 * sig)
    gk = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=dev, dtype=torch.float32) / sig) ** 2)
    gk = (gk / gk.sum())[None, None]

    def up(k):
        a, b = k[:, i0], k[:, i0 + 1]
        shape = (1, T) + (1,) * (k.dim() - 2)
        x = a + (b - a) * w.reshape(shape)  # piecewise linear ...
        flat = x.reshape(T, -1).T[:, None]
        flat = torch.nn.functional.conv1d(torch.nn.functional.pad(flat, (r, r), mode="replicate"), gk)
        return flat[:, 0].T.reshape(x.shape)  # ... then Gaussian: no kinks at the knots

    opt = torch.optim.Adam([{"params": [k6], "lr": lr_rot}, {"params": [kr], "lr": lr_root}])
    bi = skeleton.bone_index
    ee = [bi["LeftHand"], bi["RightHand"], bi["Head"]]
    fidx = list(skeleton.foot_joint_idx)
    with torch.no_grad():
        j0 = skeleton.fk(lr0, root0)[1]
        rel0 = j0[:, :, ee] - j0[:, :, :1]
        # planted feet of the estimate: GVHMR says contact and the foot barely moves there
        fv = torch.zeros_like(j0[:, :, fidx, 0])
        fv[:, 1:] = (j0[:, 1:, fidx][..., [0, 2]] - j0[:, :-1, fidx][..., [0, 2]]).norm(dim=-1) * ref.fps
        plant = ((ref.contact[None].to(dev) > 0.5) & (fv < 0.3)).float()  # [1, T, 4]
    best = (float("inf"), None, None)
    hist = []
    with torch.enable_grad():
        for it in range(steps + 1):
            d6, dr = up(k6), up(kr)
            lr = cont6d_to_matrix(base6 + d6)
            root = root0 + dr
            e = critic.evaluate(lr, root, foot_contacts=fc)
            tot = e["total"]
            j = skeleton.fk(lr, root)[1]
            rel = j[:, :, ee] - j[:, :, :1]
            fid = w_fid_rot * d6.pow(2).mean() + w_fid_root * dr.pow(2).mean() + w_ee * (rel - rel0).pow(2).sum(-1).mean()
            # a planted foot keeps its place on the floor (the cleanup must not add sliding)
            fid = fid + w_plant * ((j[:, :, fidx][..., [0, 2]] - j0[:, :, fidx][..., [0, 2]]).pow(2).sum(-1) * plant).mean()
            loss = tot.mean() + fid
            if float(loss.detach()) < best[0]:
                best = (float(loss), k6.detach().clone(), kr.detach().clone())
            if it in (0, steps // 2, steps):
                hist.append({"it": it, "loss": round(float(loss), 3), **{k: round(float(v.mean()), 3) for k, v in e.items() if k != "kin"}})
            if it == steps:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        lr = cont6d_to_matrix(base6 + up(best[1]))[0].cpu()
        root = (root0 + up(best[2]))[0].cpu()
        joints = skeleton.fk(lr[None].to(dev), root[None].to(dev))[1][0].cpu()
    out = Reference(ref.name + "_clean", ref.fps, lr, root, joints, ref.conf, ref.hand_conf, ref.contact)
    out.salience = ref.salience
    out.reliable = ref.reliable
    out.clean_log = hist
    return out
