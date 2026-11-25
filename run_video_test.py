"""Reproducible benchmark: text-only ARDY vs direct video retarget vs video->ARDY projection.

Runs in ARDY's venv:
    ..\\ardy\\venv\\Scripts\\python.exe run_video_test.py --clips pirate thief noble [--strengths 0.3 0.5 0.8]

For each clip (extracted/<name>.npz from video_extractor.py, prompt from clips/<name>.txt or
PROMPTS below) it writes results/<name>/:
    bundle.npz       joints of every condition, for render_compare.py
    metrics.json     all metrics, every condition
and results/summary.md with one table per clip.

Conditions (every ARDY generation has a free 32-frame lead-in and tail around the video segment):
    reference   GVHMR retargeted to SOMA30, no diffusion ("direct video retarget")
    baseline    vanilla ARDY, rich text prompt only, no critic
    modeA       constraints only (empty text), with critic
    modeB_s     SDEdit from the reference tokens at strength s + constraints, with critic
    modeB_s_nc  the same without critic (ablation, metrics only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ardy_video_projector as P  # noqa: E402  (sets up ARDY + critic)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import video_to_ardy_adapter as A  # noqa: E402
from ardy.motion_rep.tools import compute_heading_angle  # noqa: E402

LEAD = 32
TAIL = 32
PROMPTS = {
    "tennis": "A person plays tennis, swinging a racket forehand and backhand.",
}


def clip_prompt(name: str) -> str:
    txt = ROOT / "clips" / f"{name}.txt"
    if txt.exists():
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.startswith("action="):
                return line[len("action="):].strip()
    return PROMPTS.get(name, "A person moves.")


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------
def canon(joints, skeleton):
    """Root-relative joints rotated so each frame faces +Z (heading removed)."""
    h = compute_heading_angle(joints[None], skeleton)[0]
    c, s = torch.cos(-h), torch.sin(-h)
    rel = joints - joints[:, :1] * torch.tensor([1.0, 0.0, 1.0], device=joints.device)
    x = c[:, None] * rel[..., 0] + s[:, None] * rel[..., 2]
    z = -s[:, None] * rel[..., 0] + c[:, None] * rel[..., 2]
    return torch.stack([x, rel[..., 1], z], -1)


def procrustes(X, Y):
    """Align X to Y per frame (rotation, translation, scale). X, Y: [T, J, 3]."""
    mx, my = X.mean(1, keepdim=True), Y.mean(1, keepdim=True)
    X0, Y0 = X - mx, Y - my
    H = X0.transpose(1, 2) @ Y0
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    Rm = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    s = (S * torch.diagonal(D, dim1=1, dim2=2)).sum(-1) / (X0 ** 2).sum((1, 2))
    return s[:, None, None] * X0 @ Rm.transpose(1, 2) + my


def angular_jerk(local_rots, fps):
    rel = local_rots[:-1].transpose(-1, -2) @ local_rots[1:]
    from ardy.geometry import matrix_to_axis_angle

    w = matrix_to_axis_angle(rel) * fps  # [T-1, J, 3]
    j = (w[2:] - 2 * w[1:-1] + w[:-2]) * fps ** 2
    return j.norm(dim=-1)  # [T-3, J]


def physical(model, critic, local_rots, root, joints):
    fps = float(model.motion_rep.fps)
    fi = list(model.skeleton.foot_joint_idx)
    toes = joints[:, [fi[1], fi[3]]]
    on = toes[:-1, :, 1] < 0.05
    v = (toes[1:] - toes[:-1]).norm(dim=-1) * fps
    e = critic.evaluate(local_rots[None], root[None])
    return {
        "foot_skate_cm_s": 100 * float((v * on).sum() / on.sum().clamp_min(1)),
        "penetration_frames": float((joints[..., 1] < -0.03).any(-1).float().mean()),
        "floating_frames": float((toes[..., 1].min(-1).values > 0.10).float().mean()),
        "ang_jerk": float(angular_jerk(local_rots, fps).mean()),
        "pos_jerk": float(((joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * fps ** 3).norm(dim=-1).mean()),
        "ndf": float(e["prior"].mean()),
        "bio_stress": float(e["bio"].mean()),
        "contact_energy": float(e["contact"].mean()),
        "critic_total_p95": float(torch.quantile(critic.smooth(e["total"])[0], 0.95)),
    }


def expressive(joints, ref_joints, skeleton, anchors, mask=None):
    """Fidelity to the video estimate, on the frames it can be trusted on (``mask``)."""
    bi = skeleton.bone_index
    if mask is not None:
        m = mask.to(joints.device)
        pos = torch.cumsum(m.long(), 0) - 1
        anchors = [int(pos[a]) for a in anchors if bool(m[a])]
        joints, ref_joints = joints[m], ref_joints[m]
    cj, cr = canon(joints, skeleton), canon(ref_joints, skeleton)
    ee = [bi["LeftHand"], bi["RightHand"], bi["Head"]]
    mirror = torch.tensor([-1.0, 1.0, 1.0], device=cj.device)

    def stats(c):
        var = float(c.var(0).sum(-1).mean())
        asym = float((c[:, bi["LeftHand"]] - c[:, bi["RightHand"]] * mirror).norm(dim=-1).mean())
        return var, asym

    var, asym = stats(cj)
    var_r, asym_r = stats(cr)
    mpjpe = (cj - cr).norm(dim=-1).mean(-1)
    pa = (procrustes(cj, cr) - cr).norm(dim=-1).mean(-1)
    ee_err = (cj[:, ee] - cr[:, ee]).norm(dim=-1).mean(-1)
    a = torch.tensor(anchors, device=cj.device)
    root_err = (joints[:, 0, [0, 2]] - ref_joints[:, 0, [0, 2]]).norm(dim=-1)
    return {
        "pose_var_ratio": var / max(var_r, 1e-6),
        "asym_ratio": asym / max(asym_r, 1e-6),
        "mpjpe_cm": 100 * float(mpjpe.mean()),
        "pa_mpjpe_cm": 100 * float(pa.mean()),
        "ee_traj_err_cm": 100 * float(ee_err.mean()),
        "anchor_mpjpe_cm": 100 * float(mpjpe[a].mean()),
        "root_path_err_cm": 100 * float(root_err.mean()),
    }


def continuity(joints_full, fps, lead, T):
    """Seam test inside an ARDY sequence: mean joint acceleration in the 5 frames around each
    boundary of the video segment over the 90th percentile of the whole sequence's acceleration
    (<= 1: the seam is no harsher than ordinary motion; a pasted clip shows a spike). Acceleration
    rather than jerk, because ARDY's decoder adds a small 4-frame (one token) flicker to the jerk of
    every output, text-only included, which would drown the seam."""
    p = joints_full
    acc = ((p[2:] - 2 * p[1:-1] + p[:-2]) * fps ** 2).norm(dim=-1).mean(-1)  # frame t+1
    ref = float(torch.quantile(acc, 0.9))
    out = {}
    for name, b in (("start", lead), ("end", lead + T)):
        idx = [t for t in range(b - 3, b + 2) if 0 <= t < len(acc)]
        out[f"seam_{name}"] = float(acc[idx].max()) / max(ref, 1e-6) if idx else float("nan")
    return out


# ------------------------------------------------------------------------------------------
POSTPROCESS = False  # ARDY's post_process_motion was tried: on tennis it doubled foot skate (12 -> 26 cm/s) and spiked bio stress, so it is off


def decode(model, feats, constraints=None):
    out = model.motion_rep.inverse(feats, is_normalized=True)
    if POSTPROCESS:
        from ardy.postprocess import post_process_motion

        out.update(post_process_motion(out["local_rot_mats"], out["root_positions"], out["foot_contacts"],
                                       model.skeleton, constraint_lst=constraints or None))
    return out["local_rot_mats"][0], out["root_positions"][0], out["posed_joints"][0]


def run_clip(model, critic, name, strengths, seed, out_root):
    sk = model.skeleton
    dev = model.device
    cfg = A.AdapterConfig()
    ref = A.retarget(ROOT / "extracted" / f"{name}.npz", sk, cfg, rest_local=critic.rot_mean)
    T = ref.joints.shape[0]
    # the critic cleans the video estimate first; constraints and z_ref come from the clean copy,
    # fidelity metrics are always measured against the raw video estimate
    rel = A.mark_unreliable(ref, critic, cfg)
    print(f"[{name}] unreliable frames: {int((~rel).sum())}/{len(rel)}")
    clean = A.critic_clean(ref, critic, sk)
    print(f"[{name}] critic cleanup: {clean.clean_log[0]} -> {clean.clean_log[-1]}")
    feats_ref = A.to_features(model, clean)
    z_ref = A.encode_reference(model, feats_ref)
    nfpt_ = model.num_frames_per_token
    rel_p = torch.cat([rel, rel[-1:].expand(z_ref.shape[1] * nfpt_ - len(rel))])
    tok_ok = rel_p.reshape(-1, nfpt_).all(-1)
    nfpt = model.num_frames_per_token
    total = LEAD + T + TAIL
    total = -(-total // nfpt) * nfpt
    text = clip_prompt(name)
    print(f"[{name}] {T} frames, {len(ref.anchors)} anchors {ref.anchors}, prompt: {text}")

    # reference re-decoded through ARDY's own representation (sanity check of the retarget)
    lr_ref, root_ref, j_ref = ref.local_rots.to(dev), ref.root_pos.to(dev), ref.joints.to(dev)
    j_ref_fk = sk.fk(lr_ref[None], root_ref[None])[1][0]

    runs = {}
    t0 = time.time()
    runs["baseline"] = P.generate(model, text, total, critic=False, seed=seed)
    shifted = lambda: P.shift_constraints(A.build_constraints(clean, sk, cfg), LEAD)  # noqa: E731
    A.build_constraints(clean, sk, cfg)  # sets clean.anchors
    ref.anchors = clean.anchors
    runs["modeA"] = P.generate(model, "", total, constraints=shifted(), critic=True, seed=seed)
    for s in strengths:
        runs[f"modeB_{s}"] = P.generate(model, "", total, constraints=shifted(), critic=True, z_ref=z_ref,
                                        strength=s, token_offset=LEAD // nfpt, seed=seed, tok_ok=tok_ok)
        runs[f"modeB_{s}_nc"] = P.generate(model, "", total, constraints=shifted(), critic=False, z_ref=z_ref,
                                           strength=s, token_offset=LEAD // nfpt, seed=seed, tok_ok=tok_ok)
    # sparse variant: root every 10 frames, hands only on the anchors, anchors 15..30 frames apart
    cfg_sp = A.AdapterConfig(root_every=10, hand_every=10_000, k_min=15, k_max=30)
    sparse = lambda: P.shift_constraints(A.build_constraints(clean, sk, cfg_sp), LEAD)  # noqa: E731
    shifted_sparse = sparse
    runs["modeA_sparse"] = P.generate(model, "", total, constraints=sparse(), critic=True, seed=seed)
    runs["modeB_0.5_sparse"] = P.generate(model, "", total, constraints=sparse(), critic=True, z_ref=z_ref,
                                          strength=0.5, token_offset=LEAD // nfpt, seed=seed, tok_ok=tok_ok)
    A.build_constraints(clean, sk, cfg)  # restore the dense anchors on clean/ref
    ref.anchors = clean.anchors
    print(f"[{name}] generations {time.time() - t0:.1f}s")

    fps = float(model.motion_rep.fps)
    metrics = {}
    seq = {}
    seg = slice(LEAD, LEAD + T)
    run_cons = {k: (None if k == "baseline" else shifted_sparse() if "sparse" in k else shifted()) for k in runs}
    lr_a, root_a, j_a = decode(model, runs["modeA"], run_cons["modeA"])
    stitched = torch.cat([j_a[:LEAD], j_ref_fk, j_a[LEAD + T:]], 0)
    metrics["reference"] = {
        **physical(model, critic, lr_ref, root_ref, j_ref_fk),
        **expressive(j_ref_fk, j_ref_fk, sk, ref.anchors, rel),
        **continuity(stitched, fps, LEAD, T),
    }
    seq["reference"] = (j_ref_fk, 0)
    j_clean = clean.joints.to(dev)
    metrics["reference_clean"] = {
        **physical(model, critic, clean.local_rots.to(dev), clean.root_pos.to(dev), j_clean),
        **expressive(j_clean, j_ref_fk, sk, ref.anchors, rel),
    }
    seq["reference_clean"] = (j_clean, 0)
    for k, f in runs.items():
        lr, root, j = decode(model, f, run_cons[k])
        m = physical(model, critic, lr[seg], root[seg], j[seg])
        m.update(expressive(j[seg], j_ref_fk, sk, ref.anchors, rel))
        if k != "baseline":
            m.update(continuity(j, fps, LEAD, T))
        metrics[k] = m
        seq[k] = (j, LEAD)

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"clip": name, "prompt": text, "frames": T, "anchors": ref.anchors,
                                                  "unreliable_frames": int((~rel).sum()),
                                                  "metrics": metrics}, indent=2), encoding="utf-8")
    shown = ["reference", "baseline", "modeA"] + [f"modeB_{s}" for s in strengths] + ["modeB_0.5_sparse", "reference_clean"]
    labels = {"reference": "Direct retarget (GVHMR)", "baseline": "ARDY text only", "modeA": "Mode A + critic",
              "reference_clean": "Retarget + critic cleanup only", "modeB_0.5_sparse": "Mode B s=0.5 sparse + critic"}
    for s in strengths:
        labels[f"modeB_{s}"] = f"Mode B s={s} + critic"
    bundle = {
        "video": str(ROOT / "extracted" / "work" / name / "input_30fps.mp4"),
        "parents": sk.joint_parents.cpu().numpy(),
        "joint_names": np.array(sk.bone_order_names),
        "labels": np.array([labels[k] for k in shown]),
    }
    for i, k in enumerate(shown):
        j, off = seq[k]
        m = metrics[k]
        bundle[f"joints_{i}"] = j.cpu().numpy()
        bundle[f"offset_{i}"] = off
        bundle[f"caption_{i}"] = (f"skate {m['foot_skate_cm_s']:.1f} cm/s  NDF {m['ndf']:.2f}\n"
                                  f"PA-MPJPE {m['pa_mpjpe_cm']:.1f} cm  hands {m['ee_traj_err_cm']:.1f} cm")
    np.savez(out / "bundle.npz", **bundle)
    return metrics, text, T


def table(name, text, T, metrics):
    cols = ["foot_skate_cm_s", "penetration_frames", "pos_jerk", "ang_jerk", "ndf", "bio_stress", "pa_mpjpe_cm", "ee_traj_err_cm",
            "anchor_mpjpe_cm", "root_path_err_cm", "pose_var_ratio", "asym_ratio", "seam_start", "seam_end"]
    lines = [f"### {name} ({T} frames) — baseline prompt: “{text}”", "",
             "| condition | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k, m in metrics.items():
        lines.append(f"| {k} | " + " | ".join(f"{m[c]:.2f}" if c in m else "–" for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.5])  # Mode A + Mode B 0.5 by default
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model = P.load()
    refiner = P.ardy_critic.patch._state(model)["refiner"]
    critic = refiner.critic
    # video constraints are dense (root path every few frames): let the critic's corrective keys
    # sit right next to them, otherwise it never finds a free frame
    refiner.cfg.user_guard = 0
    out_root = ROOT / "results"
    md = ["# Video -> ARDY: 3-way comparison", ""]
    for name in args.clips:
        metrics, text, T = run_clip(model, critic, name, args.strengths, args.seed, out_root)
        md.append(table(name, text, T, metrics))
    (out_root / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
