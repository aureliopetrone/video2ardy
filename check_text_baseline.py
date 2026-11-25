"""Check of the README's claims: is "text only" really unable to perform the scene, or is the
frame-by-frame error just a timing mismatch? Short motion-style prompts, 5 seeds, and a
timing-tolerant error (subsequence DTW on heading-free, root-relative poses)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ardy_video_projector as P
import numpy as np
import run_video_test as R
import torch

SHORT = {
    "pirate": "A tired person leans forward with both hands on a railing, then raises one hand to shade the eyes and looks slowly from left to right.",
    "noble": "A person lifts the chin, waves one hand dismissively, puts the other hand on the hip and turns away.",
    "thief": "A nervous person glances over the left shoulder and then the right, shifts weight from foot to foot, then crouches down and looks around.",
    "captain": "A person points forward with the right arm, then stamps a foot and waves both arms forward.",
    "drunk": "A drunk person staggers, sways from side to side, almost falls forward and catches balance with arms spread wide.",
}
m = P.load()
sk = m.skeleton
bi = sk.bone_index
EE = [bi["LeftHand"], bi["RightHand"], bi["Head"]]


def dtw(X, Y, joints=None):
    """Subsequence DTW: all of Y (reference) matched to a free stretch of X. Mean cost per step, cm."""
    cx, cy = R.canon(X, sk), R.canon(Y, sk)
    if joints is not None:
        cx, cy = cx[:, joints], cy[:, joints]
    C = (cx[:, None] - cy[None]).norm(dim=-1).mean(-1).cpu().numpy()  # [Tx, Ty]
    Tx, Ty = C.shape
    D = np.full((Tx + 1, Ty + 1), np.inf)
    L = np.zeros((Tx + 1, Ty + 1))
    D[:, 0] = 0.0  # free start on X
    for j in range(1, Ty + 1):
        for i in range(1, Tx + 1):
            opts = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(opts))
            pi, pj = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            D[i, j] = opts[k] + C[i - 1, j - 1]
            L[i, j] = L[pi, pj] + 1
    i = int(np.argmin(D[1:, Ty] / L[1:, Ty])) + 1  # free end on X
    return 100 * D[i, Ty] / L[i, Ty]


def stats(J):
    hands = J[:, [bi["LeftHand"], bi["RightHand"]]]
    head = J[:, bi["Head"]]
    return {
        "hand_above_head_max_cm": 100 * float((hands[..., 1] - head[:, None, 1]).max()),
        "hand_to_head_min_cm": 100 * float((hands - head[:, None]).norm(dim=-1).min()),
        "hips_min_cm": 100 * float(J[:, 0, 1].min()),
        "hand_reach_max_cm": 100 * float((hands - J[:, :1]).norm(dim=-1).max()),
    }


out = {}
for name, prompt in SHORT.items():
    b = np.load(ROOT / "results" / name / "bundle.npz", allow_pickle=True)
    labels = [str(x) for x in b["labels"]]
    ref = torch.tensor(b["joints_0"]).cuda()
    T = ref.shape[0]
    rows = {"reference": stats(ref)}
    for i, lab in enumerate(labels[1:], 1):
        if lab.startswith(("ARDY text", "Mode A", "Mode B s=0.5 +", "Mode B s=0.5 sparse")):
            J = torch.tensor(b[f"joints_{i}"]).cuda()
            rows[lab] = {"dtw_body_cm": dtw(J, ref), "dtw_hands_head_cm": dtw(J, ref, EE), **stats(J[32:32 + T] if len(J) > T else J)}
    seeds = []
    for seed in range(5):
        f = P.generate(m, prompt, 32 + T + 32 + (-(32 + T + 32) % 4), critic=False, seed=seed)
        _, _, J = R.decode(m, f)
        seeds.append({"dtw_body_cm": dtw(J, ref), "dtw_hands_head_cm": dtw(J, ref, EE), **stats(J)})
    rows["text short prompt (5 seeds: median)"] = {k: float(np.median([s[k] for s in seeds])) for k in seeds[0]}
    rows["text short prompt (5 seeds: best)"] = {
        "dtw_body_cm": min(s["dtw_body_cm"] for s in seeds), "dtw_hands_head_cm": min(s["dtw_hands_head_cm"] for s in seeds),
        "hand_above_head_max_cm": max(s["hand_above_head_max_cm"] for s in seeds),
        "hand_to_head_min_cm": min(s["hand_to_head_min_cm"] for s in seeds),
        "hips_min_cm": min(s["hips_min_cm"] for s in seeds), "hand_reach_max_cm": max(s["hand_reach_max_cm"] for s in seeds)}
    out[name] = rows
    print("==", name)
    for k, v in rows.items():
        print(f"  {k:38s}", " ".join(f"{kk}={vv:6.1f}" for kk, vv in v.items()))
json.dump(out, open(ROOT / "results" / "text_baseline_check.json", "w"), indent=1)
