"""Side-by-side comparison video: the reference clip next to stick-figure renders of each condition.

Run with venv_gvhmr (has OpenCV):
    venv_gvhmr\\Scripts\\python.exe render_compare.py results\\<name>\\bundle.npz results\\<name>\\compare.mp4

bundle.npz: video (path, 30 fps), parents (J,), labels (N,), joints_<i> (T, J, 3) y-up metres,
            optional captions_<i> (str), offset_<i> (first frame of the segment in joints_<i>).
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
FFMPEG = str(ROOT / "bin" / "ffmpeg.exe")
PANEL = 480

# SOMA30 sides by joint name prefix
LEFT, RIGHT, MID = (235, 140, 60), (60, 140, 235), (200, 200, 200)  # BGR


def look_at(eye, target):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f]), eye


def project(P, R, eye, focal, size):
    c = (P - eye) @ R.T  # camera coords, looking down -z
    z = -c[..., 2]
    x = focal * c[..., 0] / np.maximum(z, 1e-3) + size / 2
    y = -focal * c[..., 1] / np.maximum(z, 1e-3) + size * 0.62
    return np.stack([x, y], -1), z


def draw_panel(joints_t, parents, colors, cam, label, caption, trail=None):
    R, eye, focal = cam
    img = np.full((PANEL, PANEL, 3), 28, np.uint8)
    # ground grid
    g = np.arange(-4, 4.01, 0.5)
    for a in g:
        for p0, p1 in (([a, 0, -4], [a, 0, 4]), ([-4, 0, a], [4, 0, a])):
            pts = np.linspace(p0, p1, 24)
            uv, z = project(pts, R, eye, focal, PANEL)
            ok = z > 0.1
            uv = uv[ok].astype(np.int32)
            if len(uv) > 1:
                cv2.polylines(img, [uv], False, (58, 58, 58), 1, cv2.LINE_AA)
    # shadow
    sh = joints_t.copy()
    sh[:, 1] = 0.0
    uv_s, _ = project(sh, R, eye, focal, PANEL)
    for j, p in enumerate(parents):
        if p >= 0:
            cv2.line(img, tuple(uv_s[j].astype(int)), tuple(uv_s[p].astype(int)), (16, 16, 16), 9, cv2.LINE_AA)
    if trail is not None and len(trail) > 1:
        tr = trail.copy()
        tr[:, 1] = 0.0
        uv_t, _ = project(tr, R, eye, focal, PANEL)
        cv2.polylines(img, [uv_t.astype(np.int32)], False, (90, 90, 140), 1, cv2.LINE_AA)
    uv, z = project(joints_t, R, eye, focal, PANEL)
    bones = [(j, p) for j, p in enumerate(parents) if p >= 0]
    bones.sort(key=lambda b: -(z[b[0]] + z[b[1]]))  # far first
    for j, p in bones:
        depth = 0.5 * (z[j] + z[p])
        th = int(np.clip(70 / depth, 4, 14))
        col = tuple(int(c * np.clip(1.25 - 0.08 * depth, 0.55, 1.0)) for c in colors[j])
        cv2.line(img, tuple(uv[j].astype(int)), tuple(uv[p].astype(int)), col, th, cv2.LINE_AA)
    hd = uv[6].astype(int)
    cv2.circle(img, tuple(hd), int(np.clip(110 / z[6], 6, 18)), colors[6], -1, cv2.LINE_AA)
    cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    for k, line in enumerate(caption.split("\n")):
        cv2.putText(img, line, (12, PANEL - 14 - 20 * (len(caption.split("\n")) - 1 - k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)
    return img


def read_video(path, n):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main(bundle_path, out_path):
    b = np.load(bundle_path, allow_pickle=True)
    parents = b["parents"].astype(int)
    names = [str(x) for x in b["joint_names"]]
    colors = [LEFT if n.startswith("Left") else RIGHT if n.startswith("Right") else MID for n in names]
    labels = [str(x) for x in b["labels"]]
    N = len(labels)
    segs = []
    for i in range(N):
        J = b[f"joints_{i}"]
        off = int(b[f"offset_{i}"]) if f"offset_{i}" in b else 0
        segs.append(J[off:])
    T = min(len(s) for s in segs)
    caps = [str(b[f"caption_{i}"]) if f"caption_{i}" in b else "" for i in range(N)]

    allroot = np.concatenate([s[:T, 0] for s in segs])
    center = np.array([np.median(allroot[:, 0]), 0.9, np.median(allroot[:, 2])])
    extent = max(1.2, np.percentile(np.linalg.norm(allroot[:, [0, 2]] - center[[0, 2]], axis=1), 95) + 1.0)
    az = np.deg2rad(25)
    dist = 2.6 * extent
    eye = center + np.array([np.sin(az) * dist, 0.55 * extent + 0.4, np.cos(az) * dist])
    R, eye = look_at(eye, center + np.array([0, -0.15, 0]))
    cam = (R, eye, PANEL * 1.15)

    vid = read_video(b["video"], T)
    cols = 3
    rows = int(np.ceil((N + 1) / cols))
    tmp = Path(out_path).with_suffix(".raw.avi")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), 30, (cols * PANEL, rows * PANEL))
    for t in range(T):
        tiles = []
        if t < len(vid):
            f = vid[t]
            h, w = f.shape[:2]
            s = PANEL / max(h, w)
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            tile = np.zeros((PANEL, PANEL, 3), np.uint8)
            y0, x0 = (PANEL - f.shape[0]) // 2, (PANEL - f.shape[1]) // 2
            tile[y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = f
        else:
            tile = np.zeros((PANEL, PANEL, 3), np.uint8)
        cv2.putText(tile, "Reference video", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(tile, f"{t / 30:5.2f} s", (PANEL - 100, PANEL - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(tile)
        for i in range(N):
            tiles.append(draw_panel(segs[i][t], parents, colors, cam, labels[i], caps[i], trail=segs[i][: t + 1, 0]))
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((PANEL, PANEL, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        wr.write(grid)
    wr.release()
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(out_path)], check=True)
    tmp.unlink()
    print(f"wrote {out_path} ({T} frames)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
