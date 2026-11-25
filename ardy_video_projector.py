"""Module 3 of the video -> ARDY pipeline: project a video reference into ARDY's motion manifold.

Mode A  constraint-only: ARDY from pure noise, conditioned on the sparse video constraints
        (root path, hands, adaptive full-body anchors) and an empty or minimal text prompt.
Mode B  hybrid SDEdit + constraints: each autoregressive window starts from the reference's own
        hybrid tokens (explicit root + latent body, ARDY's diffusion space) noised to
        t_start = denoise_strength * T, then denoised with the same constraints. Low strength keeps
        the video's nuance, high strength lets ARDY clean more.

Both run through ardy_critic (spatio-temporal NDF + biomechanical/contact critic with silent
rollback), which wraps ``Ardy.__call__``; the SDEdit initialization lives in ``_generate_window``,
so it also applies to the critic's re-generated windows.

Every generation gets a free lead-in and tail around the video segment (``lead``/``tail`` frames)
so the boundary continuity of the projected clip inside an ARDY sequence can be measured.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARDY_ROOT = Path(os.environ.get("ARDY_ROOT", HERE.parent / "ardy"))  # clone of nv-tlabs/ardy
CRITIC_ROOT = Path(os.environ.get("ARDY_CRITIC_ROOT", HERE.parent / "ardy_critic"))  # the critic package
for p in (ARDY_ROOT, CRITIC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ARDY settings (the same environment variables ARDY reads; override them as needed): local HF
# cache, offline, text encoder served by ARDY's own scripts/run_text_encoder_server.py
os.environ.setdefault("HF_HOME", str(ARDY_ROOT / "hf"))
os.environ.setdefault("LOCAL_CACHE", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TEXT_ENCODER_MODE", "api")
os.environ.setdefault("TEXT_ENCODER_URL", "http://127.0.0.1:9550/")
os.environ.setdefault("TEXT_ENCODERS_DIR", str(ARDY_ROOT / "te"))
os.environ.setdefault("TEXT_ENCODER_TMP_FOLDER", str(ARDY_ROOT / "tmp" / "text_encoder"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402

try:
    import ardy_critic  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise ImportError("ardy_critic not found: set ARDY_CRITIC_ROOT to the folder that holds the "
                      "ardy_critic package (see README, 'The critic').") from e
from ardy.model.ardy_model import get_three_mask_from_len, translate_normalized_root_motion  # noqa: E402
from ardy.motion_rep.tools import length_to_mask  # noqa: E402

ardy_critic.install()


def load(device="cuda:0"):
    from ardy.model import load_model

    return load_model("soma", device=device)


# ------------------------------------------------------------------------------------------
# SDEdit initialization inside ARDY's window generation
# ------------------------------------------------------------------------------------------
def _generate_window_sdedit(
    self, history_sequence, global_transl, history_start_frame, history_end_frame, total_frames,
    text_feat, text_pad_mask, first_heading_angle, motion_mask, observed_motion, num_denoising_steps,
    cfg_weight, indices, progress_bar=None, target_motion=None, cfg_type=None,
):
    """Copy of ``Ardy._generate_window`` whose generation tokens start from the noised reference
    tokens (when ``self._sdedit`` is set) instead of pure noise, skipping the noisier steps."""
    sd = getattr(self, "_sdedit", None)
    device = self.device
    batch_size = text_feat.shape[0]
    nfpt = self.num_frames_per_token
    latent_dim = self.denoiser.latent_embedding_dim
    nframe_root_dim = self.denoiser.nframe_root_dim
    G = self.gen_horizon_len
    n_gen_tok = G // nfpt
    generation_len = torch.ones(batch_size, device=device, dtype=torch.long) * G

    history_token_end = (history_end_frame - history_start_frame) // nfpt
    generation_token_end = history_token_end + n_gen_tok
    history_len = torch.ones(batch_size, device=device, dtype=torch.long) * history_token_end * nfpt
    future_len = torch.ones(batch_size, device=device, dtype=torch.long) * (total_frames - G - history_end_frame)
    history_mask, generation_mask, future_mask = get_three_mask_from_len(
        history_len, generation_len, future_len, total_frames - history_start_frame, device
    )
    if motion_mask is not None:
        cur_motion_mask = motion_mask[:, history_start_frame:] * (~history_mask[:, :, None])
        cur_root = self.motion_rep.extract_root(observed_motion[:, history_start_frame:])
        cur_body = self.motion_rep.extract_body(observed_motion[:, history_start_frame:])
        cur_root = translate_normalized_root_motion(cur_root, -global_transl, self.motion_rep)
        cur_observed_motion = self.motion_rep.concat_root_body(cur_root, cur_body) * cur_motion_mask
    else:
        cur_motion_mask = None
        cur_observed_motion = None
    history_token_mask, generation_token_mask, future_token_mask = self.hybrid.convert_frame_mask_to_token_mask(
        history_mask, generation_mask, future_mask, cur_motion_mask
    )

    shape = (batch_size, n_gen_tok, nframe_root_dim + latent_dim)
    x_t = torch.randn(shape, device=device)
    # SDEdit: the window's tokens that cover the reference are replaced, right before step i_start,
    # by the reference tokens noised to that level; tokens outside it (lead-in, tail) keep the
    # ordinary full denoising. A window fully inside the reference is plain SDEdit.
    inject = None
    if sd is not None:
        t0 = history_end_frame // nfpt - sd["token_offset"]
        z = sd["z_ref"]  # [1, Ntok, Dh], global frame
        tok = torch.arange(t0, t0 + n_gen_tok)
        inside = (tok >= 0) & (tok < z.shape[1])
        if sd.get("tok_ok") is not None:  # tokens over unreliable video frames are left to ARDY
            inside = inside & sd["tok_ok"].cpu()[tok.clamp(0, z.shape[1] - 1)]
        if bool(inside.any()):
            zw = z[:, tok.clamp(0, z.shape[1] - 1)].expand(batch_size, -1, -1).clone()
            # express the reference root in the window's local frame (as ARDY does for constraints)
            root = zw[..., :nframe_root_dim].reshape(batch_size, G, self.motion_rep.motion_root_dim)
            root = translate_normalized_root_motion(root, -global_transl, self.motion_rep)
            zw[..., :nframe_root_dim] = root.reshape(batch_size, n_gen_tok, nframe_root_dim)
            n_steps = len(indices)
            i_start = max(0, min(n_steps - 1, int(round(sd["strength"] * n_steps)) - 1))
            inject = (i_start, zw, inside.to(device))

    x = torch.zeros(batch_size, (total_frames - history_start_frame) // nfpt, nframe_root_dim + latent_dim, device=device)
    if history_token_end > 0:
        x[:, :history_token_end] = history_sequence[:, -history_token_end:]
    x[:, history_token_end:generation_token_end] = x_t

    for i in indices:
        if inject is not None and i == inject[0]:
            i_start, zw, inside = inject
            t = torch.full((batch_size,), i_start, device=device, dtype=torch.long)
            noised = self.diffusion.q_sample(zw, t, noise=torch.randn_like(zw))
            gen = x[:, history_token_end:generation_token_end]
            x[:, history_token_end:generation_token_end] = torch.where(inside[None, :, None], noised, gen)
        t = torch.tensor([i] * x_t.size(0), device=device)
        with torch.no_grad():
            x = self.denoising_step(
                x, history_len, generation_len, future_len, history_mask, generation_mask, future_mask,
                history_token_mask, generation_token_mask, future_token_mask, text_feat, text_pad_mask, t,
                first_heading_angle, cur_motion_mask, cur_observed_motion, num_denoising_steps, cfg_weight,
                target_motion, cfg_type=cfg_type,
            )
    if history_sequence is None:
        return x[:, :generation_token_end]
    return torch.cat([history_sequence, x[:, history_token_end:generation_token_end]], dim=1)


def set_sdedit(model, z_ref: torch.Tensor | None, strength: float = 0.5, token_offset: int = 0, tok_ok=None):
    """Enable (z_ref given) or disable SDEdit init. ``token_offset``: token index of the generated
    sequence where the reference starts (lead-in length / tokens)."""
    if not isinstance(getattr(model, "_generate_window", None), types.MethodType) or \
            model._generate_window.__func__ is not _generate_window_sdedit:
        model._generate_window = types.MethodType(_generate_window_sdedit, model)
    model._sdedit = None if z_ref is None else {"z_ref": z_ref, "strength": float(strength), "token_offset": token_offset,
                                                "tok_ok": tok_ok}


# ------------------------------------------------------------------------------------------
def shift_constraints(constraints, offset: int):
    for c in constraints:
        c.frame_indices = c.frame_indices + offset
    return constraints


def history_frames(model) -> int:
    fps, G, nfpt = model.motion_rep.fps, model.gen_horizon_len, model.num_frames_per_token
    max_window = (int(10 * fps) // nfpt) * nfpt
    return ((max_window - G) // nfpt) * nfpt


@torch.no_grad()
def generate(model, text: str, num_frames: int, constraints=None, cfg=(2.0, 2.0), critic: bool = True,
             z_ref=None, strength: float = 0.5, token_offset: int = 0, seed: int = 0, tok_ok=None):
    """One ARDY generation (offline path, as scripts/generate.py), normalized features [1, T, D]."""
    dev = model.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lengths = torch.tensor([num_frames], device=dev)
    om = mm = None
    if constraints:
        om, mm = model.motion_rep.create_conditions_from_constraints_batched(constraints, lengths, True, dev)
    set_sdedit(model, z_ref, strength, token_offset, tok_ok)
    model.enable_critic = critic
    try:
        feats = model(
            [text], num_frames, num_denoising_steps=int(model.diffusion.num_base_steps),
            pad_mask=length_to_mask(lengths), first_heading_angle=torch.zeros(1, device=dev),
            motion_mask=mm, observed_motion=om, cfg_weight=cfg,
            crop_history_length=history_frames(model), progress_bar=lambda x: x,
        )
    finally:
        set_sdedit(model, None)
    return feats
