"""
Donkeycar Vehicle Part wrapper for DreamerV3 live autopilot.

This part can be added to the Donkeycar Vehicle pipeline alongside the
camera and drivetrain parts. It receives `cam/image_array` as input and
outputs `pilot/angle` and `pilot/throttle`.

Usage in manage.py (do NOT modify manage.py — this is illustrative only):

    from dreamer.donkey_adapter import DreamerPilot
    pilot = DreamerPilot(cfg, checkpoint_path="dreamer/checkpoints/latest.pt")
    V.add(pilot,
          inputs=['cam/image_array'],
          outputs=['pilot/angle', 'pilot/throttle'],
          run_condition='run_pilot')
"""
from __future__ import annotations
import os
from typing import Optional, Tuple

import numpy as np
import torch

from dreamer.config import DreamerConfig
from dreamer.world_model import WorldModel
from dreamer.actor_critic import Actor


class DreamerPilot:
    """
    Donkeycar part wrapping a trained DreamerV3 actor for live inference.

    The part maintains its own recurrent state (h, z) across time steps,
    resetting at the start of each new episode.

    Inputs  (from Donkeycar vehicle memory):
        cam/image_array : np.ndarray (H, W, 3) uint8

    Outputs (to Donkeycar vehicle memory):
        pilot/angle     : float ∈ [-1, 1]
        pilot/throttle  : float ∈ [-1, 1]
    """

    def __init__(self, cfg: DreamerConfig, checkpoint_path: str):
        self.cfg = cfg
        dev = cfg.device
        if dev == "auto":
            if torch.backends.mps.is_available():
                dev = "mps"
            elif torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        self.device = torch.device(dev)

        # Load world model and actor
        self.world_model = WorldModel(cfg).to(self.device)
        feat_dim = cfg.rssm_hidden + cfg.rssm_categories * cfg.rssm_classes
        self.actor = Actor(cfg, feat_dim).to(self.device)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.world_model.load_state_dict(ckpt['world_model'])
            self.actor.load_state_dict(ckpt['actor'])
            print(f"[DreamerPilot] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[DreamerPilot] Warning: checkpoint not found at {checkpoint_path}. "
                  "Running with random weights.")

        self.world_model.eval()
        self.actor.eval()

        # Recurrent state (h, z) — persists across steps
        self._h: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None
        self._last_action = np.zeros(cfg.action_dim, dtype=np.float32)

    def reset(self) -> None:
        """Reset recurrent state at episode start."""
        self._h = None
        self._z = None
        self._last_action = np.zeros(self.cfg.action_dim, dtype=np.float32)

    @torch.no_grad()
    def run(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Called once per vehicle loop step by Donkeycar.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) uint8 — raw camera frame

        Returns
        -------
        (angle, throttle) : floats in [-1, 1]
        """
        cfg = self.cfg
        device = self.device

        # Preprocessing: crop top fraction then resize — must match donkey_env.py exactly
        from PIL import Image as PILImage
        h = image.shape[0]
        img_cropped = image[int(h * cfg.IMG_CROP_TOP):, :, :]          # remove ceiling rows
        img_resized = np.array(
            PILImage.fromarray(img_cropped).resize((cfg.IMG_W, cfg.IMG_H), PILImage.BILINEAR),
            dtype=np.uint8,
        )   # → (IMG_H, IMG_W, 3)  e.g. (58, 128, 3)

        # (H, W, 3) → (1, 3, H, W) tensor
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

        # Initialise recurrent state on first call
        if self._h is None:
            self._h, self._z = self.world_model.rssm.initial_state(1, device)

        # Encode image
        embed = self.world_model.encoder(img_t.float())   # (1, E)

        # Single-step RSSM posterior update, reusing stored (h, z) across calls.
        # We do NOT call rssm.observe() because that always resets h, z to zeros.
        # Instead we replicate the corrected per-step ordering from observe():
        #   1. h_t  = GRU(h_{t-1}, z_{t-1}, a_{t-1})
        #   2. z_t  ~ q(z_t | h_t, x_t)
        rssm = self.world_model.rssm
        a_prev = torch.from_numpy(self._last_action).unsqueeze(0).to(device)  # (1, A)

        # Step 1: deterministic transition
        img_in = rssm.img_in(torch.cat([self._z, a_prev], dim=-1))
        h_new  = rssm.gru_ln(rssm.gru_cell(img_in, self._h))

        # Step 2: posterior update conditioned on new h and current image embed
        post_logit = rssm.post_head(torch.cat([h_new, embed], dim=-1)).view(1, rssm.C, rssm.K)
        z_new = rssm._sample_z(post_logit)   # (1, C*K)

        self._h = h_new
        self._z = z_new

        # Compute action from features
        feat = torch.cat([self._h, self._z], dim=-1)   # (1, H+C*K)
        action = self.actor(feat).squeeze(0).cpu().numpy()   # (A,)

        self._last_action = action
        angle    = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))
        return angle, throttle

    # ── Donkeycar part interface ───────────────────────────────────────────────

    def shutdown(self) -> None:
        """Donkeycar part shutdown hook."""
        pass
