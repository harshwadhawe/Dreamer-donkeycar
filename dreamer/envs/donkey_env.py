"""
Donkeycar Tub → Offline Environment Adapter.

Replays recorded tub data from ~/dreamer-car/data/ as an offline
gym-like environment for DreamerV3 training.

Tub record format (Donkeycar v5):
    data/tub_<timestamp>/
        manifest.json        — metadata (inputs, outputs, image_path)
        catalog_<n>.catalog  — NDJSON records, each line:
            {"_index": 0, "cam/image_array": "0_cam_image_array_.jpg", ...}
        images/
            0_cam_image_array_.jpg
            1_cam_image_array_.jpg
            ...

Each record contains:
    cam/image_array  : JPEG filename (relative to tub dir)
    user/angle       : steering ∈ [-1, 1]
    user/throttle    : throttle ∈ [-1, 1]

This adapter:
  1. Scans all tubs in `tub_dir`
  2. Loads all records into Episodes
  3. Returns a ReplayBuffer pre-populated with real driving data
"""
from __future__ import annotations
import glob
import json
import os
from typing import List, Optional

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    import imageio  # fallback


def _load_image(path: str, target_size: int = 64) -> np.ndarray:
    """Load image from path, resize to (target_size, target_size, 3) uint8."""
    if HAS_PIL:
        img = Image.open(path).convert("RGB").resize((target_size, target_size))
        return np.array(img, dtype=np.uint8)
    else:
        img = imageio.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[:, :, :3]
        # Simple bilinear resize via PIL-less approach (nearest-neighbour)
        from PIL import Image as _PIL
        return np.array(_PIL.fromarray(img).resize((target_size, target_size)), dtype=np.uint8)


class DonkeyTubEnv:
    """
    Loads Donkeycar tub data and populates a ReplayBuffer.

    Usage
    -----
    env = DonkeyTubEnv(tub_dir="data/", image_size=64)
    buffer = env.load_into_buffer(cfg)
    """

    def __init__(self, tub_dir: str = "data/", image_size: int = 64):
        self.tub_dir = tub_dir
        self.image_size = image_size

    # ── Tub discovery & loading ───────────────────────────────────────────────

    def _find_tubs(self) -> List[str]:
        """
        Return list of tub directory paths.

        Donkeycar v5 supports two layouts:
          1. Nested:  data/tub_<timestamp>/manifest.json  (older workflow)
          2. Flat:    data/manifest.json                  (simulator / recent versions)
        """
        # Flat layout: data/ itself is the tub
        if os.path.exists(os.path.join(self.tub_dir, "manifest.json")):
            return [self.tub_dir]
        # Nested layout: data/tub_*/
        tubs = sorted(glob.glob(os.path.join(self.tub_dir, "tub_*")))
        return tubs

    def _load_tub(self, tub_path: str) -> Optional[List[dict]]:
        """
        Load all records from a single tub. Returns list of transition dicts,
        or None if the tub has no records.

        manifest.json in Donkeycar v5 is NDJSON (multiple JSON values per file),
        so we only check for its existence — we don't parse it.
        """
        manifest_path = os.path.join(tub_path, "manifest.json")
        if not os.path.exists(manifest_path):
            return None

        # Collect all catalog files (exclude .catalog_manifest sidecar files)
        catalog_files = sorted(
            f for f in glob.glob(os.path.join(tub_path, "catalog_*.catalog"))
            if not f.endswith("_manifest")
        )
        if not catalog_files:
            return None

        records = []
        for cat_path in catalog_files:
            with open(cat_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(rec)

        if not records:
            return None

        transitions = []
        for i, rec in enumerate(records):
            # Image path
            img_filename = rec.get("cam/image_array")
            if img_filename is None:
                continue

            # Donkeycar stores images under tub_path/images/ or directly in tub_path
            img_path = os.path.join(tub_path, "images", img_filename)
            if not os.path.exists(img_path):
                img_path = os.path.join(tub_path, img_filename)
            if not os.path.exists(img_path):
                continue

            try:
                image = _load_image(img_path, self.image_size)
            except Exception:
                continue

            angle    = float(rec.get("user/angle",    0.0))
            throttle = float(rec.get("user/throttle", 0.0))

            action = np.array([angle, throttle], dtype=np.float32)
            # Compute a simple reward: forward progress proportional to throttle
            reward = float(throttle)

            transitions.append({
                "image":       image,                 # (H, W, 3) uint8
                "action":      action,                # (2,) float32
                "reward":      reward,
                "is_first":    float(i == 0),
                "is_terminal": float(i == len(records) - 1),
            })

        return transitions if transitions else None

    # ── Public API ────────────────────────────────────────────────────────────

    def load_into_buffer(self, cfg) -> "ReplayBuffer":
        """
        Load all tub data and return a pre-filled ReplayBuffer.

        Parameters
        ----------
        cfg : DreamerConfig

        Returns
        -------
        ReplayBuffer with all episodes loaded.
        """
        from dreamer.replay_buffer import ReplayBuffer, Episode

        buffer = ReplayBuffer(cfg)
        tubs = self._find_tubs()
        total = 0

        if not tubs:
            print(f"[DonkeyTubEnv] No tubs found in '{self.tub_dir}'. "
                  "Buffer will be empty — using synthetic data for smoke test.")
            return buffer

        for tub_path in tubs:
            transitions = self._load_tub(tub_path)
            if not transitions:
                print(f"[DonkeyTubEnv] Skipping empty/invalid tub: {tub_path}")
                continue

            ep = Episode()
            for t in transitions:
                ep.add(t)
            buffer.add_episode(ep)
            total += len(ep)
            print(f"[DonkeyTubEnv] Loaded {len(ep):>5} steps from {os.path.basename(tub_path)}")

        print(f"[DonkeyTubEnv] Total: {buffer.num_episodes} episodes, {total} steps.")
        return buffer

    def make_synthetic_buffer(self, cfg, num_steps: int = 1000) -> "ReplayBuffer":
        """
        Create a buffer filled with random synthetic data.
        Used for smoke tests when no real tub data is available.
        """
        from dreamer.replay_buffer import ReplayBuffer, Episode

        buffer = ReplayBuffer(cfg)
        ep = Episode()
        rng = np.random.default_rng(42)

        for i in range(num_steps):
            ep.add({
                "image":       rng.integers(0, 255,
                                            (cfg.image_size, cfg.image_size, 3),
                                            dtype=np.uint8),
                "action":      rng.uniform(-1, 1, (cfg.action_dim,)).astype(np.float32),
                "reward":      float(rng.uniform(-0.1, 0.5)),
                "is_first":    float(i == 0),
                "is_terminal": float(i == num_steps - 1),
            })

        buffer.add_episode(ep)
        print(f"[DonkeyTubEnv] Created synthetic buffer: {num_steps} steps.")
        return buffer
