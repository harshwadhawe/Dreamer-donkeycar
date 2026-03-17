"""
Logging utilities for DreamerV3.

Supports WandB and TensorBoard. Episode videos are saved as GIF files
using imageio.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional

import numpy as np


class Logger:
    """
    Thin wrapper around WandB and TensorBoard.
    Falls back silently if either backend is unavailable.
    """
    def __init__(self, cfg, log_dir: str = "dreamer/logs"):
        self.cfg = cfg
        self.step = 0
        self._tb_writer = None
        self._wandb = None

        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                print(f"[Logger] TensorBoard writing to {log_dir}")
            except ImportError:
                print("[Logger] TensorBoard not available.")

        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))
                self._wandb = wandb
                print(f"[Logger] WandB run: {cfg.run_name}")
            except Exception as e:
                print(f"[Logger] WandB init failed: {e}")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics."""
        step = step if step is not None else self.step
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, global_step=step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_video(self, frames: List[np.ndarray], tag: str = "video", step: Optional[int] = None) -> None:
        """
        Save a list of uint8 RGB frames as a GIF.
        frames: list of (H, W, 3) uint8 arrays.
        """
        try:
            import imageio
            step = step if step is not None else self.step
            path = os.path.join(self.log_dir, f"{tag}_{step:08d}.gif")
            imageio.mimsave(path, frames, fps=10)
            print(f"[Logger] Saved video: {path}")
        except ImportError:
            print("[Logger] imageio not available, skipping video save.")

        if self._wandb is not None:
            try:
                import imageio, wandb
                step = step if step is not None else self.step
                path = os.path.join(self.log_dir, f"{tag}_{step:08d}.gif")
                self._wandb.log({tag: wandb.Video(path, fps=10, format="gif")}, step=step)
            except Exception:
                pass

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()
