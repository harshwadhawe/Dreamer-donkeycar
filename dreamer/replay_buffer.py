"""
Sequence Replay Buffer for DreamerV3.

Stores trajectories as dict observations with keys:
    image       : np.ndarray (H, W, 3)  uint8
    reward      : float
    is_first    : bool
    is_terminal : bool
    action      : np.ndarray (A,)  float32

Sequences are sampled with a fixed length (batch_seq_len) for the world model.
"""
from __future__ import annotations
import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from dreamer.config import DreamerConfig


class Episode:
    """A single recorded episode as a list of transition dicts."""
    def __init__(self):
        self.transitions: List[Dict] = []

    def add(self, obs: Dict) -> None:
        self.transitions.append(obs)

    def __len__(self) -> int:
        return len(self.transitions)


class ReplayBuffer:
    """
    Sequence replay buffer.

    Stores completed episodes and samples random contiguous sub-sequences
    of length `seq_len` for world-model training.
    """
    def __init__(self, cfg: DreamerConfig, capacity: int = 100_000):
        self.cfg = cfg
        self.capacity = capacity           # max total transitions stored
        self.episodes: deque[Episode] = deque()
        self._total_steps = 0

    # ── Insertion ─────────────────────────────────────────────────────────────

    def add_episode(self, episode: Episode) -> None:
        """Add a completed episode to the buffer, evicting old ones if needed."""
        self.episodes.append(episode)
        self._total_steps += len(episode)
        # Evict oldest episodes to stay within capacity
        while self._total_steps > self.capacity and len(self.episodes) > 1:
            evicted = self.episodes.popleft()
            self._total_steps -= len(evicted)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _sample_sequence(self, seq_len: int) -> List[Dict]:
        """Sample a random contiguous sequence of length seq_len from any episode."""
        # Only sample from episodes that are at least seq_len long
        valid = [ep for ep in self.episodes if len(ep) >= seq_len]
        if not valid:
            # Fall back to shortest episode and pad if necessary
            ep = max(self.episodes, key=len)
            start = 0
        else:
            ep = random.choice(valid)
            start = random.randint(0, len(ep) - seq_len)
        return ep.transitions[start: start + seq_len]

    def sample_batch(self, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, Tensor]:
        """
        Sample a batch of sequences.

        Returns dict with keys tensored to (T, B, *):
            'image'       (T, B, 3, image_size, image_size)  float32
            'action'      (T, B, A)
            'reward'      (T, B)
            'is_first'    (T, B)
            'is_terminal' (T, B)
        """
        seqs = [self._sample_sequence(seq_len) for _ in range(batch_size)]

        # Stack: list of B sequences, each of length T
        def collect_key(key):
            return [[t[key] for t in seq] for seq in seqs]   # (B, T, *)

        T = seq_len
        B = batch_size
        img_size = self.cfg.image_size

        # Images: (B, T, H, W, 3) → transpose → (T, B, 3, H, W)
        raw_imgs = np.array(collect_key('image'), dtype=np.uint8)   # (B, T, H, W, 3)
        imgs = torch.from_numpy(raw_imgs).permute(1, 0, 4, 2, 3).to(device)  # (T, B, 3, H, W)

        actions = torch.tensor(np.array(collect_key('action')), dtype=torch.float32,
                               device=device).permute(1, 0, 2)   # (T, B, A)

        rewards = torch.tensor(np.array(collect_key('reward')), dtype=torch.float32,
                               device=device).permute(1, 0)       # (T, B)

        is_first = torch.tensor(np.array(collect_key('is_first')), dtype=torch.float32,
                                device=device).permute(1, 0)      # (T, B)

        is_terminal = torch.tensor(np.array(collect_key('is_terminal')), dtype=torch.float32,
                                   device=device).permute(1, 0)   # (T, B)

        return {
            'image':       imgs,
            'action':      actions,
            'reward':      rewards,
            'is_first':    is_first,
            'is_terminal': is_terminal,
        }

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def ready(self, min_steps: Optional[int] = None) -> bool:
        """Return True if buffer has enough data to sample a batch."""
        needed = min_steps or (self.cfg.batch_size * self.cfg.batch_seq_len)
        return self._total_steps >= needed
