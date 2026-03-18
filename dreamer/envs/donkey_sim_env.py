"""
Online DonkeyGym data collector for DreamerV3.

Runs the current world-model + actor against a live DonkeyGym simulator,
collecting (image, action, reward, is_first, is_terminal) transitions and
adding complete episodes to the replay buffer.

Benefits over offline tub data alone
-------------------------------------
- Real CTE signal → reward penalises off-track driving
- Real crash / timeout terminations → cont_head finally gets negative examples
- Diverse state coverage → imagination rollouts stay in-distribution
- Adaptive: policy visits states it's actually uncertain about

Usage
-----
Pass --use_sim True to train.py. The simulator must already be running:

    Shell 1: open /Users/harshwadhawe/sim/DonkeySimMac/donkey_sim.app
    Shell 2: python dreamer/train.py --use_sim True --steps 100000
"""
from __future__ import annotations

import numpy as np
import torch

from dreamer.config import DreamerConfig


class DonkeySimCollector:
    """
    Collects online transitions from DonkeyGym using the current actor.

    Maintains RSSM recurrent state within each episode (same as DreamerPilot).
    Complete episodes (crash, timeout, or done=True) are added to the buffer
    immediately so the world model can learn from real termination events.
    """

    def __init__(self, cfg: DreamerConfig, world_model, actor, buffer) -> None:
        self.cfg = cfg
        self.world_model = world_model
        self.actor = actor
        self.buffer = buffer
        self.device: torch.device | None = None
        self.env = None

        # Recurrent state within current episode
        self._h: torch.Tensor | None = None
        self._z: torch.Tensor | None = None
        self._last_action = np.zeros(cfg.action_dim, dtype=np.float32)
        self._last_obs: np.ndarray | None = None
        self._episode = None
        self._episode_step: int = 0
        self._total_steps: int = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def connect(self, device: torch.device) -> bool:
        """Open the DonkeyGym environment. Returns True on success."""
        self.device = device
        try:
            import gym
            import gym_donkeycar  # noqa: F401 — registers envs as side effect
            conf = {
                "host": self.cfg.sim_host,
                "port": self.cfg.sim_port,
                "cam_resolution": (120, 160),
            }
            self.env = gym.make(self.cfg.sim_env, conf=conf)
            obs = self.env.reset()
            self._start_episode(obs)
            print(
                f"[SimCollector] Connected → {self.cfg.sim_env} "
                f"@ {self.cfg.sim_host}:{self.cfg.sim_port}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[SimCollector] Could not connect: {exc}", flush=True)
            self.env = None
            return False

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        """Crop top fraction then resize — identical to donkey_env.py and donkey_adapter.py."""
        from PIL import Image as PILImage
        h = obs.shape[0]
        img = obs[int(h * self.cfg.IMG_CROP_TOP):, :, :]
        img = np.array(
            PILImage.fromarray(img).resize((self.cfg.IMG_W, self.cfg.IMG_H), PILImage.BILINEAR),
            dtype=np.uint8,
        )
        return img

    def _start_episode(self, raw_obs: np.ndarray) -> None:
        """Reset RSSM state and open a fresh Episode."""
        from dreamer.replay_buffer import Episode
        self._h, self._z = self.world_model.rssm.initial_state(1, self.device)
        self._last_action = np.zeros(self.cfg.action_dim, dtype=np.float32)
        self._last_obs = raw_obs
        self._episode = Episode()
        self._episode_step = 0

    @torch.no_grad()
    def _select_action(self, image: np.ndarray, explore: bool) -> np.ndarray:
        """
        RSSM posterior update followed by actor sample.
        Mirrors the single-step inference in donkey_adapter.py.

        During exploration (WM warmup) uses random actions but still updates
        the RSSM state so h/z remain in-distribution for later use.
        """
        # Encode image and update RSSM state
        img_t = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        embed = self.world_model.encoder(img_t)   # (1, E)

        rssm   = self.world_model.rssm
        a_prev = torch.from_numpy(self._last_action).unsqueeze(0).to(self.device)

        img_in     = rssm.img_in(torch.cat([self._z, a_prev], dim=-1))
        h_new      = rssm.gru_ln(rssm.gru_cell(img_in, self._h))
        post_logit = rssm.post_head(torch.cat([h_new, embed], dim=-1)).view(1, rssm.C, rssm.K)

        self._h = h_new
        self._z = rssm._sample_z(post_logit)

        if explore:
            action = np.random.uniform(-1.0, 1.0, self.cfg.action_dim).astype(np.float32)
        else:
            # Stochastic sample for diverse data collection
            feat = torch.cat([self._h, self._z], dim=-1)
            action, _ = self.actor.sample_with_log_prob(feat)
            action = action.squeeze(0).cpu().numpy()

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _compute_reward(action: np.ndarray, info: dict, done: bool) -> float:
        """
        CTE-aware reward, aligned with the Option A offline reward shape:

            survival  +  speed  -  cte_penalty  -  steer_penalty  -  crash_penalty

        cte_penalty : 0 on-track, increases linearly up to 5 m off-track.
        steer_penalty: activates only when |steering| > 0.7 (lock-to-lock spin).
        crash_penalty: −1 on wall hit so cont_head gets a real negative signal.
        """
        steering = float(action[0])
        throttle  = float(action[1])
        cte       = abs(float(info.get("cte", 0.0)))

        survival      = 0.1
        speed         = max(0.0, throttle)
        cte_penalty   = 0.3 * min(cte, 5.0)
        steer_penalty = max(0.0, abs(steering) - 0.7) * 0.5
        crash_penalty = 1.0 if (done and info.get("hit", False)) else 0.0

        return survival + speed - cte_penalty - steer_penalty - crash_penalty

    # ── Public API ────────────────────────────────────────────────────────────

    def collect(self, n_steps: int, explore: bool = False) -> dict:
        """
        Collect n_steps transitions from the simulator.

        Complete episodes are added to the buffer immediately.
        Partial episodes carry over to the next call so no data is lost
        at collect() boundaries.

        Parameters
        ----------
        n_steps  : int  — number of environment steps to take
        explore  : bool — random actions when True (used during WM warmup)

        Returns
        -------
        dict of sim/* scalar metrics for logging.
        """
        if self.env is None:
            return {}

        self.world_model.eval()
        self.actor.eval()

        steps = episodes = 0
        total_rew = 0.0

        while steps < n_steps:
            image  = self._preprocess(self._last_obs)
            action = self._select_action(image, explore=explore)

            next_obs, _gym_rew, done, info = self.env.step(action)
            reward  = self._compute_reward(action, info, done)
            timeout = (self._episode_step + 1) >= self.cfg.sim_max_episode_steps

            self._episode.add({
                "image":       image,
                "action":      action,
                "reward":      float(reward),
                "is_first":    float(self._episode_step == 0),
                "is_terminal": float(done or timeout),
            })

            self._last_action = action
            total_rew += reward
            steps     += 1
            self._episode_step += 1
            self._total_steps  += 1

            if done or timeout:
                self.buffer.add_episode(self._episode)
                episodes += 1
                obs = self.env.reset()
                self._start_episode(obs)
            else:
                self._last_obs = next_obs

        return {
            "sim/steps":       steps,
            "sim/episodes":    episodes,
            "sim/mean_reward": total_rew / max(steps, 1),
            "sim/total_steps": self._total_steps,
        }
