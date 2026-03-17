#!/usr/bin/env python3
"""
drive_dreamer_sim.py — Test a trained DreamerV3 checkpoint in the DonkeyGym simulator.

Run in two shells:

  Shell 1 (start simulator):
      open /Users/harshwadhawe/sim/DonkeySimMac/donkey_sim.app

  Shell 2 (run autopilot):
      conda run -n dreamer-car python drive_dreamer_sim.py
      conda run -n dreamer-car python drive_dreamer_sim.py --checkpoint dreamer/checkpoints/ckpt_00100000.pt
      conda run -n dreamer-car python drive_dreamer_sim.py --episodes 5 --max-steps 500

Options
-------
--checkpoint   Path to .pt file  [default: dreamer/checkpoints/latest.pt]
--episodes     Number of episodes to run  [default: 3]
--max-steps    Max steps per episode  [default: 1000]
--record       Save episode frames + actions to a tub directory
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import donkeycar as dk

from dreamer.config import DreamerConfig
from dreamer.donkey_adapter import DreamerPilot


def run_sim(checkpoint: str, episodes: int, max_steps: int, record: bool) -> None:
    # Load Donkeycar config (reads myconfig.py overrides)
    cfg = dk.load_config()

    if not getattr(cfg, 'DONKEY_GYM', False):
        print("[drive_dreamer_sim] ERROR: DONKEY_GYM is not True in myconfig.py.")
        print("  Add: DONKEY_GYM = True")
        sys.exit(1)

    # Build the gym environment using settings from myconfig.py
    import gym_donkeycar  # noqa: F401
    import gym

    env_id   = cfg.DONKEY_GYM_ENV_NAME          # e.g. "donkey-warehouse-v0"
    sim_path = getattr(cfg, 'DONKEY_SIM_PATH', '')

    env_conf = {"exe_path": sim_path, "port": 9091}
    env = gym.make(env_id, conf=env_conf)

    # Load DreamerV3 pilot (MPS on Apple Silicon, CPU fallback)
    dreamer_cfg = DreamerConfig(device='auto')
    pilot = DreamerPilot(dreamer_cfg, checkpoint_path=checkpoint)

    print(f"[drive_dreamer_sim] env={env_id}  checkpoint={checkpoint}")
    print(f"[drive_dreamer_sim] Running {episodes} episode(s), max {max_steps} steps each")

    for ep in range(episodes):
        obs = env.reset()
        pilot.reset()

        total_reward = 0.0
        step = 0

        for step in range(max_steps):
            # obs is (H, W, 3) uint8 from DonkeyGym
            image = obs

            angle, throttle = pilot.run(image)
            action = np.array([angle, throttle], dtype=np.float32)

            obs, reward, done, _info = env.step(action)
            total_reward += reward

            if done:
                break

        print(f"[drive_dreamer_sim] Episode {ep + 1}/{episodes} — "
              f"steps={step + 1}  total_reward={total_reward:.2f}")

    env.close()
    print("[drive_dreamer_sim] Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DreamerV3 in the DonkeyGym simulator")
    parser.add_argument('--checkpoint', default='dreamer/checkpoints/latest.pt')
    parser.add_argument('--episodes',   type=int, default=3)
    parser.add_argument('--max-steps',  type=int, default=1000)
    parser.add_argument('--record',     action='store_true',
                        help='Save episode observations to a tub (not yet implemented)')
    args = parser.parse_args()
    run_sim(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        record=args.record,
    )
