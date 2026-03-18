"""
DreamerV3 offline training script for Donkeycar tub data.

Usage (always run with dreamer-car conda env):

    conda run -n dreamer-car python dreamer/train.py --tubs data/ --steps 100000

For a quick smoke test:
    conda run -n dreamer-car python dreamer/train.py --tubs data/ --steps 100
"""
from __future__ import annotations
import copy
import os
import sys

# Allow running from repo root: `python dreamer/train.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dreamer.config import parse_config
from dreamer.world_model import WorldModel
from dreamer.actor_critic import Actor, Critic, ReturnNormalizer, actor_critic_loss, symlog_bins
from dreamer.replay_buffer import ReplayBuffer
from dreamer.logger import Logger
from dreamer.envs.donkey_env import DonkeyTubEnv


def save_checkpoint(
    step: int,
    world_model: WorldModel,
    actor: Actor,
    critic: Critic,
    target_critic,
    wm_optim: torch.optim.Optimizer,
    actor_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer,
    cfg,
) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:08d}.pt")
    torch.save({
        'step':          step,
        'world_model':   world_model.state_dict(),
        'actor':         actor.state_dict(),
        'critic':        critic.state_dict(),
        'target_critic': target_critic.state_dict(),
        'wm_optim':      wm_optim.state_dict(),
        'actor_optim':   actor_optim.state_dict(),
        'critic_optim':  critic_optim.state_dict(),
    }, path)
    # Write a "latest" symlink for easy loading
    latest = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    try:
        os.symlink(os.path.abspath(path), latest)
    except OSError:
        pass   # Windows may not support symlinks


def train(cfg) -> None:
    if cfg.device == "auto":
        if torch.backends.mps.is_available():
            cfg.device = "mps"
        elif torch.cuda.is_available():
            cfg.device = "cuda"
        else:
            cfg.device = "cpu"
    device = torch.device(cfg.device)
    print(f"[Train] Device: {device}")
    print(f"[Train] Steps: {cfg.steps}, batch={cfg.batch_size}×{cfg.batch_seq_len}")

    # ── Load tub data ─────────────────────────────────────────────────────────
    env = DonkeyTubEnv(tub_dir=cfg.tubs, cfg=cfg)
    buffer: ReplayBuffer = env.load_into_buffer(cfg)

    if not buffer.ready():
        print(f"[Train] No real tub data found or insufficient data "
              f"(have {buffer.total_steps} steps, need "
              f"{cfg.batch_size * cfg.batch_seq_len}). "
              f"Generating synthetic data for smoke test...")
        buffer = env.make_synthetic_buffer(cfg, num_steps=max(cfg.steps * 2, 1000))

    # ── Build models ──────────────────────────────────────────────────────────
    world_model = WorldModel(cfg).to(device)
    feat_dim = cfg.rssm_hidden + cfg.rssm_categories * cfg.rssm_classes
    actor   = Actor(cfg, feat_dim).to(device)
    critic  = Critic(cfg, feat_dim).to(device)
    normalizer = ReturnNormalizer(cfg)

    # Symlog-spaced bins for two-hot critic   [paper §App.B]
    bins = symlog_bins(cfg.twohot_low, cfg.twohot_high, cfg.twohot_bins, device)

    # ── Target critic (slow EMA of critic weights for stable bootstrap) ───────
    target_critic = copy.deepcopy(critic).to(device)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    # ── Optimisers (separate per module, per DreamerV3 paper) ─────────────────
    wm_optim     = torch.optim.Adam(world_model.parameters(), lr=cfg.wm_lr,  eps=cfg.eps)
    actor_optim  = torch.optim.Adam(actor.parameters(),       lr=cfg.ac_lr,  eps=cfg.eps)
    critic_optim = torch.optim.Adam(critic.parameters(),      lr=cfg.ac_lr,  eps=cfg.eps)

    logger = Logger(cfg, log_dir=os.path.join(cfg.checkpoint_dir, "..", "logs"))

    print(f"[Train] WM warmup: {cfg.wm_warmup} steps before AC starts", flush=True)
    print("[Train] Starting training loop...", flush=True)
    _ac_started = False
    for step in range(1, cfg.steps + 1):

        # ────────────────────────────────────────────────────────────────────
        # Phase 1: World Model training
        # ────────────────────────────────────────────────────────────────────
        world_model.train()
        actor.eval()
        critic.eval()

        batch = buffer.sample_batch(cfg.batch_size, cfg.batch_seq_len, device)

        wm_optim.zero_grad()
        wm_loss, wm_metrics, rssm_out = world_model.loss(batch)
        wm_loss.backward()
        # Gradient clipping: norm 1000 for world model  [paper §App.B]
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), cfg.wm_grad_clip)
        wm_optim.step()

        # ────────────────────────────────────────────────────────────────────
        # Phase 2: Actor-Critic training via imagination
        # Delayed by wm_warmup steps so the world model has time to learn
        # useful representations before the AC starts.
        # ────────────────────────────────────────────────────────────────────
        if step <= cfg.wm_warmup:
            ac_metrics = {
                'ac/critic_loss': 0.0, 'ac/actor_loss': 0.0,
                'ac/mean_return': 0.0, 'ac/return_scale': 1.0,
                'ac/log_prob': 0.0,    'ac/advantage': 0.0,
            }
        else:
            if not _ac_started:
                print(f"[Train] step={step}: WM warmup done — starting Actor-Critic training", flush=True)
                _ac_started = True
            world_model.eval()
            actor.train()
            critic.train()

            # Use posterior states from the world-model batch as imagination seeds
            with torch.no_grad():
                T, B = rssm_out['h'].shape[:2]
                h_flat = rssm_out['h'].view(T * B, -1).detach()
                z_flat = rssm_out['z'].view(T * B, -1).detach()
                idx = torch.randperm(T * B, device=device)[:B]
                h0 = h_flat[idx]
                z0 = z_flat[idx]

                imag = world_model.rssm.imagine(
                    h0, z0,
                    actor_fn=actor,
                    horizon=cfg.imag_horizon,
                )

            actor_loss, critic_loss, ac_metrics = actor_critic_loss(
                actor, critic, imag,
                reward_head=world_model.reward_head,
                cont_head=world_model.cont_head,
                normalizer=normalizer,
                cfg=cfg,
                bins=bins,
                target_critic=target_critic,
            )

            # Separate backward passes + gradient clipping per module
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.ac_grad_clip)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.ac_grad_clip)
            critic_optim.step()

            # Update target critic via EMA
            with torch.no_grad():
                d = cfg.critic_ema_decay
                for tp, cp in zip(target_critic.parameters(), critic.parameters()):
                    tp.data.mul_(d).add_(cp.data, alpha=1.0 - d)

        # ────────────────────────────────────────────────────────────────────
        # Logging & checkpointing
        # ────────────────────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            metrics = {**wm_metrics, **ac_metrics, 'train/step': step}
            logger.log(metrics, step=step)
            print(
                f"[Train] step={step:>7}  "
                f"wm={wm_metrics['wm/total']:.4f}  "
                f"kl={wm_metrics['wm/kl']:.4f}  "
                f"rec={wm_metrics['wm/rec_loss']:.4f}  "
                f"rew={wm_metrics['wm/rew_loss']:.4f}  "
                f"cont={wm_metrics['wm/cont_loss']:.4f}  "
                f"actor={ac_metrics['ac/actor_loss']:.4f}  "
                f"critic={ac_metrics['ac/critic_loss']:.4f}  "
                f"ret={ac_metrics['ac/mean_return']:.4f}  "
                f"logp={ac_metrics['ac/log_prob']:.3f}  "
                f"adv={ac_metrics['ac/advantage']:.3f}",
                flush=True,
            )

        if step % cfg.save_every == 0:
            save_checkpoint(step, world_model, actor, critic, target_critic,
                            wm_optim, actor_optim, critic_optim, cfg)
            print(f"[Train] Saved checkpoint → ckpt_{step:08d}.pt", flush=True)

    # Final checkpoint
    save_checkpoint(cfg.steps, world_model, actor, critic, target_critic,
                    wm_optim, actor_optim, critic_optim, cfg)
    logger.close()
    print("[Train] Done.", flush=True)


if __name__ == "__main__":
    cfg = parse_config()
    train(cfg)
