"""
DreamerV3 hyperparameter configuration.
All defaults match the DreamerV3 paper (Hafner et al., 2023).
CLI overrides applied via argparse after dataclass construction.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field, fields


@dataclass
class DreamerConfig:
    # ── Data / Environment ──────────────────────────────────────────────────
    tubs: str = "data/"                 # path to donkeycar tub directory
    # Image preprocessing: native 160×120 → crop top 40% → resize to 128×58
    # Cropping removes ceiling/girders; aspect ratio 2.22:1 preserved.
    IMG_H: int = 58                     # model input height (pixels)
    IMG_W: int = 128                    # model input width  (pixels)
    IMG_CROP_TOP: float = 0.4          # fraction of original height removed from top
    action_dim: int = 2                 # [steering, throttle] ∈ [-1, 1]

    # ── Training schedule ───────────────────────────────────────────────────
    steps: int = 100_000               # total environment / replay steps
    batch_size: int = 16               # B
    batch_seq_len: int = 64            # T  — sequence length per batch
    train_ratio: int = 512             # gradient steps per data step (offline: train_ratio batches per epoch)
    wm_lr: float = 1e-4               # world-model learning rate
    ac_lr: float = 3e-5               # actor-critic learning rate
    eps: float = 1e-8                  # Adam ε

    # ── World model ─────────────────────────────────────────────────────────
    # RSSM discrete latent: 32 categories × 32 classes  (paper §2)
    rssm_categories: int = 32
    rssm_classes: int = 32
    rssm_hidden: int = 256             # GRU hidden size — reduced from 512 to bottleneck h,
                                       # forcing z to carry information (prevents KL floor collapse)
    rssm_embed: int = 512              # CNN encoder output dimension
    mlp_hidden: int = 512              # hidden size for all MLPs

    # KL balancing (paper §App.B): α=0.8 → 80% prior, 20% posterior
    kl_balance: float = 0.8
    kl_free: float = 0.1               # free-bits threshold (nats) — clip below this
    kl_scale: float = 1.0              # KL loss scale — back to 1.0; high kl_scale above the
                                       # free-bits floor has zero gradient (clamp kills it) and
                                       # actually harms z when KL > floor by over-penalising
    rec_scale: float = 1.0             # reconstruction loss scale
    rew_scale: float = 1.0             # reward prediction loss scale
    cont_scale: float = 1.0            # continuation prediction loss scale

    # Unimix: 1% uniform mixing for all categorical distributions (paper §App.B)
    unimix: float = 0.01

    # ── Actor-Critic ─────────────────────────────────────────────────────────
    imag_horizon: int = 8              # imagination rollout horizon H — reduced from 15
                                       # shorter horizon = less OOD divergence when z is still learning
    gamma: float = 0.997               # discount factor
    lam: float = 0.95                  # λ for lambda-returns  (paper Eq. 6)
    actor_entropy: float = 1e-3        # entropy regularisation weight — applied to Gaussian
                                       # entropy (not clamped log_prob), so gradient always
                                       # flows through log_std. Increased from 3e-4 to 1e-3
                                       # to counteract tanh-saturation runaway.
    actor_grad: str = "reinforce"      # "reinforce" or "dynamics" — use REINFORCE for discrete

    # Two-hot bins for symlog-spaced critic targets (paper §App.B)
    twohot_bins: int = 255
    twohot_low: float = -20.0
    twohot_high: float = 20.0

    # Return normalisation: 5th–95th percentile  (paper §App.B)
    return_norm_perc_low: float = 0.05
    return_norm_perc_high: float = 0.95
    return_norm_decay: float = 0.99    # EMA decay for percentile estimates

    # ── Gradient clipping ────────────────────────────────────────────────────
    wm_grad_clip: float = 1000.0       # world model (paper §App.B)
    ac_grad_clip: float = 100.0        # actor-critic (paper §App.B)

    # ── Training stability ───────────────────────────────────────────────────
    wm_warmup: int = 2000              # steps to train WM-only before starting AC
    critic_ema_decay: float = 0.98     # EMA decay for slow target critic
    log_prob_min: float = -10.0        # floor for actor log_prob
    log_prob_max: float = 2.0          # ceiling for actor log_prob — prevents POSITIVE tanh-Jacobian
                                       # runaway: -log(1-tanh²) → +13.8/dim when action≈±1,
                                       # which makes REINFORCE reward saturation → feedback loop

    # ── Logging / checkpointing ──────────────────────────────────────────────
    log_every: int = 100               # log metrics every N gradient steps
    save_every: int = 1000             # save checkpoint every N gradient steps
    checkpoint_dir: str = "dreamer/checkpoints"
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "dreamer-car"
    run_name: str = "dreamer_v3"

    # ── Online simulator collection ──────────────────────────────────────────
    use_sim: bool = False              # enable live DonkeyGym data collection
    sim_host: str = "localhost"
    sim_port: int = 9091
    sim_env: str = "donkey-warehouse-v0"
    sim_collect_every: int = 200       # collect sim data every N training steps
    sim_steps_per_collect: int = 100   # env steps per collection phase
    sim_max_episode_steps: int = 500   # reset episode after this many steps

    # ── Device ───────────────────────────────────────────────────────────────
    device: str = "auto"               # "auto" → MPS > CUDA > CPU


def parse_config() -> DreamerConfig:
    """Build a DreamerConfig from defaults + CLI overrides."""
    cfg = DreamerConfig()

    parser = argparse.ArgumentParser(description="DreamerV3 for Donkeycar")
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, bool):
            # argparse type=bool is broken: bool("False") == True.
            # Accept "true/1/yes" as True, everything else as False.
            parser.add_argument(
                f"--{f.name}",
                type=lambda v: v.lower() in ("true", "1", "yes"),
                default=val,
                help=f"(default: {val})",
            )
        else:
            parser.add_argument(
                f"--{f.name}",
                type=type(val),
                default=val,
                help=f"(default: {val})",
            )

    args, _ = parser.parse_known_args()
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(args, f.name))

    return cfg
