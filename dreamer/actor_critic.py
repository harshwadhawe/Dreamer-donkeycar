"""
DreamerV3 Actor-Critic.

Actor : model-state features → continuous action (Gaussian w/ tanh squash)
Critic: model-state features → value estimate via two-hot symlog targets

References: Hafner et al., "Mastering Diverse Domains with World Models" (2023)
https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dreamer.config import DreamerConfig
from dreamer.world_model import MLP, symlog, symexp


# ─────────────────────────────────────────────────────────────────────────────
# Two-hot encoding for symlog-spaced critic targets   (paper §App.B)
# ─────────────────────────────────────────────────────────────────────────────

def symlog_bins(low: float, high: float, num_bins: int, device: torch.device) -> Tensor:
    """
    Symlog-spaced bin centers used for two-hot critic encoding.   [paper §App.B]
    Returns tensor of shape (num_bins,).
    """
    # Uniform bins in symlog space then mapped back via symexp
    bins = torch.linspace(symlog(torch.tensor(low)), symlog(torch.tensor(high)),
                          num_bins, device=device)
    return symexp(bins)


def two_hot_encode(x: Tensor, bins: Tensor) -> Tensor:
    """
    Two-hot encode scalar targets x using bin vector bins.   [paper §App.B]

    For each x, find the two adjacent bins (k, k+1) and spread weight
    proportionally:  w_k = (b_{k+1} - x) / (b_{k+1} - b_k),  w_{k+1} = 1 - w_k.

    x:    (*)   — scalar targets
    bins: (B_bins,)
    Returns: (*, B_bins)
    """
    B_bins = bins.shape[0]
    # Clamp to valid range
    x = x.clamp(bins[0], bins[-1])
    # Upper bin index for each x
    below = (x.unsqueeze(-1) >= bins).sum(-1) - 1   # (*)
    below = below.clamp(0, B_bins - 2)
    above = below + 1

    b_lo = bins[below]   # (*)
    b_hi = bins[above]   # (*)
    w_hi = (x - b_lo) / (b_hi - b_lo + 1e-8)
    w_lo = 1.0 - w_hi

    target = torch.zeros(*x.shape, B_bins, device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), w_lo.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), w_hi.unsqueeze(-1))
    return target


def two_hot_decode(dist: Tensor, bins: Tensor) -> Tensor:
    """Expected value from two-hot distribution: Σ p_i * b_i.   [paper §App.B]"""
    return (dist * bins).sum(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Return Normalisation  (paper §App.B — 5th/95th percentile EMA)
# ─────────────────────────────────────────────────────────────────────────────

class ReturnNormalizer:
    """
    Normalises returns using EMA estimates of the 5th and 95th percentile.
    scale = max(1, p95 - p5)    (avoid div-by-zero when returns are small)
    [paper §App.B]
    """
    def __init__(self, cfg: DreamerConfig):
        self.cfg = cfg
        self.lo = 0.0
        self.hi = 1.0

    @torch.no_grad()
    def update(self, returns: Tensor) -> None:
        # Move to CPU: torch.quantile is not supported on MPS
        r_cpu = returns.float().cpu()
        lo = torch.quantile(r_cpu, self.cfg.return_norm_perc_low).item()
        hi = torch.quantile(r_cpu, self.cfg.return_norm_perc_high).item()
        d = self.cfg.return_norm_decay
        self.lo = d * self.lo + (1 - d) * lo
        self.hi = d * self.hi + (1 - d) * hi

    def scale(self) -> float:
        return max(1.0, self.hi - self.lo)

    def normalize(self, returns: Tensor) -> Tensor:
        return returns / self.scale()


# ─────────────────────────────────────────────────────────────────────────────
# Actor   (paper §3)
# ─────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Continuous actor outputting a squashed Gaussian action.
    Architecture: MLP → (mean, log_std) → tanh(Normal(mean, std))
    Action space: [steering, throttle] ∈ [-1, 1]   [paper §3]
    """
    def __init__(self, cfg: DreamerConfig, feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = MLP(feat_dim, cfg.action_dim * 2, cfg.mlp_hidden)

    def forward(self, feat: Tensor) -> Tensor:
        """
        Returns mean action (deterministic inference mode — tanh of mean).
        For training, use `sample_with_log_prob`.
        """
        out = self.net(feat)
        mean, _ = out.chunk(2, dim=-1)
        return torch.tanh(mean)

    def sample_with_log_prob(self, feat: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Reparameterised sample + log probability.   [paper §3, actor objective]
        Returns: (action, log_prob)  both (B, A)
        """
        out = self.net(feat)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()                                # (B, A)
        action = torch.tanh(raw)

        # log-prob with tanh correction  [paper §App.B, tanh squash]
        log_prob = dist.log_prob(raw).sum(-1) - torch.log1p(-action.pow(2) + 1e-6).sum(-1)
        return action, log_prob


# ─────────────────────────────────────────────────────────────────────────────
# Critic   (paper §3)
# ─────────────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Value function using two-hot encoding over symlog-spaced bins.   [paper §3]
    Outputs a distribution over bins; expected value = symexp(Σ p_i * b_i).
    """
    def __init__(self, cfg: DreamerConfig, feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = MLP(feat_dim, cfg.twohot_bins, cfg.mlp_hidden)

    def forward(self, feat: Tensor) -> Tensor:
        """Returns logits over bins, shape (*, twohot_bins)."""
        return self.net(feat)

    def value(self, feat: Tensor, bins: Tensor) -> Tensor:
        """Expected value (in original scale via symexp).   [paper §3]"""
        logits = self.forward(feat)
        dist = torch.softmax(logits, dim=-1)
        # decode in symlog space then symexp back
        v_symlog = (dist * symlog(bins)).sum(-1)
        return symexp(v_symlog)


# ─────────────────────────────────────────────────────────────────────────────
# Lambda-return   (paper §3, Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

def lambda_returns(
    rewards: Tensor,     # (H, B) — imagined rewards
    values: Tensor,      # (H+1, B) — critic values (last is bootstrap)
    cont: Tensor,        # (H, B) — continuation flags ∈ [0,1]
    lam: float = 0.95,
    gamma: float = 0.997,
) -> Tensor:
    """
    TD-λ lambda-returns for imagination rollout.   [paper §3, Eq. 6]

    R_t^λ = r_t + γ * c_t * ((1-λ) * V_{t+1} + λ * R_{t+1}^λ)

    Computed in reverse for efficiency.
    Returns: (H, B)
    """
    H, B = rewards.shape
    returns = torch.zeros(H + 1, B, device=rewards.device)
    returns[H] = values[H]   # bootstrap from V_{H+1}

    for t in reversed(range(H)):
        returns[t] = (rewards[t]
                      + gamma * cont[t] * ((1 - lam) * values[t + 1]
                                           + lam * returns[t + 1]))
    return returns[:H]   # (H, B)


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic training step
# ─────────────────────────────────────────────────────────────────────────────

def actor_critic_loss(
    actor: Actor,
    critic: Critic,
    imag: Dict[str, Tensor],   # output of RSSM.imagine()
    reward_head,               # world_model.reward_head
    cont_head,                 # world_model.cont_head
    normalizer: ReturnNormalizer,
    cfg: DreamerConfig,
    bins: Tensor,
    target_critic: "Critic | None" = None,   # slow EMA critic for stable bootstrap
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """
    Compute actor and critic losses over an imagination rollout.

    Returns: (actor_loss, critic_loss, metrics)
    """
    feats   = imag['features']   # (H, B, feat_dim)
    actions = imag['actions']    # (H, B, A)
    H, B = feats.shape[:2]

    # ── Predict rewards and continuation from imagined states ────────────────
    feat_flat = feats.view(H * B, -1)
    with torch.no_grad():
        imag_rew  = reward_head(feat_flat).view(H, B).float()   # symlog rewards
        imag_cont = torch.sigmoid(cont_head(feat_flat)).view(H, B)

    # ── Critic values over imagination trajectory ────────────────────────────
    # Use target critic (slow EMA) for stable bootstrap values if available,
    # otherwise fall back to live critic.
    value_critic = target_critic if target_critic is not None else critic

    # Bootstrap at step H uses s_H = [h_H, z_H] from imag (the state AFTER the
    # last imagined transition), not feats[-1] = s_{H-1}.  imag['h'][-1] = h_H
    # and imag['z'][-1] = z_H are available after the imagine() ordering fix.
    boot_feat = torch.cat([imag['h'][-1:], imag['z'][-1:]], dim=-1).detach()  # (1, B, d)
    feat_with_boot = torch.cat([feats, boot_feat], dim=0)   # (H+1, B, d)
    values = value_critic.value(feat_with_boot.view((H + 1) * B, -1), bins).view(H + 1, B)

    # ── Lambda-returns   [paper §3, Eq. 6] ──────────────────────────────────
    with torch.no_grad():
        targets = lambda_returns(
            rewards=symexp(imag_rew),    # convert symlog rew → original scale
            values=values.detach(),
            cont=imag_cont,
            lam=cfg.lam,
            gamma=cfg.gamma,
        )   # (H, B)

    # ── Update return normaliser   [paper §App.B] ────────────────────────────
    normalizer.update(targets)
    targets_norm = normalizer.normalize(targets)

    # ── Critic loss: two-hot cross-entropy on symlog targets   [paper §3] ────
    critic_logits = critic(feats.view(H * B, -1)).view(H, B, -1)   # (H, B, bins)
    # encode targets in symlog space
    target_enc = two_hot_encode(symlog(targets.detach()), symlog(bins))   # (H, B, bins)
    critic_loss = -(target_enc * F.log_softmax(critic_logits, dim=-1)).sum(-1).mean()

    # ── Actor loss: REINFORCE with normalised advantage   [paper §3] ─────────
    _, log_prob = actor.sample_with_log_prob(feats.view(H * B, -1))
    log_prob = log_prob.view(H, B)
    # Clip log_prob from below to prevent tanh-saturation positive feedback:
    # when policy std→max, the squashed Gaussian log_prob becomes large positive
    # (via the Jacobian term -log(1-tanh²)), which amplifies REINFORCE gradients
    # and causes a runaway collapse.
    log_prob = log_prob.clamp(min=cfg.log_prob_min)
    advantage = (targets_norm - values[:H].detach() / normalizer.scale())
    actor_loss = -(log_prob * advantage.detach()).mean()
    # entropy bonus   [paper §3, actor objective]
    actor_loss = actor_loss - cfg.actor_entropy * (-log_prob).mean()

    metrics = {
        'ac/critic_loss':  critic_loss.item(),
        'ac/actor_loss':   actor_loss.item(),
        'ac/mean_return':  targets.mean().item(),
        'ac/return_scale': normalizer.scale(),
        'ac/log_prob':     log_prob.mean().item(),     # monitor tanh saturation: floor=-10, healthy=-2 to 0
        'ac/advantage':    advantage.mean().item(),    # should be near 0 mean; large negative = critic overestimates
    }
    return actor_loss, critic_loss, metrics
