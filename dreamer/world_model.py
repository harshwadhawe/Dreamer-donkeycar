"""
DreamerV3 World Model.

Components
----------
- CNN Encoder  : image → embed vector
- RSSM         : (embed, action, h) → (z, h')   — discrete latents
- CNN Decoder  : (z, h) → image reconstruction
- Reward Head  : (z, h) → symlog(reward)
- Continue Head: (z, h) → p(not_terminal)

References: Hafner et al., "Mastering Diverse Domains with World Models" (2023)
https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dreamer.config import DreamerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Utility: symlog / symexp   (paper Eq. 4)
# ─────────────────────────────────────────────────────────────────────────────

def symlog(x: Tensor) -> Tensor:
    """Symmetric log transform: sign(x) * log(|x| + 1)  [paper Eq. 4]."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)  [paper Eq. 4]."""
    return torch.sign(x) * (x.abs().exp() - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Unimix categorical   (paper §App.B)
# ─────────────────────────────────────────────────────────────────────────────

def unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    """
    Mix predicted categorical distribution with a uniform prior.
    p = (1 - ε) * softmax(logits) + ε * Uniform      [paper §App.B]
    """
    probs = torch.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    return (1.0 - unimix) * probs + unimix * uniform


# ─────────────────────────────────────────────────────────────────────────────
# CNN Encoder   (paper §App.B — "CNN")
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Residual CNN encoder: (B, 3, 64, 64) → (B, embed_dim).
    Depth multiplier doubles every layer as in the paper.
    """
    def __init__(self, embed_dim: int = 512, depth: int = 48):
        super().__init__()
        # channel sizes: depth * [1, 2, 4, 8]  →  48, 96, 192, 384
        channels = [3] + [depth * (2 ** i) for i in range(4)]
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1)
            for i in range(4)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm([channels[i + 1], 64 >> (i + 1), 64 >> (i + 1)])
                                    for i in range(4)])
        flat_size = channels[-1] * 4 * 4   # 384 * 4 * 4 = 6144 after 4 strides
        self.out = nn.Linear(flat_size, embed_dim)
        self.out_ln = nn.LayerNorm(embed_dim)

    def forward(self, image: Tensor) -> Tensor:
        """image: (B, 3, H, W) with pixel values in [0, 255]."""
        x = image.float() / 255.0 - 0.5   # normalise
        for conv, norm in zip(self.convs, self.norms):
            x = F.silu(norm(conv(x)))
        x = x.flatten(1)
        return F.silu(self.out_ln(self.out(x)))


# ─────────────────────────────────────────────────────────────────────────────
# CNN Decoder   (paper §App.B — "CNN")
# ─────────────────────────────────────────────────────────────────────────────

class CNNDecoder(nn.Module):
    """
    Transposed-CNN decoder: (B, latent_dim) → (B, 3, 64, 64) in [0,1].
    Mirrors the encoder.
    """
    def __init__(self, latent_dim: int, depth: int = 48):
        super().__init__()
        channels = [depth * (2 ** i) for i in range(4)][::-1]   # 384, 192, 96, 48
        self.in_proj = nn.Linear(latent_dim, channels[0] * 4 * 4)
        self.in_ln = nn.LayerNorm(channels[0] * 4 * 4)
        self.ch0 = channels[0]

        layer_channels = channels + [3]
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose2d(layer_channels[i], layer_channels[i + 1],
                               kernel_size=4, stride=2, padding=1)
            for i in range(4)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm([layer_channels[i + 1], 4 * (2 ** (i + 1)), 4 * (2 ** (i + 1))])
            for i in range(3)
        ])

    def forward(self, latent: Tensor) -> Tensor:
        """Returns reconstructed image logits (no sigmoid); apply sigmoid for probabilities."""
        x = F.silu(self.in_ln(self.in_proj(latent)))
        x = x.view(x.shape[0], self.ch0, 4, 4)
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            if i < len(self.norms):
                x = F.silu(self.norms[i](x))
        return x   # (B, 3, 64, 64) — logits for Bernoulli / MSE target


# ─────────────────────────────────────────────────────────────────────────────
# MLP with Layer Norm   (paper §App.B)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple feed-forward MLP with LayerNorm + SiLU activations."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.net = nn.Sequential(*[
            nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.SiLU())
            for i in range(layers)
        ])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# RSSM — Recurrent State Space Model   (paper §2, Fig.2)
# ─────────────────────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    """
    Recurrent State Space Model with discrete latents.

    State representation at time t:
        h_t  — deterministic recurrent state (GRU hidden)          [paper §2]
        z_t  — stochastic latent: 32 categories × 32 classes        [paper §2]

    The full "model state" is the concatenation: s_t = [h_t ; flat(z_t)]

    Prior  p(z_t | h_t)              — used during imagination
    Posterior  q(z_t | h_t, embed_t) — used during training
    """

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.rssm_categories
        K = cfg.rssm_classes
        H = cfg.rssm_hidden
        E = cfg.rssm_embed
        A = cfg.action_dim
        self.C = C
        self.K = K
        self.H = H
        self.latent_dim = C * K           # flat discrete latent size

        # Input projection for GRU: (z_{t-1}, a_{t-1}) → r_t
        self.img_in = nn.Sequential(
            nn.Linear(self.latent_dim + A, H),
            nn.LayerNorm(H),
            nn.SiLU(),
        )
        # GRU recurrent core with layer norm on the update
        self.gru_cell = nn.GRUCell(H, H)
        self.gru_ln = nn.LayerNorm(H)

        # Prior: h_t → logits over z_t   (Eq. 2 / Fig.2 left branch)
        self.prior_head = MLP(H, C * K, cfg.mlp_hidden, layers=1)

        # Posterior: [h_t, embed_t] → logits over z_t   (Fig.2 right branch)
        self.post_head = MLP(H + E, C * K, cfg.mlp_hidden, layers=1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def initial_state(self, batch: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Returns (h_0, z_0) both zeroed, z in flat form (B, C*K)."""
        h = torch.zeros(batch, self.H, device=device)
        z = torch.zeros(batch, self.latent_dim, device=device)
        return h, z

    def _sample_z(self, logits: Tensor) -> Tensor:
        """
        Straight-through gradient for discrete categorical sample.
        logits: (B, C, K)
        Returns flat one-hot: (B, C*K)   [paper §App.B straight-through]
        """
        B, C, K = logits.shape
        # mix in uniform prior (unimix)
        probs = unimix_probs(logits.view(B * C, K), self.cfg.unimix).view(B, C, K)
        # torch.multinomial works on MPS; Categorical.sample() does not reliably
        indices = torch.multinomial(probs.view(B * C, K), num_samples=1).squeeze(-1).view(B, C)
        z_hard = F.one_hot(indices, K).float()                            # (B, C, K)
        # straight-through: forward = hard sample, backward = soft probs
        z_soft = probs
        z_st = z_hard + (z_soft - z_soft.detach())                        # (B, C, K)
        return z_st.view(B, C * K)

    # ── forward passes ────────────────────────────────────────────────────────

    def observe(
        self,
        embeds: Tensor,   # (T, B, E)  — encoder outputs
        actions: Tensor,  # (T, B, A)  — actions taken at each step
        is_first: Tensor, # (T, B)     — 1 at episode start (resets state)
    ) -> Dict[str, Tensor]:
        """
        Roll RSSM over a sequence with posterior updates.
        Returns dict with keys:
            'h'          (T, B, H)
            'z'          (T, B, C*K)
            'prior_logits'    (T, B, C, K)
            'post_logits'     (T, B, C, K)
        """
        T, B = embeds.shape[:2]
        device = embeds.device
        h, z = self.initial_state(B, device)

        hs, zs, prior_logits_list, post_logits_list = [], [], [], []

        for t in range(T):
            # Reset state at episode boundaries
            first_mask = is_first[t].unsqueeze(-1).float()   # (B, 1)
            h = h * (1.0 - first_mask)
            z = z * (1.0 - first_mask)

            # Prior: predict z from h (before seeing embed)  [paper §2 Eq.2]
            prior_logit = self.prior_head(h).view(B, self.C, self.K)

            # Posterior: refine z using image embed  [paper §2 Eq.3]
            post_logit = self.post_head(torch.cat([h, embeds[t]], dim=-1)).view(B, self.C, self.K)
            z = self._sample_z(post_logit)   # (B, C*K)

            # Recurrent update with (z, a)  [paper §2 Eq.1]
            img_in = self.img_in(torch.cat([z, actions[t]], dim=-1))
            h = self.gru_ln(self.gru_cell(img_in, h))

            hs.append(h)
            zs.append(z)
            prior_logits_list.append(prior_logit)
            post_logits_list.append(post_logit)

        return {
            'h': torch.stack(hs),                      # (T, B, H)
            'z': torch.stack(zs),                      # (T, B, C*K)
            'prior_logits': torch.stack(prior_logits_list),   # (T, B, C, K)
            'post_logits':  torch.stack(post_logits_list),    # (T, B, C, K)
        }

    def imagine(
        self,
        h: Tensor,          # (B, H)  — starting recurrent state
        z: Tensor,          # (B, C*K)
        actor_fn,           # callable: features (B, H+C*K) → action (B, A)
        horizon: int = 15,
    ) -> Dict[str, Tensor]:
        """
        Imagination rollout using prior only (no observations).   [paper §3]
        Returns dict with keys: 'h', 'z', 'features', 'actions'   each (H, B, .).
        """
        B = h.shape[0]
        hs, zs, feats, acts = [], [], [], []

        for _ in range(horizon):
            feat = torch.cat([h, z], dim=-1)    # (B, H + C*K)
            action = actor_fn(feat)             # (B, A)

            # Prior step  [paper §2 Eq.2]
            prior_logit = self.prior_head(h).view(B, self.C, self.K)
            z = self._sample_z(prior_logit)

            # Recurrent step
            img_in = self.img_in(torch.cat([z, action], dim=-1))
            h = self.gru_ln(self.gru_cell(img_in, h))

            hs.append(h)
            zs.append(z)
            feats.append(feat)
            acts.append(action)

        return {
            'h':        torch.stack(hs),      # (H, B, H_dim)
            'z':        torch.stack(zs),      # (H, B, C*K)
            'features': torch.stack(feats),   # (H, B, H_dim+C*K)
            'actions':  torch.stack(acts),    # (H, B, A)
        }


# ─────────────────────────────────────────────────────────────────────────────
# KL Loss with balancing and free bits   (paper §App.B)
# ─────────────────────────────────────────────────────────────────────────────

def kl_loss(
    post_logits: Tensor,   # (*, C, K)
    prior_logits: Tensor,  # (*, C, K)
    balance: float = 0.8,
    free_bits: float = 1.0,
    unimix: float = 0.01,
) -> Tensor:
    """
    KL loss with α-balancing and free-bits clipping.

    L_kl = α * KL[sg(q) || p] + (1-α) * KL[q || sg(p)]   [paper §App.B]

    'sg' = stop_gradient; free_bits clips each dimension below the threshold.
    """
    post_probs  = unimix_probs(post_logits,  unimix)   # (*, C, K)
    prior_probs = unimix_probs(prior_logits, unimix)

    # KL[q || p] per category  — sum over K, mean over C and batch dims
    def _kl(p_probs, q_probs):
        # Both (*, C, K); returns (*, C)
        return (p_probs * (p_probs.clamp(min=1e-8).log() - q_probs.clamp(min=1e-8).log())).sum(-1)

    # α * KL[sg(q) || p]
    kl_prior = _kl(post_probs.detach(), prior_probs)   # (*, C)
    # (1-α) * KL[q || sg(p)]
    kl_post  = _kl(post_probs, prior_probs.detach())   # (*, C)

    # Free bits: clip per-category KL below threshold  [paper §App.B]
    kl_prior = kl_prior.clamp(min=free_bits)
    kl_post  = kl_post.clamp(min=free_bits)

    loss = balance * kl_prior.mean() + (1.0 - balance) * kl_post.mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# World Model (top-level module)
# ─────────────────────────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Bundles encoder, RSSM, decoder, reward head, and continuation head.
    Exposes a single `loss()` method used in train.py.
    """
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CNNEncoder(embed_dim=cfg.rssm_embed)
        self.rssm = RSSM(cfg)
        latent_dim = cfg.rssm_hidden + cfg.rssm_categories * cfg.rssm_classes  # h + z
        self.decoder = CNNDecoder(latent_dim=latent_dim)
        self.reward_head = MLP(latent_dim, 1, cfg.mlp_hidden)
        self.cont_head   = MLP(latent_dim, 1, cfg.mlp_hidden)   # logit for p(continue)

    def features(self, h: Tensor, z: Tensor) -> Tensor:
        """Concatenate deterministic and stochastic states into feature vector."""
        return torch.cat([h, z], dim=-1)   # (*, H + C*K)

    def loss(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total world-model loss over a (T, B) batch.

        Returns (total_loss, metrics_dict).

        batch keys:
            'image'       (T, B, 3, 64, 64)  — uint8
            'action'      (T, B, A)
            'reward'      (T, B)
            'is_first'    (T, B)
            'is_terminal' (T, B)
        """
        T, B = batch['image'].shape[:2]
        device = batch['image'].device

        # ── Encode all images simultaneously ──────────────────────────────
        imgs = batch['image'].reshape(T * B, 3, self.cfg.image_size, self.cfg.image_size)
        embeds = self.encoder(imgs).reshape(T, B, -1)   # (T, B, E)

        # ── RSSM posterior rollout ─────────────────────────────────────────
        rssm_out = self.rssm.observe(embeds, batch['action'], batch['is_first'])
        h = rssm_out['h']   # (T, B, H)
        z = rssm_out['z']   # (T, B, C*K)
        feat = self.features(h, z)   # (T, B, H+C*K)

        # ── Reconstruction loss  (symlog MSE)   [paper §App.B Eq.~decoder] ──
        feat_flat = feat.reshape(T * B, -1)
        recon_logits = self.decoder(feat_flat).reshape(T, B, 3,
                                                        self.cfg.image_size,
                                                        self.cfg.image_size)
        target_symlog = symlog(imgs.float().reshape(T, B, 3,
                                                     self.cfg.image_size,
                                                     self.cfg.image_size))
        rec_loss = F.mse_loss(recon_logits, target_symlog)

        # ── Reward prediction loss   (symlog targets)   [paper §App.B] ──────
        rew_pred = self.reward_head(feat_flat).reshape(T, B)
        rew_target = symlog(batch['reward'])
        rew_loss = F.mse_loss(rew_pred, rew_target)

        # ── Continuation prediction loss   [paper §App.B] ───────────────────
        cont_logit = self.cont_head(feat_flat).reshape(T, B)
        cont_target = (1.0 - batch['is_terminal'].float())
        cont_loss = F.binary_cross_entropy_with_logits(cont_logit, cont_target)

        # ── KL loss   [paper §App.B] ─────────────────────────────────────────
        kl = kl_loss(
            rssm_out['post_logits'],   # (T, B, C, K)
            rssm_out['prior_logits'],
            balance=self.cfg.kl_balance,
            free_bits=self.cfg.kl_free,
            unimix=self.cfg.unimix,
        )

        # ── Total loss ───────────────────────────────────────────────────────
        total = (self.cfg.rec_scale  * rec_loss
               + self.cfg.rew_scale  * rew_loss
               + self.cfg.cont_scale * cont_loss
               + self.cfg.kl_scale   * kl)

        metrics = {
            'wm/rec_loss':  rec_loss.item(),
            'wm/rew_loss':  rew_loss.item(),
            'wm/cont_loss': cont_loss.item(),
            'wm/kl':        kl.item(),
            'wm/total':     total.item(),
        }
        return total, metrics, rssm_out
