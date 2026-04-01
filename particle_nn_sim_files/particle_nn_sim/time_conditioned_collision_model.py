from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn


@dataclass
class TimeEncodingConfig:
    """Configuration for Fourier time encoding."""

    num_frequencies: int = 8
    include_raw_time: bool = True
    base_frequency: float = 1.0


class FourierTimeEncoder(nn.Module):
    """
    Fourier-feature time encoder:
        gamma(t) = [t, sin(2*pi*f_k*t), cos(2*pi*f_k*t)]_k
    with dyadic frequencies f_k = base_frequency * 2^k.
    """

    def __init__(self, cfg: TimeEncodingConfig):
        super().__init__()
        if cfg.num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        self.cfg = cfg
        freqs = [cfg.base_frequency * (2.0**k) for k in range(cfg.num_frequencies)]
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32), persistent=False)

    @property
    def out_dim(self) -> int:
        base = 1 if self.cfg.include_raw_time else 0
        return base + 2 * self.cfg.num_frequencies

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (B,) or (B,1), scalar time in seconds.
        Returns:
            gamma(t): shape (B, out_dim)
        """
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"t must be shape (B,) or (B,1), got {tuple(t.shape)}")

        # (B, K)
        wt = 2.0 * math.pi * t * self.freqs.unsqueeze(0)
        sin_part = torch.sin(wt)
        cos_part = torch.cos(wt)
        parts = []
        if self.cfg.include_raw_time:
            parts.append(t)
        parts.extend([sin_part, cos_part])
        return torch.cat(parts, dim=1)


@dataclass
class TimeConditionedCollisionModelConfig:
    """Model hyperparameters."""

    state_dim: int = 4  # [x0, y0, vx0, vy0]
    trunk_width: int = 256
    trunk_depth: int = 3
    activation: str = "gelu"  # {"gelu", "silu"}
    dropout: float = 0.0
    alpha_gate: float = 5.0
    time_encoding: TimeEncodingConfig = field(default_factory=TimeEncodingConfig)


def _make_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class TimeConditionedCollisionModel(nn.Module):
    """
    Pure-ML time-conditioned model with event-gated velocity:
        f_theta(s0, gamma(t)) -> [x(t), y(t), v_pre, v_post, z_event]

    Final velocity is a learned convex combination:
        gate = sigmoid(alpha_gate * z_event)
        v_pred = (1-gate)*v_pre + gate*v_post

    No wall-reflection physics is hard-coded; gating is fully learned.
    """

    def __init__(self, cfg: TimeConditionedCollisionModelConfig):
        super().__init__()
        self.cfg = cfg
        self.time_encoder = FourierTimeEncoder(cfg.time_encoding)

        in_dim = cfg.state_dim + self.time_encoder.out_dim
        width = cfg.trunk_width
        act = _make_activation(cfg.activation)
        drop = float(cfg.dropout)

        trunk_layers: list[nn.Module] = [nn.Linear(in_dim, width), act]
        if drop > 0.0:
            trunk_layers.append(nn.Dropout(drop))
        for _ in range(max(0, cfg.trunk_depth - 1)):
            trunk_layers.extend([nn.Linear(width, width), act])
            if drop > 0.0:
                trunk_layers.append(nn.Dropout(drop))
        self.trunk = nn.Sequential(*trunk_layers)

        self.pos_head = nn.Linear(width, 2)
        self.vel_pre_head = nn.Linear(width, 2)
        self.vel_post_head = nn.Linear(width, 2)
        self.event_head = nn.Linear(width, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s0: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            s0: (B,4), initial state
            t:  (B,) or (B,1), queried time in seconds
        Returns dict with:
            pos: (B,2)
            v_pre: (B,2)
            v_post: (B,2)
            event_logit: (B,1)
            gate: (B,1)
            state: (B,4) => [x(t), y(t), vx(t), vy(t)]
        """
        if s0.ndim != 2 or s0.shape[1] != 4:
            raise ValueError(f"s0 must be shape (B,4), got {tuple(s0.shape)}")
        gamma_t = self.time_encoder(t)
        x = torch.cat([s0, gamma_t], dim=1)
        h = self.trunk(x)
        pos = self.pos_head(h)
        v_pre = self.vel_pre_head(h)
        v_post = self.vel_post_head(h)
        event_logit = self.event_head(h)
        gate = torch.sigmoid(float(self.cfg.alpha_gate) * event_logit)
        v_pred = (1.0 - gate) * v_pre + gate * v_post
        state_pred = torch.cat([pos, v_pred], dim=1)
        return {
            "pos": pos,
            "v_pre": v_pre,
            "v_post": v_post,
            "event_logit": event_logit,
            "gate": gate,
            "state": state_pred,
        }
