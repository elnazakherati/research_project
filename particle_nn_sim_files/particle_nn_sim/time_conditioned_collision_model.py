from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn


@dataclass
class TimeEncodingConfig:
    """Configuration for time encoding."""

    mode: str = "raw"  # {"raw", "low_freq_fourier", "fourier"}
    num_frequencies: int = 8
    include_raw_time: bool = True
    base_frequency: float = 1.0
    normalize_time: bool = True
    max_time: float = 1.0


class TimeEncoder(nn.Module):
    """
    Time encoder with three modes:
      raw: [t]
      low_freq_fourier: [t, sin/cos(2*pi*f_k*t)] with low linear frequencies
      fourier: [t, sin/cos(2*pi*f_k*t)] with dyadic frequencies

    All modes are pure learned feature preprocessing; no physics is hard-coded.
    """

    def __init__(self, cfg: TimeEncodingConfig):
        super().__init__()
        if cfg.max_time <= 0.0:
            raise ValueError("max_time must be > 0")
        if cfg.mode not in {"raw", "low_freq_fourier", "fourier"}:
            raise ValueError(f"Unsupported time encoding mode: {cfg.mode}")
        if cfg.mode != "raw" and cfg.num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1 for Fourier modes")
        self.cfg = cfg
        if cfg.mode == "fourier":
            freqs = [cfg.base_frequency * (2.0**k) for k in range(cfg.num_frequencies)]
        elif cfg.mode == "low_freq_fourier":
            freqs = [cfg.base_frequency * float(k + 1) for k in range(cfg.num_frequencies)]
        else:
            freqs = []
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32), persistent=False)

    @property
    def out_dim(self) -> int:
        if self.cfg.mode == "raw":
            return 1
        base = 1 if self.cfg.include_raw_time else 0
        return base + 2 * int(self.cfg.num_frequencies)

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

        t_enc = t / float(self.cfg.max_time) if self.cfg.normalize_time else t
        if self.cfg.mode == "raw":
            return t_enc

        wt = 2.0 * math.pi * t_enc * self.freqs.unsqueeze(0)
        sin_part = torch.sin(wt)
        cos_part = torch.cos(wt)
        parts = []
        if self.cfg.include_raw_time:
            parts.append(t_enc)
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
    model_variant: str = "simple_tcno"  # {"simple_tcno", "gated_tcno"}
    alpha_gate: float = 5.0
    enforce_t0_anchor: bool = True
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
    Pure-ML time-conditioned model supporting two variants:
      simple_tcno: position + velocity + event heads (no gating)
      gated_tcno:  position + pre/post velocity + event heads with learned gate
    """

    def __init__(self, cfg: TimeConditionedCollisionModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.model_variant not in {"simple_tcno", "gated_tcno"}:
            raise ValueError(f"Unsupported model_variant: {cfg.model_variant}")
        self.time_encoder = TimeEncoder(cfg.time_encoding)

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
        if cfg.model_variant == "gated_tcno":
            self.vel_pre_head = nn.Linear(width, 2)
            self.vel_post_head = nn.Linear(width, 2)
            self.vel_head = None
        else:
            self.vel_head = nn.Linear(width, 2)
            self.vel_pre_head = None
            self.vel_post_head = None
        self.event_head = nn.Linear(width, 1)

        # Optional affine map for converting input-space s0 into output-space anchor.
        # This is useful when input and output states are normalized with different
        # standardizers: state_anchor = s0 * scale + bias.
        self.register_buffer("s0_anchor_scale", torch.ones(1, 4), persistent=False)
        self.register_buffer("s0_anchor_bias", torch.zeros(1, 4), persistent=False)

        self._init_weights()

    def set_s0_anchor_affine(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Set affine mapping from model input s0-space to model output state-space.

        Args:
            scale: shape (4,) or (1,4)
            bias:  shape (4,) or (1,4)
        """
        scale = scale.reshape(1, 4).to(dtype=self.s0_anchor_scale.dtype, device=self.s0_anchor_scale.device)
        bias = bias.reshape(1, 4).to(dtype=self.s0_anchor_bias.dtype, device=self.s0_anchor_bias.device)
        self.s0_anchor_scale.copy_(scale)
        self.s0_anchor_bias.copy_(bias)

    def output_anchor_from_s0(self, s0: torch.Tensor) -> torch.Tensor:
        """Map input-space s0 to output-space anchor state."""
        return s0 * self.s0_anchor_scale + self.s0_anchor_bias

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
            vel: (B,2) for simple_tcno
            v_pre: (B,2)
            v_post: (B,2)
            event_logit: (B,1)
            gate: (B,1), zeros for simple_tcno
            state: (B,4) => [x(t), y(t), vx(t), vy(t)]
        """
        if s0.ndim != 2 or s0.shape[1] != 4:
            raise ValueError(f"s0 must be shape (B,4), got {tuple(s0.shape)}")
        if t.ndim == 1:
            t_col = t.unsqueeze(1)
        elif t.ndim == 2 and t.shape[1] == 1:
            t_col = t
        else:
            raise ValueError(f"t must be shape (B,) or (B,1), got {tuple(t.shape)}")

        gamma_t = self.time_encoder(t_col)
        x = torch.cat([s0, gamma_t], dim=1)
        h = self.trunk(x)
        pos = self.pos_head(h)
        event_logit = self.event_head(h)

        if self.cfg.model_variant == "gated_tcno":
            assert self.vel_pre_head is not None and self.vel_post_head is not None
            v_pre = self.vel_pre_head(h)
            v_post = self.vel_post_head(h)
            gate = torch.sigmoid(float(self.cfg.alpha_gate) * event_logit)
            v_pred = (1.0 - gate) * v_pre + gate * v_post
            vel = v_pred
        else:
            assert self.vel_head is not None
            vel = self.vel_head(h)
            v_pre = vel
            v_post = vel
            gate = torch.zeros_like(event_logit)
            v_pred = vel

        state_delta = torch.cat([pos, v_pred], dim=1)
        if self.cfg.enforce_t0_anchor:
            # Reparameterization: s_hat(t) = s0_anchor + t * h_theta(s0, t)
            # This guarantees exact initial-state matching at t=0.
            s0_anchor = self.output_anchor_from_s0(s0)
            state_pred = s0_anchor + t_col * state_delta
        else:
            state_pred = state_delta
        return {
            "pos": pos,
            "vel": vel,
            "v_pre": v_pre,
            "v_post": v_post,
            "event_logit": event_logit,
            "gate": gate,
            "state": state_pred,
        }
