"""Multi-task, multimodal ScamShield network: shared trunk + three heads.

Data flow::

    text hidden states (B,T,768) ─ Linear+LN ─ [TransformerEncoder x text_layers] ─┐ tokens
                                         └─ masked attention pooling ─────────────┐ pooled
    URL char ids (B,L) ─ CharCNNURLEncoder ─ (B,128) pooled / (B,L,128) chars ─ Linear ─┤
    acoustic feats (B,50) ─ standardise ─ MLP ─ (B,d) pooled / k pseudo-tokens ────────┤
                                                                                     v
                                     GatedMultimodalFusion ─ fused (B,d), gates (B,3)
                                                                                     v
                         Shared trunk: depth x pre-LN residual MLP blocks (with dropout)
                    ┌───────────────────────────────┼──────────────────────────────┐
             Head 1: risk logit (B,)     Head 2: taxonomy logits (B,C)    Head 3: span logits
                                                                          (B, T+L, 3) on
                                                                          [token state ; trunk]

The text transformer itself (e.g. RoBERTa-base) is kept frozen and outside
this module (see ``src/features/text_encoder.py``); the network consumes its
768-d hidden states. That keeps the model trainable on a CPU and lets the
teacher and the student share the same cached text features during
distillation.

Dropout is placed in the trunk, heads and adapters so that the same model can
be sampled with Monte Carlo dropout at inference time
(``src/inference/uncertainty.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import MODALITIES, ModelConfig
from src.models.char_cnn_url import URL_VOCAB_SIZE, CharCNNURLEncoder
from src.models.gated_fusion import GatedMultimodalFusion


@dataclass
class MultimodalInputs:
    """One padded batch. Missing modalities are zero tensors with presence 0."""

    text_hidden: torch.Tensor  # (B, T, text_dim) float
    text_mask: torch.Tensor    # (B, T) bool
    url_ids: torch.Tensor      # (B, L) long
    url_mask: torch.Tensor     # (B, L) bool
    url_stats: torch.Tensor    # (B, URL_STATS_DIM) float
    acoustic: torch.Tensor     # (B, ACOUSTIC_DIM) float, raw (un-standardised)
    presence: torch.Tensor     # (B, 3) float, columns follow config.MODALITIES

    def to(self, device: torch.device | str) -> "MultimodalInputs":
        return MultimodalInputs(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def index_select(self, idx: torch.Tensor) -> "MultimodalInputs":
        return MultimodalInputs(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def repeat(self, n: int) -> "MultimodalInputs":
        """Tile the batch n times along dim 0 (sample-major: [b0..bB, b0..bB, ...])."""
        def rep(t: torch.Tensor) -> torch.Tensor:
            return t.repeat(n, *([1] * (t.dim() - 1)))
        return MultimodalInputs(**{f.name: rep(getattr(self, f.name)) for f in fields(self)})

    @property
    def batch_size(self) -> int:
        return self.presence.shape[0]


@dataclass
class MultiTaskOutput:
    risk_logit: torch.Tensor        # (B,)
    taxonomy_logits: torch.Tensor   # (B, C)
    span_logits: torch.Tensor       # (B, T + L, num_span_tags); text tokens first, then URL chars
    span_mask: torch.Tensor         # (B, T + L) bool
    gates: torch.Tensor             # (B, 3)
    text_len: int                   # T, to split span_logits back into text / URL parts
    embedding: torch.Tensor         # (B, d) trunk output, useful for OOD / retrieval


class ResidualMLPBlock(nn.Module):
    """Pre-LN residual block: x + Drop(W2 Drop(GELU(W1 LN(x))))."""

    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.norm(x))))))


class MaskedAttentionPool(nn.Module):
    """softmax(q·h_t / sqrt(d)) weighted average over valid tokens."""

    def __init__(self, d: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.scale = d ** -0.5

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (h @ self.query) * self.scale                     # (B, T)
        scores = scores.masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=-1) * mask.to(h.dtype)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
        return (w.unsqueeze(-1) * h).sum(1)


def _head(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out))


class MultiTaskScamNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        d = cfg.d_model

        # ---- text branch ------------------------------------------------- #
        self.text_proj = nn.Sequential(nn.Linear(cfg.text_dim, d), nn.LayerNorm(d), nn.Dropout(cfg.dropout))
        if cfg.text_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d, cfg.n_heads, dim_feedforward=2 * d, dropout=cfg.attn_dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.text_ctx: nn.Module | None = nn.TransformerEncoder(
                layer, cfg.text_layers, enable_nested_tensor=False
            )
        else:
            self.text_ctx = None
        self.text_pool = MaskedAttentionPool(d)

        # ---- URL branch -------------------------------------------------- #
        self.url_encoder = CharCNNURLEncoder(
            vocab_size=URL_VOCAB_SIZE,
            char_embed_dim=cfg.url_char_embed_dim,
            channels=cfg.url_channels,
            kernel_sizes=cfg.url_kernel_sizes,
            out_dim=cfg.url_embed_dim,
            stats_dim=cfg.url_stats_dim,
            dropout=cfg.dropout,
        )
        self.url_proj = nn.Linear(cfg.url_embed_dim, d)

        # ---- acoustic branch --------------------------------------------- #
        # Standardisation stats are buffers so they travel with the checkpoint.
        self.register_buffer("acoustic_mean", torch.zeros(cfg.acoustic_dim))
        self.register_buffer("acoustic_std", torch.ones(cfg.acoustic_dim))
        self.acoustic_mlp = nn.Sequential(
            nn.Linear(cfg.acoustic_dim, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.acoustic_tokens = nn.Linear(d, cfg.acoustic_tokens * d)

        # ---- fusion, trunk, heads ---------------------------------------- #
        self.fusion = GatedMultimodalFusion(
            d, cfg.n_heads, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
            cross_attention=cfg.cross_attention,
        )
        self.trunk = nn.Sequential(*[ResidualMLPBlock(d, cfg.trunk_hidden_mult, cfg.dropout) for _ in range(cfg.trunk_depth)])
        self.trunk_norm = nn.LayerNorm(d)
        self.risk_head = _head(d, d // 2, 1, cfg.dropout)
        self.taxonomy_head = _head(d, d // 2, cfg.num_classes, cfg.dropout)
        self.span_head = _head(2 * d, d, cfg.num_span_tags, cfg.dropout)

    # ------------------------------------------------------------------ #
    def set_acoustic_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.acoustic_mean.copy_(mean)
        self.acoustic_std.copy_(std.clamp(min=1e-6))

    def standardize_acoustic(self, acoustic: torch.Tensor) -> torch.Tensor:
        return (acoustic - self.acoustic_mean) / self.acoustic_std

    def forward(self, x: MultimodalInputs) -> MultiTaskOutput:
        url_emb = self.url_encoder.embed(x.url_ids)
        return self.forward_from_embeddings(
            x.text_hidden, url_emb, x.acoustic,
            text_mask=x.text_mask, url_mask=x.url_mask, url_stats=x.url_stats, presence=x.presence,
        )

    def forward_from_embeddings(
        self,
        text_hidden: torch.Tensor,
        url_char_emb: torch.Tensor,
        acoustic: torch.Tensor,
        *,
        text_mask: torch.Tensor,
        url_mask: torch.Tensor,
        url_stats: torch.Tensor,
        presence: torch.Tensor,
    ) -> MultiTaskOutput:
        """Forward pass from continuous inputs (used directly by Integrated Gradients)."""
        pres = {k: presence[:, i : i + 1] for i, k in enumerate(MODALITIES)}

        # text
        t = self.text_proj(text_hidden)
        if self.text_ctx is not None:
            # A row with no valid tokens would make attention NaN; keep position 0 visible.
            safe_mask = text_mask.clone()
            safe_mask[:, 0] = True
            t = self.text_ctx(t, src_key_padding_mask=~safe_mask)
        t = t * text_mask.unsqueeze(-1).to(t.dtype)
        text_pooled = self.text_pool(t, text_mask) * pres["text"]

        # URL
        url_pooled, url_chars = self.url_encoder.forward_embeddings(url_char_emb, url_mask, url_stats)
        u_tokens = self.url_proj(url_chars)
        url_pooled = self.url_proj(url_pooled) * pres["url"]

        # acoustic
        a = self.acoustic_mlp(self.standardize_acoustic(acoustic)) * pres["acoustic"]
        bsz, d = a.shape
        a_tokens = self.acoustic_tokens(a).view(bsz, self.cfg.acoustic_tokens, d)
        a_mask = torch.ones(bsz, self.cfg.acoustic_tokens, dtype=torch.bool, device=a.device)

        fused, gates, _ = self.fusion(
            pooled={"text": text_pooled, "acoustic": a, "url": url_pooled},
            tokens={"text": t, "acoustic": a_tokens, "url": u_tokens},
            token_masks={"text": text_mask, "acoustic": a_mask, "url": url_mask},
            presence=presence,
        )
        z = self.trunk_norm(self.trunk(fused))

        token_states = torch.cat([t, u_tokens], dim=1)                       # (B, T+L, d)
        ctx = z.unsqueeze(1).expand(-1, token_states.shape[1], -1)
        span_logits = self.span_head(torch.cat([token_states, ctx], dim=-1))
        span_mask = torch.cat(
            [text_mask & (pres["text"] > 0), url_mask & (pres["url"] > 0)], dim=1
        )
        return MultiTaskOutput(
            risk_logit=self.risk_head(z).squeeze(-1),
            taxonomy_logits=self.taxonomy_head(z),
            span_logits=span_logits,
            span_mask=span_mask,
            gates=gates,
            text_len=text_hidden.shape[1],
            embedding=z,
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
