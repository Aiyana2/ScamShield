r"""Gated Multimodal Fusion Unit (GMU) with cross-modal attention.

Based on the GMU of Arevalo et al. (2017), extended to M = 3 modalities
(text, acoustic, URL), per-dimension gates, explicit missing-modality masking
and an optional cross-attention step before gating.

Notation (per sample; m indexes modalities, d = d_model)::

    x_m  in R^{d}            pooled modality vector (already projected to d)
    S_m  in R^{N_m x d}      token sequence of modality m (text tokens, URL chars,
                             acoustic pseudo-tokens)
    pi_m in {0, 1}           presence flag (0 if the modality is missing / dropped)

1. Candidate representation (GMU)::

       h_m = tanh(W_m x_m + b_m)

2. Cross-modal attention (teacher only, ``cross_attention=True``). Each
   modality queries the tokens of the *other* present modalities::

       K_m = V_m = [ S_j for j != m with pi_j = 1 ] ++ [ n ]
       c_m = MHA(q = h_m, K_m, V_m)
       h_m <- LayerNorm(h_m + Dropout(c_m))

   ``n`` is a learned null token that is never masked: it keeps softmax
   well-defined when every other modality is missing, and lets a modality
   "attend to nothing" when the others carry no useful evidence.

3. Gating. With g = [pi_1 h_1 ; pi_2 h_2 ; pi_3 h_3 ; pi] in R^{3d + 3}::

       z_m = sigmoid(W_z^{(m)} g + b_z^{(m)}) * pi_m        in (0, 1)^d
       f   = sum_m z_m ⊙ h_m

   Each z_m is an independent sigmoid (not a softmax over modalities), so the
   network can turn *all* modalities up or down; a noisy modality is
   suppressed by learning z_m -> 0 on the inputs where it hurts, and a missing
   modality is forced to exactly 0 by pi_m. ``gates`` (mean of z_m over d) is
   returned for monitoring and for the API response.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import MODALITIES


class GatedMultimodalFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        cross_attention: bool = True,
        modalities: tuple[str, ...] = MODALITIES,
    ):
        super().__init__()
        self.modalities = modalities
        self.cross_attention = cross_attention
        m = len(modalities)

        self.candidate = nn.ModuleDict({k: nn.Linear(d_model, d_model) for k in modalities})
        if cross_attention:
            self.attn = nn.ModuleDict(
                {
                    k: nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
                    for k in modalities
                }
            )
            self.attn_norm = nn.ModuleDict({k: nn.LayerNorm(d_model) for k in modalities})
            self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.null_token, std=0.02)
        self.attn_drop = nn.Dropout(dropout)
        self.gate = nn.Linear(m * d_model + m, m * d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        pooled: dict[str, torch.Tensor],
        tokens: dict[str, torch.Tensor],
        token_masks: dict[str, torch.Tensor],
        presence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        pooled[m]:      (B, d)
        tokens[m]:      (B, N_m, d)
        token_masks[m]: (B, N_m) bool, True = valid token
        presence:       (B, M) float in {0, 1}, column order = self.modalities

        Returns fused (B, d), gates (B, M), attention weights per modality.
        """
        bsz = presence.shape[0]
        h = {k: torch.tanh(self.candidate[k](pooled[k])) for k in self.modalities}
        attn_maps: dict[str, torch.Tensor] = {}

        if self.cross_attention:
            null = self.null_token.expand(bsz, -1, -1)
            null_mask = torch.ones(bsz, 1, dtype=torch.bool, device=presence.device)
            updated = {}
            for i, k in enumerate(self.modalities):
                kv, valid = [], []
                for j, other in enumerate(self.modalities):
                    if other == k:
                        continue
                    kv.append(tokens[other])
                    valid.append(token_masks[other] & (presence[:, j : j + 1] > 0))
                kv.append(null)
                valid.append(null_mask)
                kv_t = torch.cat(kv, dim=1)
                key_padding_mask = ~torch.cat(valid, dim=1)  # True = ignore
                ctx, weights = self.attn[k](
                    h[k].unsqueeze(1), kv_t, kv_t, key_padding_mask=key_padding_mask, need_weights=True
                )
                updated[k] = self.attn_norm[k](h[k] + self.attn_drop(ctx.squeeze(1)))
                attn_maps[k] = weights.squeeze(1)
            h = updated

        stacked = torch.stack([h[k] for k in self.modalities], dim=1)        # (B, M, d)
        pres = presence.unsqueeze(-1)                                         # (B, M, 1)
        gate_in = torch.cat([(stacked * pres).flatten(1), presence], dim=-1)  # (B, M*d + M)
        z = torch.sigmoid(self.gate(gate_in)).view_as(stacked) * pres         # (B, M, d)
        fused = self.out_norm((z * stacked).sum(dim=1))
        gates = z.mean(dim=-1)                                                # (B, M)
        return fused, gates, attn_maps
