"""Character-level CNN encoder for raw URL strings.

Why characters and not words: phishing URLs are built to *look* like a trusted
domain (``paypa1-secure.com``, ``micros0ft-support.net``, ``аpple.com`` with a
Cyrillic "а"). A word tokenizer maps those to UNK or to the same pieces as the
real brand; a character model sees the exact substitution.

Pipeline::

    url -> char ids (B, L) -> Embedding (B, L, E)
        -> K parallel branches, kernel k in {3,5,7}:
              Conv1d(k) -> BN -> GELU -> MaxPool1d(3, stride 1) -> Conv1d(k) -> BN -> GELU
        -> concat (B, K*C, L) -> 1x1 Conv -> BN -> GELU   = per-character features (B, L, 128)
        -> [masked global max-pool ; masked mean-pool ; lexical stats] -> Linear -> LayerNorm
        = URL embedding (B, 128)

The per-character features are kept (not only the pooled vector) because the
span head tags suspicious URL characters / sub-domains with them.
Kernel widths 3/5/7 act as learned n-gram detectors (typosquat substitutions,
``-secure-``, ``login.``), while global max-pooling makes the embedding
position-invariant. Lexical statistics (Shannon entropy, digit ratio, ...) are
concatenated before the projection so that randomly generated (DGA-like)
domains are easy to separate even with little data.
"""
from __future__ import annotations

import ipaddress
import math
import string
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlsplit

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import URL_STATS_DIM

PAD_ID, UNK_ID, NON_ASCII_ID = 0, 1, 2
# Case is preserved on purpose: "PayPaI" (capital i) vs "PayPal" is a signal.
URL_ALPHABET = string.ascii_letters + string.digits + string.punctuation
_CHAR_TO_ID = {c: i + 3 for i, c in enumerate(URL_ALPHABET)}
URL_VOCAB_SIZE = len(URL_ALPHABET) + 3


def shannon_entropy(s: str) -> float:
    """H(s) = -sum_c p(c) log2 p(c) over the character distribution of s."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def url_lexical_stats(url: str) -> list[float]:
    """Seven cheap, roughly unit-scaled lexical statistics for one URL."""
    if not url:
        return [0.0] * URL_STATS_DIM
    n = len(url)
    parts = urlsplit(url if "://" in url else "http://" + url)
    host = parts.hostname or ""
    try:
        ipaddress.ip_address(host)
        has_ip = 1.0
    except ValueError:
        has_ip = 0.0
    stats = [
        math.log1p(n) / 6.0,
        shannon_entropy(url) / 6.0,
        sum(ch.isdigit() for ch in url) / n,
        sum(ch in string.punctuation for ch in url) / n,
        min(max(host.count(".") - 1, 0), 10) / 5.0,
        has_ip,
        sum(ord(ch) > 127 for ch in url) / n,
    ]
    assert len(stats) == URL_STATS_DIM
    return stats


@dataclass
class EncodedURLs:
    ids: torch.Tensor    # (B, L) long
    mask: torch.Tensor   # (B, L) bool, True = real character
    stats: torch.Tensor  # (B, URL_STATS_DIM) float
    urls: list[str]      # truncated strings; position i in ids == urls[b][i]


class URLCharTokenizer:
    """Maps URLs to fixed-alphabet character ids. Char position == string index."""

    def __init__(self, max_len: int = 256):
        self.max_len = max_len
        self.vocab_size = URL_VOCAB_SIZE

    def char_id(self, ch: str) -> int:
        if ord(ch) > 127:
            return NON_ASCII_ID  # IDN homoglyphs get their own token, not UNK
        return _CHAR_TO_ID.get(ch, UNK_ID)

    def __call__(self, urls: list[str | None]) -> EncodedURLs:
        cleaned = [(u or "").strip()[: self.max_len] for u in urls]
        length = max(1, max((len(u) for u in cleaned), default=1))
        ids = torch.full((len(cleaned), length), PAD_ID, dtype=torch.long)
        for b, u in enumerate(cleaned):
            if u:
                ids[b, : len(u)] = torch.tensor([self.char_id(c) for c in u], dtype=torch.long)
        stats = torch.tensor([url_lexical_stats(u) for u in cleaned], dtype=torch.float32)
        return EncodedURLs(ids=ids, mask=ids != PAD_ID, stats=stats, urls=cleaned)


class _ConvBranch(nn.Module):
    """Two same-padded convolutions of width k with a stride-1 max-pool in between.

    Stride-1 pooling keeps the sequence length (needed for per-character span
    tagging) while still giving each position a max over its local window,
    which makes the detector tolerant to a one-character shift.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L); mask: (B, 1, L) float
        h = F.gelu(self.bn1(self.conv1(x))) * mask
        # Padding positions must not win the max-pool.
        h = self.pool(h.masked_fill(mask == 0, -1e4)).masked_fill(mask == 0, 0.0)
        return F.gelu(self.bn2(self.conv2(h))) * mask


class CharCNNURLEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = URL_VOCAB_SIZE,
        char_embed_dim: int = 48,
        channels: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        out_dim: int = 128,
        stats_dim: int = URL_STATS_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=PAD_ID)
        self.branches = nn.ModuleList(_ConvBranch(char_embed_dim, channels, k) for k in kernel_sizes)
        self.mix = nn.Sequential(
            nn.Conv1d(channels * len(kernel_sizes), out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(2 * out_dim + stats_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)

    def forward_embeddings(
        self, emb: torch.Tensor, mask: torch.Tensor, stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode from character embeddings (B, L, E).

        Split out from :meth:`forward` so attribution methods can integrate
        gradients in the continuous embedding space.

        Returns (pooled (B, out_dim), char_features (B, L, out_dim)).
        """
        m = mask.unsqueeze(1).to(emb.dtype)                   # (B, 1, L)
        x = emb.transpose(1, 2) * m                           # (B, E, L)
        h = torch.cat([branch(x, m) for branch in self.branches], dim=1)
        h = self.mix(h) * m                                   # (B, D, L)
        char_features = self.dropout(h.transpose(1, 2))       # (B, L, D)

        # Masked global pooling. Rows without a URL pool to 0.
        lengths = m.sum(-1).clamp(min=1.0)                    # (B, 1)
        mean_pool = h.sum(-1) / lengths
        max_pool = h.masked_fill(m == 0, -1e4).amax(-1)
        has_url = (mask.sum(-1, keepdim=True) > 0).to(emb.dtype)
        max_pool = max_pool * has_url
        pooled = self.project(torch.cat([max_pool, mean_pool, stats], dim=-1)) * has_url
        return pooled, char_features

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embed(ids), mask, stats)
