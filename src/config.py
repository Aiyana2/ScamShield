"""Shared constants and model hyper-parameters for ScamShield v2.

Everything that has to agree between training, inference and the API lives here
(label spaces, feature dimensions, default teacher / student sizes), so a
checkpoint can be rebuilt from its saved config dict alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

# --------------------------------------------------------------------------- #
# Label spaces
# --------------------------------------------------------------------------- #
TAXONOMY: tuple[str, ...] = (
    "Benign",
    "Phishing",
    "Credential Harvesting",
    "Tech Support Impersonation",
    "Financial Coercion",
    "Crypto Fraud",
)
# BIO tagging for the span-extraction head (one suspicious-span type).
SPAN_TAGS: tuple[str, ...] = ("O", "B-SUSP", "I-SUSP")
MODALITIES: tuple[str, ...] = ("text", "acoustic", "url")
IGNORE_INDEX = -100

# --------------------------------------------------------------------------- #
# Feature dimensions
# --------------------------------------------------------------------------- #
TEXT_DIM = 768        # hidden size of a BERT/RoBERTa-base style encoder
URL_EMBED_DIM = 128   # Char-CNN output size
N_MFCC = 20

ACOUSTIC_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"mfcc_mean_{i}" for i in range(N_MFCC))
    + tuple(f"mfcc_std_{i}" for i in range(N_MFCC))
    + (
        "pitch_mean_hz",
        "pitch_std_hz",
        "voiced_ratio",
        "jitter_local",
        "shimmer_local",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_flatness_mean",
    )
)
ACOUSTIC_DIM = len(ACOUSTIC_FEATURE_NAMES)  # 50

URL_STAT_NAMES: tuple[str, ...] = (
    "log_length",
    "shannon_entropy",
    "digit_ratio",
    "special_ratio",
    "subdomain_depth",
    "has_ip_host",
    "non_ascii_ratio",
)
URL_STATS_DIM = len(URL_STAT_NAMES)


@dataclass
class ModelConfig:
    """Architecture hyper-parameters shared by the teacher and student nets."""

    text_dim: int = TEXT_DIM
    acoustic_dim: int = ACOUSTIC_DIM
    url_stats_dim: int = URL_STATS_DIM
    url_max_len: int = 256
    url_char_embed_dim: int = 48
    url_channels: int = 128
    url_kernel_sizes: tuple[int, ...] = (3, 5, 7)
    url_embed_dim: int = URL_EMBED_DIM
    d_model: int = 256
    n_heads: int = 4
    text_layers: int = 1          # contextual adapter layers on top of the frozen LM
    acoustic_tokens: int = 4      # acoustic vector is unfolded into k pseudo-tokens
    cross_attention: bool = True  # GMU with cross-modal attention (teacher) or plain GMU
    trunk_depth: int = 3
    trunk_hidden_mult: int = 2
    dropout: float = 0.2
    attn_dropout: float = 0.1
    num_classes: int = len(TAXONOMY)
    num_span_tags: int = len(SPAN_TAGS)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["url_kernel_sizes"] = list(self.url_kernel_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "url_kernel_sizes" in kwargs:
            kwargs["url_kernel_sizes"] = tuple(kwargs["url_kernel_sizes"])
        return cls(**kwargs)


def teacher_config() -> ModelConfig:
    """Heavy teacher: cross-attention GMU, contextual text adapter, deep trunk."""
    return ModelConfig(
        d_model=384,
        n_heads=6,
        text_layers=2,
        url_channels=192,
        url_kernel_sizes=(3, 5, 7),
        cross_attention=True,
        trunk_depth=4,
        dropout=0.2,
    )


def student_config() -> ModelConfig:
    """Compact student: plain sigmoid GMU (no cross-attention), shallow trunk."""
    return ModelConfig(
        d_model=128,
        n_heads=2,
        text_layers=0,
        url_char_embed_dim=32,
        url_channels=64,
        url_kernel_sizes=(3, 5),
        acoustic_tokens=2,
        cross_attention=False,
        trunk_depth=1,
        dropout=0.1,
        attn_dropout=0.0,
    )
