"""Text feature extractors that produce 768-d token hidden states + char offsets.

* :class:`HFTextEncoder` — a frozen Hugging Face encoder (default
  ``distilroberta-base``, 768-d). This is what you train and serve with.
* :class:`HashingTextEncoder` — a deterministic, download-free stand-in that
  maps each word to a fixed pseudo-random 768-d vector. It exists only so the
  test-suite and smoke tests run offline; it carries no semantics.

Both return character offsets per token so span labels written as character
ranges can be converted to token-level BIO tags, and predicted tags can be
mapped back to substrings of the original message.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np
import torch

from src.config import TEXT_DIM


@dataclass
class EncodedText:
    hidden: torch.Tensor                      # (B, T, 768) float32
    mask: torch.Tensor                        # (B, T) bool
    offsets: list[list[tuple[int, int]]]      # per token (start, end) in the source string; (0, 0) = special
    texts: list[str]


class HFTextEncoder:
    def __init__(self, model_name: str = "distilroberta-base", max_length: int = 128, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer  # lazy: heavy import

        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.model.config.hidden_size != TEXT_DIM:
            raise ValueError(f"{model_name} has hidden size {self.model.config.hidden_size}, expected {TEXT_DIM}")

    @torch.no_grad()
    def __call__(self, texts: list[str | None]) -> EncodedText:
        texts = [t or "" for t in texts]
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = [[tuple(map(int, o)) for o in row] for row in enc.pop("offset_mapping").tolist()]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state.float().cpu()
        mask = enc["attention_mask"].bool().cpu()
        # Rows that were empty strings carry only special tokens -> mark as no text.
        for b, t in enumerate(texts):
            if not t.strip():
                mask[b] = False
        return EncodedText(hidden=hidden, mask=mask, offsets=offsets, texts=texts)


class HashingTextEncoder:
    """Offline, deterministic word-hash embeddings. For tests only (no semantics)."""

    model_name = "hashing"
    _WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, max_length: int = 128, dim: int = TEXT_DIM):
        self.max_length, self.dim = max_length, dim
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, token: str) -> np.ndarray:
        if token not in self._cache:
            seed = int.from_bytes(hashlib.md5(token.lower().encode()).digest()[:4], "little")
            self._cache[token] = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return self._cache[token]

    def _position(self, n: int) -> np.ndarray:
        pos = np.arange(n)[:, None]
        i = np.arange(self.dim // 2)[None, :]
        angle = pos / np.power(10000.0, 2 * i / self.dim)
        pe = np.zeros((n, self.dim), dtype=np.float32)
        pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
        return 0.1 * pe

    def __call__(self, texts: list[str | None]) -> EncodedText:
        texts = [t or "" for t in texts]
        toks = [[(m.group(), m.start(), m.end()) for m in self._WORD.finditer(t)][: self.max_length] for t in texts]
        length = max(1, max((len(t) for t in toks), default=1))
        hidden = torch.zeros(len(texts), length, self.dim)
        mask = torch.zeros(len(texts), length, dtype=torch.bool)
        offsets: list[list[tuple[int, int]]] = []
        for b, row in enumerate(toks):
            offs = [(0, 0)] * length
            if row:
                vecs = np.stack([self._vec(w) for w, _, _ in row]) + self._position(len(row))
                hidden[b, : len(row)] = torch.from_numpy(vecs / math.sqrt(2.0))
                mask[b, : len(row)] = True
                offs[: len(row)] = [(s, e) for _, s, e in row]
            offsets.append(offs)
        return EncodedText(hidden=hidden, mask=mask, offsets=offsets, texts=texts)


def build_text_encoder(name: str, max_length: int = 128, device: str = "cpu"):
    if name == "hashing":
        return HashingTextEncoder(max_length=max_length)
    return HFTextEncoder(name, max_length=max_length, device=device)
