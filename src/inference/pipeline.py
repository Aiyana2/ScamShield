"""End-to-end multimodal inference.

    raw inputs (text, URL, audio file / acoustic vector)
      -> [audio] Librosa acoustic features (+ optional Whisper transcript if no text given)
      -> [text]  frozen transformer hidden states + char offsets
      -> [url]   char ids + lexical stats (URL auto-extracted from text if not given)
      -> MC-dropout passes of the multi-task net
      -> risk probability + epistemic / aleatoric decomposition + OOD flag
      -> taxonomy distribution, suspicious spans (text substrings / URL substrings)
      -> optional Integrated-Gradients attributions (tokens, URL segments, acoustic features)

The pipeline never fetches URLs; analysis is purely lexical.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.config import ACOUSTIC_DIM, ACOUSTIC_FEATURE_NAMES, MODALITIES, SPAN_TAGS, TAXONOMY
from src.explainability.attributions import MultimodalAttributor
from src.inference.uncertainty import decompose_uncertainty, mc_dropout_predict
from src.models.char_cnn_url import URLCharTokenizer
from src.models.multitask_net import MultimodalInputs, MultiTaskScamNet

URL_IN_TEXT = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
URL_SEGMENT = re.compile(r"[^./:?=&#\-_]+")
TAG_O = SPAN_TAGS.index("O")


def extract_first_url(text: str | None) -> str | None:
    if not text:
        return None
    m = URL_IN_TEXT.search(text)
    return m.group(0).rstrip(".,;!") if m else None


def whisper_transcriber(model_name: str = "base") -> Callable[[str], str]:
    """Lazy OpenAI-Whisper transcriber (needs `openai-whisper` and ffmpeg)."""
    import whisper

    model = whisper.load_model(model_name)
    return lambda path: model.transcribe(str(path), fp16=False)["text"].strip()


def _decode_spans(tag_probs: np.ndarray, offsets: list[tuple[int, int]], source: str) -> list[dict]:
    """Merge contiguous non-O tokens (a B tag starts a new span) into char ranges."""
    spans: list[dict] = []
    current: dict | None = None
    for i, (s, e) in enumerate(offsets):
        if s == e or i >= len(tag_probs):
            current = None
            continue
        tag = int(tag_probs[i].argmax())
        score = float(1.0 - tag_probs[i][TAG_O])
        if tag == TAG_O:
            current = None
            continue
        if current is None or SPAN_TAGS[tag].startswith("B") or s > current["end"] + 1:
            current = {"start": s, "end": e, "scores": [score]}
            spans.append(current)
        else:
            current["end"] = e
            current["scores"].append(score)
    return [
        {"text": source[sp["start"] : sp["end"]], "start": sp["start"], "end": sp["end"],
         "score": round(float(np.mean(sp["scores"])), 4)}
        for sp in spans
    ]


class ScamShieldPipeline:
    def __init__(self, model: MultiTaskScamNet, text_encoder, ood_threshold: float | None = None,
                 device: str = "cpu", transcriber: Callable[[str], str] | None = None,
                 risk_threshold: float = 0.5):
        self.model = model.to(device).eval()
        self.text_encoder = text_encoder
        self.url_tokenizer = URLCharTokenizer(max_len=model.cfg.url_max_len)
        self.ood_threshold = ood_threshold
        self.device = device
        self.transcriber = transcriber
        self.risk_threshold = risk_threshold
        self.attributor = MultimodalAttributor(model)
        # A checkpoint trained without any audio keeps the default 0/1 acoustic stats and an
        # untrained acoustic branch; feeding it voice features gives the same score for every call.
        self.acoustic_trained = not (bool((model.acoustic_mean == 0).all()) and bool((model.acoustic_std == 1).all()))

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", text_encoder=None,
                        whisper_model: str | None = None) -> "ScamShieldPipeline":
        from src.features.text_encoder import build_text_encoder
        from src.training.train_distill import load_checkpoint

        model, ckpt = load_checkpoint(path, device)
        encoder = text_encoder or build_text_encoder(ckpt["text_encoder"], device=device)
        transcriber = whisper_transcriber(whisper_model) if whisper_model else None
        pipe = cls(model, encoder, ckpt.get("ood_threshold"), device, transcriber)
        pipe.checkpoint_kind = ckpt.get("kind")
        return pipe

    # ------------------------------------------------------------------ #
    def build_inputs(self, texts: list[str | None], urls: list[str | None],
                     acoustic: list[np.ndarray | None]) -> tuple[MultimodalInputs, list, list[str]]:
        enc = self.text_encoder(texts)
        u = self.url_tokenizer(urls)
        bsz = len(texts)
        ac = torch.zeros(bsz, ACOUSTIC_DIM)
        for b, a in enumerate(acoustic):
            if a is not None:
                ac[b] = torch.as_tensor(np.asarray(a, dtype=np.float32))
        presence = torch.tensor([
            [float(bool(enc.mask[b].any())), float(acoustic[b] is not None), float(bool(u.urls[b]))]
            for b in range(bsz)
        ])
        x = MultimodalInputs(text_hidden=enc.hidden, text_mask=enc.mask, url_ids=u.ids, url_mask=u.mask,
                             url_stats=u.stats, acoustic=ac, presence=presence)
        return x.to(self.device), enc.offsets, u.urls

    def predict(self, text: str | None = None, url: str | None = None, audio_path: str | Path | None = None,
                acoustic: np.ndarray | None = None, mc_samples: int = 30, explain: bool = False,
                top_k: int = 8) -> dict[str, Any]:
        transcript = None
        if audio_path is not None:
            from src.features.acoustic import extract_acoustic_features

            acoustic = extract_acoustic_features(audio_path)
            if not text and self.transcriber is not None:
                transcript = text = self.transcriber(str(audio_path))
        warnings = []
        if acoustic is not None and not self.acoustic_trained:
            acoustic = None
            warnings.append("voice features ignored: this checkpoint was trained without audio")
        url = url or extract_first_url(text)
        if not (text or url or acoustic is not None):
            if audio_path is not None and self.transcriber is not None:
                raise ValueError("Whisper is on but heard no speech in this recording, so there is nothing to "
                                 "analyse: check the file has clear spoken words, or paste the call text")
            if audio_path is not None:
                raise ValueError("Whisper is off on this backend, so the recording can't be transcribed: restart "
                                 "the backend with SCAMSHIELD_WHISPER=base, or paste the call text")
            raise ValueError("provide at least one of text, url, audio")
        result = self.predict_batch([text], [url], [acoustic], mc_samples, explain, top_k, [transcript])[0]
        if warnings:
            result["warnings"] = warnings
        return result

    @torch.no_grad()
    def predict_batch(self, texts, urls, acoustic, mc_samples: int = 30, explain: bool = False,
                      top_k: int = 8, transcripts: list[str | None] | None = None) -> list[dict[str, Any]]:
        x, offsets, url_strs = self.build_inputs(texts, urls, acoustic)
        samples = mc_dropout_predict(self.model, x, max(1, mc_samples))
        rep = decompose_uncertainty(samples)
        attr = None
        if explain:
            with torch.enable_grad():
                attr = self.attributor.attribute(x, "risk")

        results = []
        t_len = samples.text_len
        for b in range(x.batch_size):
            risk = float(rep.risk_mean[b])
            ood_score = float(rep.ood_score[b])
            is_ood = self.ood_threshold is not None and ood_score > self.ood_threshold
            tax = rep.taxonomy_mean[b].cpu().numpy()
            order = np.argsort(-tax)
            span_probs = samples.span_mean[b].cpu().numpy()
            text_src = texts[b] or ""
            result: dict[str, Any] = {
                "risk_score": round(risk, 4),
                "verdict": "scam" if risk >= self.risk_threshold else "safe",
                "needs_review": bool(is_ood),
                "category": TAXONOMY[int(order[0])],
                "taxonomy": {TAXONOMY[i]: round(float(tax[i]), 4) for i in order},
                "uncertainty": {
                    "risk_std": round(float(rep.risk_std[b]), 5),
                    "epistemic_variance": round(float(rep.epistemic_variance[b]), 6),
                    "aleatoric_variance": round(float(rep.aleatoric_variance[b]), 6),
                    "total_variance": round(float(rep.total_variance[b]), 6),
                    "predictive_entropy": round(float(rep.predictive_entropy[b]), 5),
                    "expected_entropy": round(float(rep.expected_entropy[b]), 5),
                    "mutual_information": round(float(rep.mutual_information[b]), 6),
                    "ood_score": round(ood_score, 6),
                    "ood_threshold": self.ood_threshold,
                    "is_out_of_distribution": bool(is_ood),
                    "mc_samples": int(samples.risk.shape[0]),
                },
                "modality_gates": {m: round(float(samples.gates_mean[b, i]), 4) for i, m in enumerate(MODALITIES)},
                "modalities_present": [m for i, m in enumerate(MODALITIES) if x.presence[b, i] > 0],
                "suspicious_spans": {
                    "text": _decode_spans(span_probs[:t_len], offsets[b], text_src),
                    "url": _decode_spans(span_probs[t_len:], [(i, i + 1) for i in range(len(url_strs[b]))], url_strs[b]),
                },
                "url_analyzed": url_strs[b] or None,
                "transcript": (transcripts or [None] * len(texts))[b],
            }
            if attr is not None:
                result["explanation"] = self._format_attributions(attr, b, offsets[b], text_src, url_strs[b], top_k)
            results.append(result)
        return results

    @staticmethod
    def _format_attributions(attr, b: int, offsets, text: str, url: str, top_k: int) -> dict[str, Any]:
        tok = attr.text[b].cpu().numpy()
        tokens = [(text[s:e], float(tok[i])) for i, (s, e) in enumerate(offsets) if s != e and i < len(tok)]
        tokens.sort(key=lambda t: -abs(t[1]))

        char = attr.url[b].cpu().numpy()
        segments = [(m.group(), float(char[m.start() : m.end()].sum())) for m in URL_SEGMENT.finditer(url)]
        segments.sort(key=lambda t: -abs(t[1]))

        ac = attr.acoustic[b].cpu().numpy()
        feats = sorted(zip(ACOUSTIC_FEATURE_NAMES, ac.tolist()), key=lambda t: -abs(t[1]))
        return {
            "method": attr.method,
            "target": attr.target,
            "convergence_delta": None if attr.delta is None else round(float(attr.delta[b]), 6),
            "modality_totals": {
                "text": round(float(tok.sum()), 5),
                "url": round(float(char.sum()), 5),
                "acoustic": round(float(ac.sum()), 5),
            },
            "top_text_tokens": [{"token": t, "attribution": round(a, 5)} for t, a in tokens[:top_k]],
            "top_url_segments": [{"segment": s, "attribution": round(a, 5)} for s, a in segments[:top_k]],
            "top_acoustic_features": [{"feature": f, "attribution": round(a, 5)} for f, a in feats[:top_k] if a != 0.0],
        }
