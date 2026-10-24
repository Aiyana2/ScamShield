"""Input-ablation benchmark for the results slide.

Runs the trained checkpoint over the validation split three times:
  * Text-only      : every URL input removed
  * URL-only       : every text input removed
  * ScamShield     : all available inputs (text + URL found in the record or inside the text)

and reports Precision, Recall, F2, PR-AUC and per-item latency for the risk head.
The validation set is a mixed stream of messages and links, so each single-input
setting has to score items whose channel it cannot see; that is the point being measured.

Usage:
    python scripts/evaluate_ablation.py --checkpoint artifacts/student.pt --val data/val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_jsonl  # noqa: E402
from src.inference.pipeline import ScamShieldPipeline, extract_first_url  # noqa: E402


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """PR-AUC as average precision (step-wise area under the precision-recall curve)."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def metrics(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = s >= threshold
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return {"precision": p, "recall": r, "f2": f2, "pr_auc": average_precision(y, s), "false_negatives": fn}


def score(pipe: ScamShieldPipeline, texts, urls, mc: int, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        res = pipe.predict_batch(texts[i:i + batch], urls[i:i + batch], [None] * len(texts[i:i + batch]), mc)
        out += [r["risk_score"] for r in res]
    return np.array(out)


def latency_ms(pipe: ScamShieldPipeline, texts, urls, mc: int, n: int = 50) -> float:
    idx = [i for i in range(len(texts)) if texts[i] or urls[i]][:n]
    pipe.predict_batch([texts[idx[0]]], [urls[idx[0]]], [None], mc)  # warm-up
    t0 = time.perf_counter()
    for i in idx:
        pipe.predict_batch([texts[i]], [urls[i]], [None], mc)
    return (time.perf_counter() - t0) * 1000 / len(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="artifacts/student.pt")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/ablation.json")
    args = ap.parse_args()

    pipe = ScamShieldPipeline.from_checkpoint(args.checkpoint, device=args.device)
    recs = load_jsonl(args.val)
    y = np.array([r.is_scam for r in recs])
    text = [r.text or None for r in recs]
    url = [r.url or extract_first_url(r.text) for r in recs]
    none = [None] * len(recs)

    settings = {
        "Text-only": (text, none),
        "URL-only": (none, url),
        "ScamShield (text + URL)": (text, url),
    }
    rows = {}
    for name, (t, u) in settings.items():
        print(f"scoring {name} ...", flush=True)
        rows[name] = metrics(y, score(pipe, t, u, args.mc_samples))
        rows[name]["latency_ms"] = latency_ms(pipe, t, u, args.mc_samples)

    summary = {"checkpoint": args.checkpoint, "val_items": len(recs), "scams": int(y.sum()),
               "mc_samples": args.mc_samples, "threshold": 0.5, "results": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print(f"\n{len(recs)} validation items ({int(y.sum())} scams), risk threshold 0.5, {args.mc_samples} MC passes\n")
    print("| Model | Precision | Recall | F2 | PR-AUC | Missed scams | Latency (ms) |")
    print("|---|---|---|---|---|---|---|")
    for name, m in rows.items():
        print(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f2']:.3f} | {m['pr_auc']:.3f} | "
              f"{m['false_negatives']} | {m['latency_ms']:.0f} |")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
