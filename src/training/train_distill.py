"""Teacher training and teacher -> student knowledge distillation.

Stages
------
1. ``teacher``: train :class:`MultiTaskScamNet` (teacher config) with the
   multi-task focal loss and learned task weights.
2. ``student``: freeze the teacher (eval mode, no dropout) and train
   :class:`StudentScamNet` on  alpha_kd * KL(teacher || student) + (1 - alpha_kd) * focal.
3. After each stage the OOD threshold is calibrated on the validation split
   with MC dropout and stored in the checkpoint.

Both stages share the same cached frozen-encoder text features and the same
modality-dropout masks per batch, so the student is distilled on exactly the
inputs the teacher saw.

Usage::

    # real data (JSONL schema in src/data/dataset.py)
    python -m src.training.train_distill --train data/train.jsonl --val data/val.jsonl \
        --text-encoder distilroberta-base --stage both --out artifacts/

    # offline smoke run on toy synthetic data (not a benchmark)
    python -m src.training.train_distill --synthetic 256 --text-encoder hashing --epochs 2 --out artifacts/smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import IGNORE_INDEX, ModelConfig, TAXONOMY, student_config, teacher_config
from src.data.dataset import (
    MultimodalCollator, MultimodalScamDataset, ScamRecord, fit_acoustic_stats, load_jsonl,
)
from src.features.text_encoder import build_text_encoder
from src.inference.uncertainty import calibrate_ood_threshold
from src.models.multitask_net import MultimodalInputs, MultiTaskScamNet, count_parameters
from src.models.student_model import StudentScamNet
from src.training.losses import DistillationLoss, MultiTaskFocalLoss

log = logging.getLogger("scamshield.train")


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    modality_dropout: float = 0.15
    kd_temperature: float = 3.0
    kd_alpha: float = 0.7
    mc_samples: int = 20
    ood_quantile: float = 0.95
    patience: int = 3
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Batch-level helpers
# --------------------------------------------------------------------------- #
def apply_modality_dropout(x: MultimodalInputs, p: float, generator: torch.Generator | None = None) -> MultimodalInputs:
    """Randomly mark present modalities as missing (presence -> 0) with prob p.

    At least one originally-present modality is always kept per sample. This
    teaches the sigmoid gates to cope with missing inputs at inference time.
    """
    if p <= 0:
        return x
    pres = x.presence
    drop = (torch.rand(pres.shape, generator=generator, device=pres.device) < p).to(pres.dtype)
    new = pres * (1 - drop)
    empty = new.sum(-1) == 0
    if empty.any():
        new[empty] = pres[empty]
    return MultimodalInputs(**{**x.__dict__, "presence": new})


def _lr_lambda(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))  # cosine decay

    return fn


def _optimizer(params, cfg: TrainConfig, total_steps: int):
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total_steps, cfg.warmup_ratio))
    return opt, sched


# --------------------------------------------------------------------------- #
# Metrics (numpy only)
# --------------------------------------------------------------------------- #
def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U formulation with tie-averaged ranks."""
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = score.argsort()
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def macro_f1(y: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


@torch.no_grad()
def evaluate(model: MultiTaskScamNet, loader: DataLoader, loss_fn: MultiTaskFocalLoss, device: str) -> dict[str, float]:
    model.eval()
    losses, risk_y, risk_p, tax_y, tax_p = [], [], [], [], []
    span_tp = span_fp = span_fn = 0
    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        out = model(inputs)
        losses.append(loss_fn(out, targets["risk"], targets["taxonomy"], targets["span"])["loss"].item())
        risk_y.append(targets["risk"].cpu().numpy())
        risk_p.append(torch.sigmoid(out.risk_logit).cpu().numpy())
        tax_y.append(targets["taxonomy"].cpu().numpy())
        tax_p.append(out.taxonomy_logits.argmax(-1).cpu().numpy())
        valid = (targets["span"] != IGNORE_INDEX) & out.span_mask
        gold = (targets["span"] > 0) & valid
        pred = (out.span_logits.argmax(-1) > 0) & valid
        span_tp += int((gold & pred).sum())
        span_fp += int((~gold & pred).sum())
        span_fn += int((gold & ~pred).sum())
    ry, rp = np.concatenate(risk_y), np.concatenate(risk_p)
    ty, tp = np.concatenate(tax_y), np.concatenate(tax_p)
    return {
        "loss": float(np.mean(losses)),
        "risk_auroc": roc_auc(ry, rp),
        "risk_accuracy": float(np.mean((rp >= 0.5) == ry)),
        "taxonomy_macro_f1": macro_f1(ty, tp, len(TAXONOMY)),
        "span_token_f1": 2 * span_tp / max(1, 2 * span_tp + span_fp + span_fn),
    }


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, model: MultiTaskScamNet, kind: str, text_encoder: str,
                    ood_threshold: float | None, metrics: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": kind,  # "teacher" | "student"
        "model_config": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "text_encoder": text_encoder,
        "ood_threshold": ood_threshold,
        "metrics": metrics or {},
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[MultiTaskScamNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = (StudentScamNet if ckpt["kind"] == "student" else MultiTaskScamNet)(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
def _hard_loss(cfg: TrainConfig) -> MultiTaskFocalLoss:
    return MultiTaskFocalLoss(risk_alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)


def train_teacher(model: MultiTaskScamNet, train_loader: DataLoader, val_loader: DataLoader,
                  cfg: TrainConfig) -> tuple[MultiTaskScamNet, list[dict]]:
    device = cfg.device
    model.to(device)
    loss_fn = _hard_loss(cfg).to(device)
    total = cfg.epochs * len(train_loader)
    opt, sched = _optimizer(list(model.parameters()) + list(loss_fn.parameters()), cfg, total)
    return _fit(model, train_loader, val_loader, cfg, opt, sched, loss_fn,
                step_fn=lambda x, t: loss_fn(model(x), t["risk"], t["taxonomy"], t["span"]))


def distill_student(teacher: MultiTaskScamNet, student: StudentScamNet, train_loader: DataLoader,
                    val_loader: DataLoader, cfg: TrainConfig) -> tuple[StudentScamNet, list[dict]]:
    device = cfg.device
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.to(device)
    hard = _hard_loss(cfg).to(device)
    kd_loss = DistillationLoss(hard, temperature=cfg.kd_temperature, alpha_kd=cfg.kd_alpha).to(device)
    total = cfg.epochs * len(train_loader)
    opt, sched = _optimizer(list(student.parameters()) + list(hard.parameters()), cfg, total)

    def step(x: MultimodalInputs, t: dict) -> dict:
        with torch.no_grad():
            t_out = teacher(x)
        return kd_loss(student(x), t_out, t["risk"], t["taxonomy"], t["span"])

    return _fit(student, train_loader, val_loader, cfg, opt, sched, hard, step_fn=step)


def _fit(model, train_loader, val_loader, cfg: TrainConfig, opt, sched, eval_loss, step_fn):
    history: list[dict] = []
    best, best_state, bad_epochs = float("inf"), None, 0
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for inputs, targets, _ in train_loader:
            inputs = apply_modality_dropout(inputs, cfg.modality_dropout, gen).to(cfg.device)
            targets = {k: v.to(cfg.device) for k, v in targets.items()}
            out = step_fn(inputs, targets)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            running += out["loss"].item()
        metrics = evaluate(model, val_loader, eval_loss, cfg.device)
        metrics.update(epoch=epoch, train_loss=running / max(1, len(train_loader)))
        history.append(metrics)
        log.info("epoch %d %s", epoch, json.dumps({k: round(v, 4) for k, v in metrics.items()}))
        if metrics["loss"] < best - 1e-4:
            best, bad_epochs = metrics["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log.info("early stopping at epoch %d", epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_loaders(train: list[ScamRecord], val: list[ScamRecord], text_encoder, batch_size: int):
    collate = MultimodalCollator(text_encoder)
    train_ds = MultimodalScamDataset(train, text_encoder)
    val_ds = MultimodalScamDataset(val, text_encoder)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=len(train_ds) > batch_size),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate),
    )


def run(args: argparse.Namespace) -> dict:
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      kd_alpha=args.kd_alpha, kd_temperature=args.temperature, mc_samples=args.mc_samples)
    if args.device:
        cfg.device = args.device
    set_seed(cfg.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from src.data.synthetic import make_synthetic_records

        log.warning("Training on SYNTHETIC toy data: metrics only show the code runs, not real performance.")
        records = make_synthetic_records(args.synthetic, seed=cfg.seed)
        split = int(0.8 * len(records))
        train, val = records[:split], records[split:]
    else:
        train, val = load_jsonl(args.train), load_jsonl(args.val)

    text_encoder = build_text_encoder(args.text_encoder, device=cfg.device)
    train_loader, val_loader = build_loaders(train, val, text_encoder, cfg.batch_size)
    mean, std = fit_acoustic_stats(train)
    summary: dict = {"train_size": len(train), "val_size": len(val), "synthetic": bool(args.synthetic),
                     "train_config": asdict(cfg)}

    teacher_path = out_dir / "teacher.pt"
    if args.stage in ("teacher", "both"):
        teacher = MultiTaskScamNet(teacher_config())
        teacher.set_acoustic_normalization(mean, std)
        log.info("teacher parameters: %s", f"{count_parameters(teacher):,}")
        teacher, hist = train_teacher(teacher, train_loader, val_loader, cfg)
        thr = calibrate_ood_threshold(teacher, val_loader, cfg.mc_samples, cfg.ood_quantile, cfg.device)
        save_checkpoint(teacher_path, teacher, "teacher", args.text_encoder, thr, hist[-1])
        summary["teacher"] = {"params": count_parameters(teacher), "history": hist, "ood_threshold": thr}
    else:
        teacher, _ = load_checkpoint(teacher_path, cfg.device)

    if args.stage in ("student", "both"):
        student = StudentScamNet(student_config())
        student.set_acoustic_normalization(mean, std)
        log.info("student parameters: %s", f"{count_parameters(student):,}")
        student, hist = distill_student(teacher, student, train_loader, val_loader, cfg)
        thr = calibrate_ood_threshold(student, val_loader, cfg.mc_samples, cfg.ood_quantile, cfg.device)
        save_checkpoint(out_dir / "student.pt", student, "student", args.text_encoder, thr, hist[-1])
        summary["student"] = {"params": count_parameters(student), "history": hist, "ood_threshold": thr}

    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--train", help="training JSONL")
    src.add_argument("--synthetic", type=int, help="generate N toy synthetic records instead (smoke test only)")
    p.add_argument("--val", help="validation JSONL (required with --train)")
    p.add_argument("--text-encoder", default="distilroberta-base", help="HF model name or 'hashing' (offline tests)")
    p.add_argument("--stage", choices=["teacher", "student", "both"], default="both")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--kd-alpha", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--mc-samples", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="artifacts")
    args = p.parse_args(argv)
    if args.train and not args.val:
        p.error("--val is required with --train")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
