"""Compact student network for knowledge distillation.

The student keeps the teacher's *interface* (same inputs, same three heads,
same output dataclass) so the distillation loss can match head-by-head, but
removes the expensive parts:

* plain sigmoid GMU instead of GMU + cross-modal multi-head attention
* no contextual Transformer adapter on top of the frozen text encoder
* two Char-CNN branches (k = 3, 5) with 64 channels instead of three with 192
* one residual trunk block, d_model = 128 instead of 384

Use :func:`compare_sizes` to print the parameter reduction for your configs.
"""
from __future__ import annotations

import time

import torch

from src.config import ModelConfig, student_config, teacher_config
from src.models.multitask_net import MultimodalInputs, MultiTaskScamNet, count_parameters


class StudentScamNet(MultiTaskScamNet):
    """Lightweight multi-task model distilled from :class:`MultiTaskScamNet`."""

    def __init__(self, cfg: ModelConfig | None = None):
        cfg = cfg or student_config()
        if cfg.cross_attention:
            raise ValueError("StudentScamNet is meant to run without cross-attention; use MultiTaskScamNet")
        super().__init__(cfg)


def build_teacher(cfg: ModelConfig | None = None) -> MultiTaskScamNet:
    return MultiTaskScamNet(cfg or teacher_config())


def build_student(cfg: ModelConfig | None = None) -> StudentScamNet:
    return StudentScamNet(cfg or student_config())


@torch.no_grad()
def measure_latency(model: torch.nn.Module, inputs: MultimodalInputs, n_runs: int = 20) -> float:
    """Mean wall-clock milliseconds per forward pass (eval mode, current device)."""
    model.eval()
    model(inputs)  # warm-up
    start = time.perf_counter()
    for _ in range(n_runs):
        model(inputs)
    return 1000.0 * (time.perf_counter() - start) / n_runs


def compare_sizes(teacher: torch.nn.Module, student: torch.nn.Module) -> dict[str, float]:
    t, s = count_parameters(teacher), count_parameters(student)
    return {"teacher_params": t, "student_params": s, "compression_ratio": t / max(s, 1)}
