r"""Loss functions: focal losses, multi-task weighting and KL distillation.

------------------------------------------------------------------------------
Binary focal loss (Lin et al., 2017) — Head 1, scam risk
------------------------------------------------------------------------------
With logit s, p = sigmoid(s) and label y in {0, 1}::

    p_t     = p        if y = 1 else 1 - p
    alpha_t = alpha    if y = 1 else 1 - alpha
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

Cross-entropy is the special case gamma = 0, alpha_t = 1. The modulating factor
(1 - p_t)^gamma -> 0 for well-classified examples (p_t -> 1), so the loss is
dominated by hard and minority-class examples instead of the large number of
easy benign messages. log(p_t) is computed as -BCEWithLogits for numerical
stability; p_t = exp(-BCE).

------------------------------------------------------------------------------
Multi-class focal loss — Head 2 (taxonomy) and Head 3 (per-token BIO tags)
------------------------------------------------------------------------------
With softmax probabilities p_c and target class t::

    FL = - alpha_t * (1 - p_t)^gamma * log p_t

Tokens labelled IGNORE_INDEX (padding, special tokens) contribute nothing.

------------------------------------------------------------------------------
Homoscedastic uncertainty task weighting (Kendall, Gal & Cipolla, 2018)
------------------------------------------------------------------------------
Instead of hand-tuned lambdas, learn s_k = log sigma_k^2 per task::

    L_total = sum_k  exp(-s_k) * L_k  +  s_k

A task with noisy / harder labels learns a larger sigma_k and is down-weighted;
the +s_k term stops the trivial solution sigma_k -> infinity.

------------------------------------------------------------------------------
Knowledge distillation (Hinton, Vinyals & Dean, 2015)
------------------------------------------------------------------------------
Teacher logits z^T, student logits z^S, temperature tau::

    p^T_tau = softmax(z^T / tau),   p^S_tau = softmax(z^S / tau)
    KL(p^T_tau || p^S_tau) = sum_c p^T_tau,c * (log p^T_tau,c - log p^S_tau,c)
    L_KD = tau^2 * KL(p^T_tau || p^S_tau)

tau > 1 softens both distributions, exposing the teacher's "dark knowledge"
(e.g. that a Crypto Fraud message is closer to Financial Coercion than to
Benign). Gradients of the softened KL scale as 1/tau^2, so multiplying by
tau^2 keeps their magnitude comparable to the hard loss when tau changes.
The binary risk head is distilled as the two-class distribution
softmax([0, s] / tau) = [1 - sigmoid(s/tau), sigmoid(s/tau)].

Final student objective::

    L_student = alpha_kd * L_KD  +  (1 - alpha_kd) * L_hard_focal
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import IGNORE_INDEX
from src.models.multitask_net import MultiTaskOutput


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # = -log p_t
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return _reduce(loss, self.reduction)


class MultiClassFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma, self.ignore_index, self.reduction = gamma, ignore_index, reduction
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[-1]
        logits = logits.reshape(-1, num_classes)
        targets = targets.reshape(-1)
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        logits, targets = logits[valid], targets[valid]
        log_p = F.log_softmax(logits, dim=-1)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        loss = -(1.0 - p_t).pow(self.gamma) * log_p_t
        if self.alpha.numel():
            loss = loss * self.alpha.to(loss.dtype)[targets]
        return _reduce(loss, self.reduction)


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class MultiTaskFocalLoss(nn.Module):
    """Hard-label loss over the three heads with optional learned task weights."""

    TASKS = ("risk", "taxonomy", "span")

    def __init__(
        self,
        risk_alpha: float = 0.25,
        gamma: float = 2.0,
        taxonomy_alpha: torch.Tensor | None = None,
        span_alpha: torch.Tensor | None = None,
        learn_task_weights: bool = True,
        fixed_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
    ):
        super().__init__()
        self.risk = BinaryFocalLoss(alpha=risk_alpha, gamma=gamma)
        self.taxonomy = MultiClassFocalLoss(gamma=gamma, alpha=taxonomy_alpha)
        self.span = MultiClassFocalLoss(gamma=gamma, alpha=span_alpha)
        self.learn_task_weights = learn_task_weights
        # s_k = log sigma_k^2, initialised to 0 (weight 1).
        self.log_vars = nn.Parameter(torch.zeros(len(self.TASKS)), requires_grad=learn_task_weights)
        self.register_buffer("fixed_weights", torch.tensor(fixed_weights))

    def forward(
        self, out: MultiTaskOutput, risk_y: torch.Tensor, taxonomy_y: torch.Tensor, span_y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        span_targets = span_y.masked_fill(~out.span_mask, IGNORE_INDEX)
        parts = torch.stack([
            self.risk(out.risk_logit, risk_y),
            self.taxonomy(out.taxonomy_logits, taxonomy_y),
            self.span(out.span_logits, span_targets),
        ])
        if self.learn_task_weights:
            total = (torch.exp(-self.log_vars) * parts + self.log_vars).sum()
        else:
            total = (self.fixed_weights * parts).sum()
        return {"loss": total, **{f"{k}_loss": parts[i].detach() for i, k in enumerate(self.TASKS)}}


def kl_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """tau^2 * KL(softmax(z_T/tau) || softmax(z_S/tau)), averaged over rows (or masked rows)."""
    c = student_logits.shape[-1]
    log_p_s = F.log_softmax(student_logits.reshape(-1, c) / tau, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.reshape(-1, c) / tau, dim=-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1)  # per row
    if mask is not None:
        m = mask.reshape(-1).to(kl.dtype)
        kl = (kl * m).sum() / m.sum().clamp(min=1.0)
    else:
        kl = kl.mean()
    return (tau ** 2) * kl


def _binary_as_two_class(logit: torch.Tensor) -> torch.Tensor:
    """softmax([0, s]) == [1 - sigmoid(s), sigmoid(s)]."""
    return torch.stack([torch.zeros_like(logit), logit], dim=-1)


class DistillationLoss(nn.Module):
    """alpha_kd * (KL on all three heads) + (1 - alpha_kd) * hard multi-task focal loss."""

    def __init__(
        self,
        hard_loss: MultiTaskFocalLoss,
        temperature: float = 3.0,
        alpha_kd: float = 0.7,
        head_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
    ):
        super().__init__()
        if not 0.0 <= alpha_kd <= 1.0:
            raise ValueError("alpha_kd must be in [0, 1]")
        self.hard_loss = hard_loss
        self.tau, self.alpha_kd = temperature, alpha_kd
        self.head_weights = head_weights

    def forward(
        self,
        student: MultiTaskOutput,
        teacher: MultiTaskOutput,
        risk_y: torch.Tensor,
        taxonomy_y: torch.Tensor,
        span_y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        w_r, w_c, w_s = self.head_weights
        kd_risk = kl_distillation(_binary_as_two_class(student.risk_logit),
                                  _binary_as_two_class(teacher.risk_logit), self.tau)
        kd_tax = kl_distillation(student.taxonomy_logits, teacher.taxonomy_logits, self.tau)
        kd_span = kl_distillation(student.span_logits, teacher.span_logits, self.tau, mask=student.span_mask)
        kd = w_r * kd_risk + w_c * kd_tax + w_s * kd_span

        hard = self.hard_loss(student, risk_y, taxonomy_y, span_y)
        total = self.alpha_kd * kd + (1.0 - self.alpha_kd) * hard["loss"]
        return {
            "loss": total,
            "kd_loss": kd.detach(),
            "kd_risk": kd_risk.detach(),
            "kd_taxonomy": kd_tax.detach(),
            "kd_span": kd_span.detach(),
            "hard_loss": hard["loss"].detach(),
        }
