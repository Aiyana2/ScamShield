r"""Monte Carlo dropout, uncertainty decomposition and OOD detection.

MC dropout (Gal & Ghahramani, 2016) treats dropout at test time as sampling
weights from an approximate posterior q(W). With N stochastic passes
p_s = f(x; W_s), W_s ~ q(W):

Binary risk head — law of total variance
----------------------------------------
For a Bernoulli prediction with mean p̄ = (1/N) sum_s p_s::

    Var[y | x]  =  E_s[ p_s (1 - p_s) ]   +   Var_s[ p_s ]
     (total)        (aleatoric: noise        (epistemic: disagreement
     = p̄(1-p̄)       the model expects        between weight samples,
                    even with known W)       shrinks with more data)

The identity holds exactly with the population variance, which is what is
computed here.

Taxonomy head — entropy decomposition (BALD, Houlsby et al. 2011)
-----------------------------------------------------------------
With p̄_c = (1/N) sum_s p_{s,c}::

    H[p̄]                         total predictive entropy
    E_s[ H[p_s] ]                 expected entropy        -> aleatoric
    I(y; W | x) = H[p̄] - E_s[H[p_s]]   mutual information -> epistemic

OOD / zero-day flag
-------------------
Epistemic uncertainty is high where the training data gave the weights no
constraint, which is what an unseen scam pattern looks like. The OOD score is

    score = I_taxonomy + I_risk          (I_risk: the same MI for the 2-class risk head)

and a sample is flagged when score > tau, where tau is the ``quantile``
(default 95th percentile) of the score on in-distribution validation data,
i.e. a target ~5 % false-alarm rate on known traffic.

Only ``nn.Dropout`` and ``nn.MultiheadAttention`` modules are switched to
train mode; BatchNorm stays in eval mode so its running statistics are used
and not corrupted.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.models.multitask_net import MultimodalInputs, MultiTaskScamNet

EPS = 1e-8


@contextlib.contextmanager
def mc_dropout_mode(model: nn.Module):
    """Eval mode everywhere except dropout layers; restores previous modes on exit."""
    previous = {m: m.training for m in model.modules()}
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.MultiheadAttention)):
            m.train()
    try:
        yield model
    finally:
        for m, was_training in previous.items():
            m.train(was_training)


@dataclass
class MCSamples:
    risk: torch.Tensor          # (N, B) probabilities
    taxonomy: torch.Tensor      # (N, B, C) probabilities
    span_mean: torch.Tensor     # (B, T+L, S) mean token-tag probabilities
    span_mask: torch.Tensor     # (B, T+L)
    gates_mean: torch.Tensor    # (B, M)
    text_len: int


@torch.no_grad()
def mc_dropout_predict(model: MultiTaskScamNet, inputs: MultimodalInputs, n_samples: int = 30,
                       max_rows: int = 1024) -> MCSamples:
    """N stochastic forward passes, vectorised by tiling the batch.

    Each tiled copy draws its own dropout mask, so one forward pass over
    k * B rows equals k independent MC samples. Chunks keep memory bounded.
    """
    bsz = inputs.batch_size
    per_chunk = max(1, min(n_samples, max_rows // max(1, bsz)))
    risk, tax, span_sum, gate_sum = [], [], None, None
    done = 0
    with mc_dropout_mode(model):
        while done < n_samples:
            k = min(per_chunk, n_samples - done)
            out = model(inputs.repeat(k))
            risk.append(torch.sigmoid(out.risk_logit).view(k, bsz))
            tax.append(torch.softmax(out.taxonomy_logits, -1).view(k, bsz, -1))
            sp = torch.softmax(out.span_logits, -1).view(k, bsz, *out.span_logits.shape[1:]).sum(0)
            gt = out.gates.view(k, bsz, -1).sum(0)
            span_sum = sp if span_sum is None else span_sum + sp
            gate_sum = gt if gate_sum is None else gate_sum + gt
            span_mask, text_len = out.span_mask[:bsz], out.text_len
            done += k
    return MCSamples(
        risk=torch.cat(risk), taxonomy=torch.cat(tax), span_mean=span_sum / n_samples,
        span_mask=span_mask, gates_mean=gate_sum / n_samples, text_len=text_len,
    )


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(p * torch.log(p + EPS)).sum(dim)


@dataclass
class UncertaintyReport:
    risk_mean: torch.Tensor            # (B,)
    risk_std: torch.Tensor             # (B,)
    epistemic_variance: torch.Tensor   # (B,)  Var_s[p_s]
    aleatoric_variance: torch.Tensor   # (B,)  E_s[p_s (1 - p_s)]
    total_variance: torch.Tensor       # (B,)  p̄ (1 - p̄) = epistemic + aleatoric
    taxonomy_mean: torch.Tensor        # (B, C)
    predictive_entropy: torch.Tensor   # (B,)  H[p̄]
    expected_entropy: torch.Tensor     # (B,)  E_s[H[p_s]]   aleatoric
    mutual_information: torch.Tensor   # (B,)  H[p̄] - E_s[H[p_s]]   epistemic
    ood_score: torch.Tensor            # (B,)


def decompose_uncertainty(samples: MCSamples) -> UncertaintyReport:
    r = samples.risk                                    # (N, B)
    p_bar = r.mean(0)
    epistemic = r.var(0, unbiased=False)
    aleatoric = (r * (1 - r)).mean(0)

    t = samples.taxonomy                                # (N, B, C)
    t_bar = t.mean(0)
    h_total = _entropy(t_bar)
    h_expected = _entropy(t).mean(0)
    mi_tax = (h_total - h_expected).clamp(min=0)

    r2 = torch.stack([1 - r, r], -1)                    # (N, B, 2)
    mi_risk = (_entropy(r2.mean(0)) - _entropy(r2).mean(0)).clamp(min=0)

    return UncertaintyReport(
        risk_mean=p_bar, risk_std=r.std(0, unbiased=False),
        epistemic_variance=epistemic, aleatoric_variance=aleatoric, total_variance=p_bar * (1 - p_bar),
        taxonomy_mean=t_bar, predictive_entropy=h_total, expected_entropy=h_expected,
        mutual_information=mi_tax, ood_score=mi_tax + mi_risk,
    )


@torch.no_grad()
def calibrate_ood_threshold(model: MultiTaskScamNet, loader, n_samples: int = 20, quantile: float = 0.95,
                            device: str = "cpu") -> float:
    """tau = quantile of the OOD score over an in-distribution (validation) loader."""
    scores = []
    for inputs, _, _ in loader:
        rep = decompose_uncertainty(mc_dropout_predict(model, inputs.to(device), n_samples))
        scores.append(rep.ood_score.cpu().numpy())
    if not scores:
        raise ValueError("empty calibration loader")
    return float(np.quantile(np.concatenate(scores), quantile))
