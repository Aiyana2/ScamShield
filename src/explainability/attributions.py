r"""Token-, character- and acoustic-level attributions for risk audits.

Integrated Gradients (Sundararajan et al., 2017) via Captum
-----------------------------------------------------------
For input x, baseline x' and model output F::

    IG_i(x) = (x_i - x'_i) * ∫_0^1  dF(x' + a (x - x')) / dx_i  da

approximated with an ``n_steps`` Riemann / Gauss-Legendre sum. IG satisfies
*completeness*: sum_i IG_i = F(x) - F(x'), and Captum's convergence delta
reports how far the approximation is from that identity.

The attribution is taken jointly over the three continuous model inputs:

* text: the 768-d hidden state of every token. Baseline 0. Per-token score
  = sum over the hidden dimension.
* URL: the Char-CNN *embedding* of every character (ids are discrete, so we
  integrate in embedding space). Baseline 0 = the PAD embedding. Per-char
  score = sum over the embedding dimension; the pipeline aggregates chars to
  URL segments (sub-domain labels, domain, path parts).
* acoustic: the raw 50-d feature vector. Baseline = the training-set mean
  (the model's standardiser maps it to 0), so attributions read as "how much
  this feature's deviation from a typical call moved the score".

Dropout is disabled during attribution (plain eval mode) so the explanation is
of the deterministic mean network, not of one random MC sample.

SHAP (KernelSHAP) on the acoustic block is offered as a model-agnostic cross
check: :func:`shap_acoustic_attributions`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.multitask_net import MultimodalInputs, MultiTaskScamNet


@dataclass
class AttributionResult:
    text: torch.Tensor       # (B, T)
    url: torch.Tensor        # (B, L)
    acoustic: torch.Tensor   # (B, A)
    delta: torch.Tensor | None  # (B,) completeness error (IG only)
    method: str
    target: str


class MultimodalAttributor:
    def __init__(self, model: MultiTaskScamNet, method: str = "integrated_gradients", n_steps: int = 32,
                 internal_batch_size: int | None = 64):
        if method not in ("integrated_gradients", "gradient_x_input"):
            raise ValueError(f"unknown method {method}")
        self.model = model
        self.method = method
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    def _forward(self, target_class: int | None):
        def f(text_h, url_e, acoustic, text_mask, url_mask, url_stats, presence):
            out = self.model.forward_from_embeddings(
                text_h, url_e, acoustic, text_mask=text_mask, url_mask=url_mask,
                url_stats=url_stats, presence=presence,
            )
            return out.risk_logit if target_class is None else out.taxonomy_logits[:, target_class]
        return f

    def attribute(self, x: MultimodalInputs, target: str | int = "risk") -> AttributionResult:
        """target: "risk" (scam logit) or a taxonomy class index."""
        target_class = None if target == "risk" else int(target)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                url_emb = self.model.url_encoder.embed(x.url_ids)
            inputs = (x.text_hidden.detach(), url_emb.detach(), x.acoustic.detach())
            baselines = (
                torch.zeros_like(inputs[0]),
                torch.zeros_like(inputs[1]),
                self.model.acoustic_mean.expand_as(inputs[2]).clone(),
            )
            extra = (x.text_mask, x.url_mask, x.url_stats, x.presence)
            fwd = self._forward(target_class)
            method = self.method
            try:
                from captum.attr import IntegratedGradients
            except ImportError:
                method = "gradient_x_input"
            if method == "integrated_gradients":
                ig = IntegratedGradients(fwd)
                attrs, delta = ig.attribute(
                    inputs, baselines=baselines, additional_forward_args=extra, n_steps=self.n_steps,
                    internal_batch_size=self.internal_batch_size, return_convergence_delta=True,
                )
            else:
                attrs, delta = self._gradient_x_input(fwd, inputs, baselines, extra), None
        finally:
            self.model.train(was_training)

        text = attrs[0].sum(-1) * x.text_mask
        url = attrs[1].sum(-1) * x.url_mask
        acoustic = attrs[2]
        # Absent modalities get exactly zero attribution.
        text = text * x.presence[:, 0:1]
        acoustic = acoustic * x.presence[:, 1:2]
        url = url * x.presence[:, 2:3]
        return AttributionResult(text=text.detach(), url=url.detach(), acoustic=acoustic.detach(),
                                 delta=None if delta is None else delta.detach(), method=method,
                                 target="risk" if target_class is None else f"taxonomy:{target_class}")

    @staticmethod
    def _gradient_x_input(fwd, inputs, baselines, extra):
        """(x - x') ⊙ dF/dx — a one-step IG, used when Captum is not installed."""
        leaves = tuple(t.clone().requires_grad_(True) for t in inputs)
        out = fwd(*leaves, *extra).sum()
        grads = torch.autograd.grad(out, leaves)
        return tuple((x - b) * g for x, b, g in zip(inputs, baselines, grads))


def shap_acoustic_attributions(model: MultiTaskScamNet, x: MultimodalInputs, background: np.ndarray,
                               index: int = 0, nsamples: int = 200) -> np.ndarray:
    """KernelSHAP values of the risk probability w.r.t. the acoustic features of sample ``index``.

    Text and URL inputs of that sample are held fixed; ``background`` (K, A)
    is a small set of raw acoustic vectors from training data.
    """
    import shap

    single = x.index_select(torch.tensor([index]))
    model.eval()

    def f(acoustic_matrix: np.ndarray) -> np.ndarray:
        n = acoustic_matrix.shape[0]
        batch = single.repeat(n)
        batch.acoustic = torch.as_tensor(acoustic_matrix, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(model(batch).risk_logit).numpy()

    explainer = shap.KernelExplainer(f, background)
    values = explainer.shap_values(single.acoustic.numpy(), nsamples=nsamples, silent=True)
    return np.asarray(values).reshape(-1)
