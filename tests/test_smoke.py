"""Offline smoke tests on toy synthetic data (hashing text encoder, no downloads).

They check shapes, mathematical identities and that the end-to-end path
(train -> distil -> calibrate -> pipeline -> API) runs. They do NOT measure
real-world detection quality.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import ACOUSTIC_DIM, IGNORE_INDEX, TAXONOMY, student_config, teacher_config
from src.data.dataset import MultimodalCollator, MultimodalScamDataset, char_spans_to_bio, fit_acoustic_stats
from src.data.synthetic import make_synthetic_records
from src.explainability.attributions import MultimodalAttributor
from src.features.text_encoder import HashingTextEncoder
from src.inference.uncertainty import decompose_uncertainty, mc_dropout_mode, mc_dropout_predict
from src.models.char_cnn_url import CharCNNURLEncoder, URLCharTokenizer, shannon_entropy
from src.models.gated_fusion import GatedMultimodalFusion
from src.models.multitask_net import MultiTaskScamNet, count_parameters
from src.models.student_model import build_student, build_teacher
from src.training.losses import BinaryFocalLoss, MultiClassFocalLoss, kl_distillation
from src.training.train_distill import TrainConfig, apply_modality_dropout, build_loaders, distill_student, train_teacher

torch.manual_seed(0)


@pytest.fixture(scope="module")
def encoder():
    return HashingTextEncoder()


@pytest.fixture(scope="module")
def batch(encoder):
    recs = make_synthetic_records(24, seed=1)
    ds = MultimodalScamDataset(recs, encoder)
    return MultimodalCollator(encoder)([ds[i] for i in range(len(ds))])


# ---------------------------------------------------------------- models -- #
def test_char_cnn_shapes_and_padding_invariance():
    tok = URLCharTokenizer()
    enc = CharCNNURLEncoder().eval()
    a = tok(["http://paypa1-secure.com/login"])
    b = tok(["http://paypa1-secure.com/login", "https://github.com/a/very/long/path/that/pads/the/first/row"])
    pa, ca = enc(a.ids, a.mask, a.stats)
    pb, _ = enc(b.ids, b.mask, b.stats)
    assert pa.shape == (1, 128) and ca.shape == (1, a.ids.shape[1], 128)
    assert torch.allclose(pa[0], pb[0], atol=1e-5), "padding must not change the URL embedding"


def test_url_tokenizer_keeps_case_and_flags_non_ascii():
    tok = URLCharTokenizer()
    e = tok(["PayPaI.com", "аpple.com"])  # second starts with Cyrillic a
    assert e.ids[0, 0] != tok.char_id("p")
    assert int(e.ids[1, 0]) == 2
    assert shannon_entropy("aaaa") == 0.0 and math.isclose(shannon_entropy("ab"), 1.0)


def test_gmu_missing_modality_gate_is_zero():
    fusion = GatedMultimodalFusion(32, n_heads=4, cross_attention=True).eval()
    b = 3
    pooled = {m: torch.randn(b, 32) for m in ("text", "acoustic", "url")}
    tokens = {m: torch.randn(b, 5, 32) for m in ("text", "acoustic", "url")}
    masks = {m: torch.ones(b, 5, dtype=torch.bool) for m in ("text", "acoustic", "url")}
    presence = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    fused, gates, _ = fusion(pooled, tokens, masks, presence)
    assert fused.shape == (b, 32) and torch.isfinite(fused).all()
    assert gates[0, 1] == 0 and gates[1, 0] == 0 and gates[1, 1] == 0
    assert (gates[2] > 0).all()


@pytest.mark.parametrize("builder", [build_teacher, build_student])
def test_multitask_forward_backward(builder, batch):
    x, t, _ = batch
    model = builder()
    out = model(x)
    b = x.batch_size
    assert out.risk_logit.shape == (b,)
    assert out.taxonomy_logits.shape == (b, len(TAXONOMY))
    assert out.span_logits.shape == (b, x.text_hidden.shape[1] + x.url_ids.shape[1], 3)
    out.risk_logit.sum().backward()


def test_student_is_smaller():
    assert count_parameters(build_student()) * 3 < count_parameters(build_teacher())


# ---------------------------------------------------------------- losses -- #
def test_focal_reduces_to_weighted_bce_when_gamma_zero():
    logits, y = torch.randn(64), torch.randint(0, 2, (64,)).float()
    fl = BinaryFocalLoss(alpha=0.5, gamma=0.0)(logits, y)
    assert torch.allclose(fl, 0.5 * F.binary_cross_entropy_with_logits(logits, y), atol=1e-6)


def test_focal_downweights_easy_examples():
    easy, hard = torch.tensor([6.0]), torch.tensor([-1.0])
    y = torch.tensor([1.0])
    fl, ce = BinaryFocalLoss(alpha=0.5, gamma=2.0), BinaryFocalLoss(alpha=0.5, gamma=0.0)
    assert fl(easy, y) / ce(easy, y) < fl(hard, y) / ce(hard, y)


def test_multiclass_focal_ignores_index():
    logits = torch.randn(4, 3)
    y = torch.tensor([0, IGNORE_INDEX, 2, IGNORE_INDEX])
    full = MultiClassFocalLoss(gamma=0.0)(logits, y)
    assert torch.allclose(full, F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX), atol=1e-6)


def test_kl_distillation_zero_iff_equal():
    z = torch.randn(8, 6)
    assert kl_distillation(z, z, tau=3.0).abs() < 1e-6
    assert kl_distillation(z, torch.randn(8, 6), tau=3.0) > 0


# ------------------------------------------------------------ data utils -- #
def test_bio_tags():
    offsets = [(0, 0), (0, 4), (5, 10), (11, 15), (0, 0)]
    tags = char_spans_to_bio(offsets, [(5, 15)], 5)
    assert tags == [IGNORE_INDEX, 0, 1, 2, IGNORE_INDEX]


def test_modality_dropout_keeps_one():
    from src.models.multitask_net import MultimodalInputs
    pres = torch.tensor([[1.0, 0.0, 0.0]] * 50)
    inp = MultimodalInputs(torch.zeros(50, 1, 768), torch.ones(50, 1, dtype=torch.bool), torch.zeros(50, 1, dtype=torch.long),
                           torch.zeros(50, 1, dtype=torch.bool), torch.zeros(50, 7), torch.zeros(50, ACOUSTIC_DIM), pres)
    out = apply_modality_dropout(inp, p=0.99)
    assert (out.presence.sum(-1) >= 1).all()


# ----------------------------------------------------------- uncertainty -- #
def test_mc_dropout_variance_decomposition(batch):
    x, _, _ = batch
    model = build_teacher().eval()
    samples = mc_dropout_predict(model, x, n_samples=12)
    rep = decompose_uncertainty(samples)
    assert samples.risk.shape == (12, x.batch_size)
    assert torch.allclose(rep.epistemic_variance + rep.aleatoric_variance, rep.total_variance, atol=1e-6)
    assert (rep.mutual_information >= 0).all()
    assert (samples.risk.std(0) > 0).any(), "dropout must make MC samples differ"
    assert not any(m.training for m in model.modules()), "previous (eval) modes must be restored"


def test_mc_dropout_mode_keeps_batchnorm_in_eval():
    model = build_teacher()
    with mc_dropout_mode(model):
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert bns and not any(m.training for m in bns)
        assert drops and all(m.training for m in drops)


# -------------------------------------------------------- explainability -- #
def test_integrated_gradients_completeness(batch):
    x, _, _ = batch
    model = build_student().eval()
    attr = MultimodalAttributor(model, n_steps=64).attribute(x.index_select(torch.arange(4)), "risk")
    assert attr.method == "integrated_gradients"
    assert attr.text.shape[0] == 4 and attr.acoustic.shape == (4, ACOUSTIC_DIM)
    assert attr.delta.abs().max() < 0.05


# ------------------------------------------------- end-to-end smoke run -- #
@pytest.fixture(scope="module")
def trained(encoder, tmp_path_factory):
    recs = make_synthetic_records(160, seed=2)
    train_loader, val_loader = build_loaders(recs[:128], recs[128:], encoder, batch_size=16)
    cfg = TrainConfig(epochs=3, batch_size=16, lr=1e-3, mc_samples=5, device="cpu", patience=10)
    mean, std = fit_acoustic_stats(recs[:128])
    teacher, student = build_teacher(), build_student()
    teacher.set_acoustic_normalization(mean, std)
    student.set_acoustic_normalization(mean, std)
    teacher, t_hist = train_teacher(teacher, train_loader, val_loader, cfg)
    student, s_hist = distill_student(teacher, student, train_loader, val_loader, cfg)
    return teacher, student, t_hist, s_hist, val_loader, tmp_path_factory.mktemp("ckpt")


def test_training_reduces_loss(trained):
    _, _, t_hist, s_hist, _, _ = trained
    assert t_hist[-1]["train_loss"] < t_hist[0]["train_loss"]
    assert all(np.isfinite(h["loss"]) for h in t_hist + s_hist)


def test_pipeline_and_api(trained, encoder):
    from fastapi.testclient import TestClient

    import app as api
    from src.inference.pipeline import ScamShieldPipeline
    from src.inference.uncertainty import calibrate_ood_threshold
    from src.training.train_distill import save_checkpoint

    _, student, _, _, val_loader, tmp = trained
    thr = calibrate_ood_threshold(student, val_loader, n_samples=5)
    path = tmp / "student.pt"
    save_checkpoint(path, student, "student", "hashing", thr)

    pipe = ScamShieldPipeline.from_checkpoint(path, text_encoder=encoder)
    res = pipe.predict(text="Final notice: pay the overdue tax via gift cards at http://ch4se-verify.xyz/login",
                       acoustic=np.random.randn(ACOUSTIC_DIM).astype(np.float32), mc_samples=8, explain=True)
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["url_analyzed"] == "http://ch4se-verify.xyz/login"
    assert set(res["modalities_present"]) == {"text", "acoustic", "url"}
    assert res["explanation"]["top_text_tokens"]
    assert res["uncertainty"]["ood_threshold"] == pytest.approx(thr)

    api.STATE.update(pipeline=pipe, error=None)
    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"] is True
    r = client.post("/v1/predict", json={"url": "http://micr0soft-support.top/claim", "mc_samples": 4})
    assert r.status_code == 200, r.text
    assert r.json()["modalities_present"] == ["url"]
    assert client.post("/v1/predict", json={}).status_code == 422


def test_audio_ignored_when_checkpoint_has_no_audio(encoder):
    from src.inference.pipeline import ScamShieldPipeline

    pipe = ScamShieldPipeline(build_student(), encoder)  # default 0/1 stats = trained without audio
    assert not pipe.acoustic_trained
    res = pipe.predict(text="Your KYC expires today, verify at http://bank-kyc.xyz",
                       acoustic=np.random.randn(ACOUSTIC_DIM).astype(np.float32), mc_samples=4)
    assert "acoustic" not in res["modalities_present"] and res["warnings"]
    with pytest.raises(ValueError):
        pipe.predict(acoustic=np.random.randn(ACOUSTIC_DIM).astype(np.float32), mc_samples=4)
