"""Acoustic (paralinguistic) features from a call recording with Librosa.

Produces the 50-d vector described by ``config.ACOUSTIC_FEATURE_NAMES``:

* MFCC 1..20: mean and standard deviation over frames (40 values) — timbre /
  channel characteristics; synthetic or replayed voices and VoIP call-centre
  audio shift these statistics.
* Pitch (F0) from probabilistic YIN: mean, std over voiced frames, voiced ratio.
* Jitter (local): mean |T_i - T_{i-1}| / mean T over consecutive voiced frames,
  where T = 1 / F0 is the period. Frame-level approximation of the
  cycle-to-cycle measure Praat computes; scripted / stressed / TTS speech
  tends to have atypical jitter.
* Shimmer (local): the same ratio on frame RMS amplitude at voiced frames.
* RMS energy mean / std, zero-crossing rate, spectral centroid, spectral flatness.

Raw (un-standardised) values are returned; the model standardises them with
mean / std buffers fitted on the training set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ACOUSTIC_DIM, N_MFCC

SAMPLE_RATE = 16_000
MAX_SECONDS = 60.0


def _relative_mean_abs_diff(x: np.ndarray) -> float:
    if x.size < 2 or float(np.mean(x)) <= 0:
        return 0.0
    return float(np.mean(np.abs(np.diff(x))) / np.mean(x))


def extract_acoustic_features_from_signal(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    y = y[: int(MAX_SECONDS * sr)]
    if y.size < sr // 10:
        raise ValueError("audio is shorter than 0.1 s")

    hop = 256
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]

    f0, voiced, _ = librosa.pyin(y, fmin=65.0, fmax=400.0, sr=sr, hop_length=hop)
    voiced = voiced & np.isfinite(f0)
    f0_v = f0[voiced]
    n = min(len(rms), len(voiced))
    amp_v = rms[:n][voiced[:n]]

    if f0_v.size:
        pitch_mean, pitch_std = float(np.mean(f0_v)), float(np.std(f0_v))
        jitter = _relative_mean_abs_diff(1.0 / f0_v)
    else:
        pitch_mean = pitch_std = jitter = 0.0
    shimmer = _relative_mean_abs_diff(amp_v)

    feats = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [
            pitch_mean,
            pitch_std,
            float(np.mean(voiced)),
            jitter,
            shimmer,
            float(rms.mean()),
            float(rms.std()),
            float(zcr.mean()),
            float(centroid.mean()),
            float(flatness.mean()),
        ],
    ]).astype(np.float32)
    assert feats.shape == (ACOUSTIC_DIM,)
    return np.nan_to_num(feats)


def extract_acoustic_features(path: str | Path) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    return extract_acoustic_features_from_signal(y, sr)
