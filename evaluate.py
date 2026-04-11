"""
Evaluation metrics for fMRI→EEG synthesis.

Three metrics (NOT editable by the Coding Agent):
  1. Pearson correlation — per-channel temporal correlation
  2. ERP recovery — cross-correlation with known ERP templates (P300, N170)
  3. Spectral similarity — PSD cosine similarity across canonical bands

Composite score = weighted average of all three.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal as sig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 250  # Hz — target EEG sample rate

# Canonical EEG frequency bands (Hz)
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# Composite score weights
WEIGHTS = {
    "pearson": 0.4,
    "erp_recovery": 0.35,
    "spectral_similarity": 0.25,
}


@dataclass
class EvalResult:
    """Container for all evaluation metrics."""
    pearson: float
    erp_recovery: float
    spectral_similarity: float
    composite: float
    per_channel_pearson: list[float]
    per_band_spectral: dict[str, float]
    erp_details: dict[str, float]


# ===================================================================
# Metric 1: Pearson Correlation
# ===================================================================

def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> tuple[float, list[float]]:
    """Per-channel Pearson correlation, averaged across channels.

    Args:
        pred:   (T, 64) predicted EEG.
        target: (T, 64) ground-truth EEG.

    Returns:
        mean_r: scalar mean correlation.
        per_channel: list of 64 per-channel correlations.
    """
    n_channels = pred.shape[1]
    per_channel = []

    for ch in range(n_channels):
        p = pred[:, ch]
        t = target[:, ch]
        # Handle constant signals
        if np.std(p) < 1e-10 or np.std(t) < 1e-10:
            per_channel.append(0.0)
            continue
        r = np.corrcoef(p, t)[0, 1]
        per_channel.append(float(r) if not np.isnan(r) else 0.0)

    return float(np.mean(per_channel)), per_channel


# ===================================================================
# Metric 2: ERP Recovery
# ===================================================================

# Canonical ERP templates (simplified waveforms at 250Hz)
# These are idealized shapes — real evaluation should use empirically
# derived templates from the experimental paradigm.

def _p300_template(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a canonical P300 template: positive peak ~300ms post-stimulus.

    Shape: Gaussian-windowed sinusoid peaking at 300ms, duration ~600ms.
    """
    duration = 0.6  # seconds
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples)
    peak_t = 0.3
    sigma = 0.08
    template = np.exp(-((t - peak_t) ** 2) / (2 * sigma ** 2))
    return template / np.max(np.abs(template))


def _n170_template(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a canonical N170 template: negative peak ~170ms post-stimulus.

    Shape: inverted Gaussian peaking at 170ms, duration ~400ms.
    """
    duration = 0.4
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples)
    peak_t = 0.17
    sigma = 0.05
    template = -np.exp(-((t - peak_t) ** 2) / (2 * sigma ** 2))
    return template / np.max(np.abs(template))


ERP_TEMPLATES = {
    "P300": _p300_template,
    "N170": _n170_template,
}

# Approximate channel indices for ERP-relevant scalp regions (10-20 system)
# P300: centroparietal (Pz, CPz) — roughly channels 30-35 in a 64-ch montage
# N170: occipitotemporal (P7/P8, PO7/PO8) — roughly channels 50-55
ERP_CHANNEL_GROUPS = {
    "P300": list(range(30, 36)),
    "N170": list(range(50, 56)),
}


def _normalized_cross_correlation(signal_segment: np.ndarray,
                                   template: np.ndarray) -> float:
    """Peak normalized cross-correlation between a signal and template."""
    if len(signal_segment) < len(template):
        return 0.0

    template_norm = template / (np.linalg.norm(template) + 1e-10)
    corr = np.correlate(signal_segment, template_norm, mode="valid")
    signal_norms = np.array([
        np.linalg.norm(signal_segment[i:i + len(template)])
        for i in range(len(corr))
    ])
    signal_norms = np.maximum(signal_norms, 1e-10)
    ncc = corr / signal_norms
    return float(np.max(ncc))


def erp_recovery(pred: np.ndarray, target: np.ndarray,
                 event_onsets: list[int] | None = None,
                 sr: int = SAMPLE_RATE) -> tuple[float, dict[str, float]]:
    """Measure how well predicted EEG recovers known ERP components.

    For each ERP type:
      1. Average predicted signal across relevant channels and trials.
      2. Compute normalized cross-correlation with the canonical template.
      3. Compare against the same metric for the ground truth.

    Args:
        pred:   (T, 64) predicted EEG.
        target: (T, 64) ground-truth EEG.
        event_onsets: list of sample indices where stimuli occurred.
                      If None, uses evenly spaced pseudo-events.
        sr: sample rate in Hz.

    Returns:
        mean_recovery: average recovery score across ERP types.
        details: per-ERP-type recovery scores.
    """
    # Generate pseudo-events if none provided (every 2 seconds)
    if event_onsets is None:
        event_onsets = list(range(0, pred.shape[0] - sr, sr * 2))

    if not event_onsets:
        return 0.0, {name: 0.0 for name in ERP_TEMPLATES}

    details = {}

    for erp_name, template_fn in ERP_TEMPLATES.items():
        template = template_fn(sr)
        channels = ERP_CHANNEL_GROUPS.get(erp_name, list(range(pred.shape[1])))
        # Clamp channel indices to valid range
        channels = [ch for ch in channels if ch < pred.shape[1]]
        if not channels:
            details[erp_name] = 0.0
            continue

        epoch_len = len(template) + int(0.1 * sr)  # template + 100ms buffer

        # Extract and average epochs
        pred_epochs = []
        target_epochs = []
        for onset in event_onsets:
            end = onset + epoch_len
            if end > pred.shape[0]:
                continue
            pred_epochs.append(pred[onset:end, channels].mean(axis=1))
            target_epochs.append(target[onset:end, channels].mean(axis=1))

        if not pred_epochs:
            details[erp_name] = 0.0
            continue

        pred_erp = np.mean(pred_epochs, axis=0)
        target_erp = np.mean(target_epochs, axis=0)

        # Cross-correlation with template
        ncc_pred = _normalized_cross_correlation(pred_erp, template)
        ncc_target = _normalized_cross_correlation(target_erp, template)

        # Recovery = ratio of predicted NCC to ground-truth NCC (clamped to [0, 1])
        recovery = ncc_pred / (ncc_target + 1e-10) if ncc_target > 0 else ncc_pred
        details[erp_name] = float(np.clip(recovery, 0.0, 1.0))

    mean_recovery = float(np.mean(list(details.values()))) if details else 0.0
    return mean_recovery, details


# ===================================================================
# Metric 3: Spectral Similarity
# ===================================================================

def _bandpower(data: np.ndarray, sr: int, band: tuple[float, float]) -> float:
    """Compute average band power using Welch's method."""
    freqs, psd = sig.welch(data, fs=sr, nperseg=min(len(data), sr * 2))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0
    return float(np.mean(psd[band_mask]))


def spectral_similarity(pred: np.ndarray, target: np.ndarray,
                         sr: int = SAMPLE_RATE) -> tuple[float, dict[str, float]]:
    """Cosine similarity of power spectral density across canonical EEG bands.

    Computes PSD for each band (delta, theta, alpha, beta, gamma) on both
    predicted and target EEG, then measures cosine similarity of the
    resulting power vectors.

    Args:
        pred:   (T, 64) predicted EEG.
        target: (T, 64) ground-truth EEG.
        sr: sample rate in Hz.

    Returns:
        mean_similarity: average cosine similarity across bands.
        per_band: dict of per-band cosine similarities.
    """
    n_channels = pred.shape[1]
    per_band = {}

    for band_name, (flo, fhi) in FREQ_BANDS.items():
        pred_powers = np.array([_bandpower(pred[:, ch], sr, (flo, fhi))
                                for ch in range(n_channels)])
        target_powers = np.array([_bandpower(target[:, ch], sr, (flo, fhi))
                                  for ch in range(n_channels)])

        # Cosine similarity
        dot = np.dot(pred_powers, target_powers)
        norm_p = np.linalg.norm(pred_powers) + 1e-10
        norm_t = np.linalg.norm(target_powers) + 1e-10
        cos_sim = dot / (norm_p * norm_t)
        per_band[band_name] = float(np.clip(cos_sim, -1.0, 1.0))

    mean_sim = float(np.mean(list(per_band.values())))
    return mean_sim, per_band


# ===================================================================
# Composite Score
# ===================================================================

def evaluate(pred: np.ndarray, target: np.ndarray,
             event_onsets: list[int] | None = None,
             sr: int = SAMPLE_RATE) -> EvalResult:
    """Run full evaluation: Pearson + ERP recovery + spectral similarity.

    Args:
        pred:   (T, 64) predicted EEG time series at sr Hz.
        target: (T, 64) ground-truth EEG time series at sr Hz.
        event_onsets: optional stimulus onset sample indices for ERP.
        sr: sample rate.

    Returns:
        EvalResult with all metrics and composite score.
    """
    assert pred.shape == target.shape, (
        f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    )

    r_mean, r_per_ch = pearson_correlation(pred, target)
    erp_mean, erp_details = erp_recovery(pred, target, event_onsets, sr)
    spec_mean, spec_bands = spectral_similarity(pred, target, sr)

    composite = (
        WEIGHTS["pearson"] * r_mean
        + WEIGHTS["erp_recovery"] * erp_mean
        + WEIGHTS["spectral_similarity"] * spec_mean
    )

    logger.info("Pearson: %.4f | ERP: %.4f | Spectral: %.4f | Composite: %.4f",
                r_mean, erp_mean, spec_mean, composite)

    return EvalResult(
        pearson=r_mean,
        erp_recovery=erp_mean,
        spectral_similarity=spec_mean,
        composite=composite,
        per_channel_pearson=r_per_ch,
        per_band_spectral=spec_bands,
        erp_details=erp_details,
    )


def run_evaluate() -> None:
    """Standalone evaluation with synthetic data for testing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    T = 5000  # 20 seconds at 250Hz
    pred = np.random.randn(T, 64).astype(np.float32)
    target = np.random.randn(T, 64).astype(np.float32)
    result = evaluate(pred, target)
    print(f"Composite score: {result.composite:.4f}")
