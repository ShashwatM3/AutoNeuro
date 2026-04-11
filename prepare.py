"""
Data preprocessing pipeline: TRIBE v2 fMRI -> parcellated graph signals.

Stages:
  1. Spatial parcellation (Glasser360 cortical + 10 subcortical nodes)
  2. Graph construction (DTI structural connectivity, 370x370)
  3. Temporal processing (BOLD upsampling, stimulus embeddings, hybrid gating)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import scipy.interpolate as interp
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CORTICAL_VERTICES = 20_484      # fsaverage5 cortical mesh
N_SUBCORTICAL_VOXELS = 8_802      # TRIBE v2 subcortical output
N_GLASSER_PARCELS = 360           # 180 per hemisphere
N_THALAMUS_NODES = 6              # L/R x anterior/medial/posterior
N_HIPPOCAMPUS_NODES = 4           # L/R x anterior/posterior
N_SUBCORTICAL_NODES = N_THALAMUS_NODES + N_HIPPOCAMPUS_NODES  # 10
N_TOTAL_NODES = N_GLASSER_PARCELS + N_SUBCORTICAL_NODES       # 370
BOLD_SR = 1                       # Hz — TRIBE v2 output rate
TARGET_SR = 250                   # Hz — target EEG sample rate
SPARSITY_KEEP = 0.12              # top 12% of edges

CACHE_DIR = Path(".cache")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class GraphSignal:
    """Model-ready graph data coming out of prepare.py."""
    node_features: torch.Tensor       # (n_windows, window_samples, 370, n_feat)
    adjacency: torch.Tensor           # (370, 370) sparse or dense
    edge_index: torch.Tensor          # (2, n_edges) for PyG
    edge_weight: torch.Tensor         # (n_edges,)
    meta: dict = field(default_factory=dict)


# ===================================================================
# Stage 1 — Spatial Parcellation
# ===================================================================

def load_glasser_labels(hemi: Literal["lh", "rh"]) -> np.ndarray:
    """Load Glasser HCP-MMP1.0 atlas labels for one hemisphere on fsaverage5.

    Returns an array of shape (n_vertices_hemi,) with integer parcel IDs
    (1-180 for lh, 181-360 for rh). 0 = medial wall / unassigned.
    """
    # Glasser atlas on fsaverage5 — stored as FreeSurfer annot files
    # We ship them in .cache/atlases/ or fetch via nilearn
    atlas_dir = CACHE_DIR / "atlases" / "glasser"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    annot_path = atlas_dir / f"{hemi}.HCP-MMP1.annot"
    if annot_path.exists():
        labels, _, _ = nib.freesurfer.read_annot(str(annot_path))
        if hemi == "rh":
            # Shift rh labels to 181-360 range
            labels = np.where(labels > 0, labels + 180, 0)
        return labels

    # Fallback: try to load from FreeSurfer subjects dir
    fs_dir = Path.home() / "freesurfer" / "subjects" / "fsaverage5" / "label"
    annot_path_fs = fs_dir / f"{hemi}.HCP-MMP1.annot"
    if annot_path_fs.exists():
        labels, _, _ = nib.freesurfer.read_annot(str(annot_path_fs))
        if hemi == "rh":
            labels = np.where(labels > 0, labels + 180, 0)
        return labels

    raise FileNotFoundError(
        f"Glasser atlas annotation not found. Place {hemi}.HCP-MMP1.annot "
        f"in {atlas_dir} or the FreeSurfer fsaverage5 label directory."
    )


def parcellate_cortical(bold_ctx: np.ndarray) -> np.ndarray:
    """Mean-pool cortical vertices into 360 Glasser parcels.

    Args:
        bold_ctx: (n_timesteps, 20484) — TRIBE v2 cortical predictions.

    Returns:
        (n_timesteps, 360) — one signal per Glasser parcel.
    """
    n_t = bold_ctx.shape[0]
    assert bold_ctx.shape[1] == N_CORTICAL_VERTICES, (
        f"Expected {N_CORTICAL_VERTICES} cortical vertices, got {bold_ctx.shape[1]}"
    )

    # fsaverage5: 10242 vertices per hemisphere
    n_verts_hemi = N_CORTICAL_VERTICES // 2
    lh_labels = load_glasser_labels("lh")[:n_verts_hemi]
    rh_labels = load_glasser_labels("rh")[:n_verts_hemi]
    labels = np.concatenate([lh_labels, rh_labels])  # (20484,)

    parcellated = np.zeros((n_t, N_GLASSER_PARCELS), dtype=bold_ctx.dtype)
    for parcel_id in range(1, N_GLASSER_PARCELS + 1):
        mask = labels == parcel_id
        if mask.sum() > 0:
            parcellated[:, parcel_id - 1] = bold_ctx[:, mask].mean(axis=1)
        else:
            logger.warning("Parcel %d has no vertices — filling with zeros", parcel_id)

    return parcellated


# ---------------------------------------------------------------------------
# Subcortical node extraction
# ---------------------------------------------------------------------------

# FreeSurfer aseg label IDs for structures of interest
_ASEG_THALAMUS_L = 10
_ASEG_THALAMUS_R = 49
_ASEG_HIPPOCAMPUS_L = 17
_ASEG_HIPPOCAMPUS_R = 53


def _load_subcortical_lookup() -> np.ndarray:
    """Load the voxel-to-aseg-label mapping for TRIBE v2's 8802 subcortical voxels.

    Returns:
        labels: (8802,) integer array of FreeSurfer aseg label IDs.
    """
    lookup_path = CACHE_DIR / "atlases" / "tribe_subcortical_aseg_labels.npy"
    if lookup_path.exists():
        return np.load(lookup_path)

    raise FileNotFoundError(
        f"Subcortical label lookup not found at {lookup_path}. "
        "Generate it by mapping TRIBE v2's subcortical voxel coordinates "
        "to FreeSurfer aseg labels in MNI152 space."
    )


def _load_subcortical_coords() -> np.ndarray:
    """Load MNI coordinates for TRIBE v2's 8802 subcortical voxels.

    Returns:
        coords: (8802, 3) — x, y, z in MNI152 space.
    """
    coords_path = CACHE_DIR / "atlases" / "tribe_subcortical_coords_mni.npy"
    if coords_path.exists():
        return np.load(coords_path)

    raise FileNotFoundError(
        f"Subcortical coordinates not found at {coords_path}. "
        "Extract MNI coordinates from TRIBE v2's subcortical voxel grid."
    )


def _split_by_y_thirds(coords_y: np.ndarray) -> np.ndarray:
    """Split voxels into anterior/medial/posterior thirds along the y-axis.

    Returns:
        segment: (n_voxels,) with values 0=anterior, 1=medial, 2=posterior.
    """
    terciles = np.percentile(coords_y, [33.3, 66.6])
    segment = np.zeros(len(coords_y), dtype=int)
    segment[coords_y <= terciles[0]] = 0   # anterior (most positive y in MNI)
    segment[(coords_y > terciles[0]) & (coords_y <= terciles[1])] = 1  # medial
    segment[coords_y > terciles[1]] = 2    # posterior
    return segment


def _split_by_y_half(coords_y: np.ndarray) -> np.ndarray:
    """Split voxels into anterior/posterior halves along the y-axis (uncal apex).

    Returns:
        segment: (n_voxels,) with values 0=anterior, 1=posterior.
    """
    median_y = np.median(coords_y)
    return (coords_y > median_y).astype(int)


def parcellate_subcortical(bold_sctx: np.ndarray) -> np.ndarray:
    """Extract 10 subcortical nodes from TRIBE v2's 8802 subcortical voxels.

    Thalamus: L/R x anterior/medial/posterior = 6 nodes
    Hippocampus: L/R x anterior/posterior = 4 nodes

    Args:
        bold_sctx: (n_timesteps, 8802) — TRIBE v2 subcortical predictions.

    Returns:
        (n_timesteps, 10) — mean-pooled subcortical node signals.

    Node order:
        [0] L-Thalamus-Anterior    [1] L-Thalamus-Medial     [2] L-Thalamus-Posterior
        [3] R-Thalamus-Anterior    [4] R-Thalamus-Medial     [5] R-Thalamus-Posterior
        [6] L-Hippocampus-Anterior [7] L-Hippocampus-Posterior
        [8] R-Hippocampus-Anterior [9] R-Hippocampus-Posterior
    """
    n_t = bold_sctx.shape[0]
    assert bold_sctx.shape[1] == N_SUBCORTICAL_VOXELS, (
        f"Expected {N_SUBCORTICAL_VOXELS} subcortical voxels, got {bold_sctx.shape[1]}"
    )

    aseg_labels = _load_subcortical_lookup()      # (8802,)
    coords = _load_subcortical_coords()            # (8802, 3)

    nodes = np.zeros((n_t, N_SUBCORTICAL_NODES), dtype=bold_sctx.dtype)

    # --- Thalamus (6 nodes) ---
    for hemi_idx, (aseg_id, label_prefix) in enumerate([
        (_ASEG_THALAMUS_L, "L"), (_ASEG_THALAMUS_R, "R")
    ]):
        mask = aseg_labels == aseg_id
        if mask.sum() == 0:
            logger.warning("No voxels found for %s-Thalamus (aseg=%d)", label_prefix, aseg_id)
            continue
        y_coords = coords[mask, 1]
        segments = _split_by_y_thirds(y_coords)
        for seg in range(3):  # anterior, medial, posterior
            seg_mask_local = segments == seg
            if seg_mask_local.sum() == 0:
                continue
            # Map local segment mask back to full voxel indices
            full_indices = np.where(mask)[0][seg_mask_local]
            node_idx = hemi_idx * 3 + seg
            nodes[:, node_idx] = bold_sctx[:, full_indices].mean(axis=1)

    # --- Hippocampus (4 nodes) ---
    for hemi_idx, (aseg_id, label_prefix) in enumerate([
        (_ASEG_HIPPOCAMPUS_L, "L"), (_ASEG_HIPPOCAMPUS_R, "R")
    ]):
        mask = aseg_labels == aseg_id
        if mask.sum() == 0:
            logger.warning("No voxels found for %s-Hippocampus (aseg=%d)", label_prefix, aseg_id)
            continue
        y_coords = coords[mask, 1]
        segments = _split_by_y_half(y_coords)
        for seg in range(2):  # anterior, posterior
            seg_mask_local = segments == seg
            if seg_mask_local.sum() == 0:
                continue
            full_indices = np.where(mask)[0][seg_mask_local]
            node_idx = N_THALAMUS_NODES + hemi_idx * 2 + seg
            nodes[:, node_idx] = bold_sctx[:, full_indices].mean(axis=1)

    return nodes


def parcellate(bold_ctx: np.ndarray, bold_sctx: np.ndarray) -> np.ndarray:
    """Full spatial parcellation: cortical + subcortical → 370 nodes.

    Args:
        bold_ctx:  (n_timesteps, 20484) cortical vertices.
        bold_sctx: (n_timesteps, 8802) subcortical voxels.

    Returns:
        (n_timesteps, 370) parcellated BOLD signal.
    """
    cortical = parcellate_cortical(bold_ctx)    # (T, 360)
    subcortical = parcellate_subcortical(bold_sctx)  # (T, 10)
    return np.concatenate([cortical, subcortical], axis=1)  # (T, 370)


# ===================================================================
# Stage 2 — Graph Construction (DTI Structural Connectivity)
# ===================================================================

def _load_enigma_sc() -> tuple[np.ndarray, np.ndarray]:
    """Load HCP group-average structural connectivity from ENIGMA Toolbox.

    Returns:
        sc_ctx:  (360, 360) cortico-cortical structural connectivity.
        sc_sctx: (14, 360) subcortico-cortical structural connectivity.
    """
    from enigmatoolbox.datasets import load_sc
    sc_ctx, _, sc_sctx, _ = load_sc(
        parcellation="glasser_360"
    )
    return np.array(sc_ctx), np.array(sc_sctx)


# ENIGMA subcortical label order (14 regions, 7 per hemisphere):
# L-Accumbens, L-Amygdala, L-Caudate, L-Hippocampus, L-Pallidum,
# L-Putamen, L-Thalamus, R-Accumbens, R-Amygdala, R-Caudate,
# R-Hippocampus, R-Pallidum, R-Putamen, R-Thalamus
_ENIGMA_THALAMUS_L_IDX = 6
_ENIGMA_THALAMUS_R_IDX = 13
_ENIGMA_HIPPOCAMPUS_L_IDX = 3
_ENIGMA_HIPPOCAMPUS_R_IDX = 10


def _expand_subcortical_connectivity(
    sc_sctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand ENIGMA's 14-region subcortical connectivity to our 10-node scheme.

    Thalamus L → 3 nodes (ant/med/post), each gets parent weight / 3.
    Thalamus R → 3 nodes.
    Hippocampus L → 2 nodes (ant/post), each gets parent weight / 2.
    Hippocampus R → 2 nodes.

    Returns:
        sctx_to_ctx: (10, 360) subcortical-to-cortical connectivity.
        sctx_to_sctx: (10, 10) subcortical-to-subcortical connectivity.
    """
    sctx_to_ctx = np.zeros((N_SUBCORTICAL_NODES, N_GLASSER_PARCELS), dtype=np.float64)

    # Thalamus L → nodes 0, 1, 2
    thal_l = sc_sctx[_ENIGMA_THALAMUS_L_IDX, :]
    for i in range(3):
        sctx_to_ctx[i, :] = thal_l / 3.0

    # Thalamus R → nodes 3, 4, 5
    thal_r = sc_sctx[_ENIGMA_THALAMUS_R_IDX, :]
    for i in range(3):
        sctx_to_ctx[3 + i, :] = thal_r / 3.0

    # Hippocampus L → nodes 6, 7
    hipp_l = sc_sctx[_ENIGMA_HIPPOCAMPUS_L_IDX, :]
    for i in range(2):
        sctx_to_ctx[6 + i, :] = hipp_l / 2.0

    # Hippocampus R → nodes 8, 9
    hipp_r = sc_sctx[_ENIGMA_HIPPOCAMPUS_R_IDX, :]
    for i in range(2):
        sctx_to_ctx[8 + i, :] = hipp_r / 2.0

    # Subcortical-to-subcortical: thalamo-hippocampal connections within hemisphere
    sctx_to_sctx = np.zeros((N_SUBCORTICAL_NODES, N_SUBCORTICAL_NODES), dtype=np.float64)
    # L thalamus <-> L hippocampus (known anatomical connection)
    for t in range(3):     # thalamus nodes
        for h in range(2):  # hippocampus nodes
            # Use geometric mean of their cortical connectivity as proxy
            weight = np.sqrt(
                np.dot(sctx_to_ctx[t, :], sctx_to_ctx[6 + h, :])
            )
            sctx_to_sctx[t, 6 + h] = weight
            sctx_to_sctx[6 + h, t] = weight
    # R thalamus <-> R hippocampus
    for t in range(3):
        for h in range(2):
            weight = np.sqrt(
                np.dot(sctx_to_ctx[3 + t, :], sctx_to_ctx[8 + h, :])
            )
            sctx_to_sctx[3 + t, 8 + h] = weight
            sctx_to_sctx[8 + h, 3 + t] = weight

    return sctx_to_ctx, sctx_to_sctx


def build_adjacency(sparsity_keep: float = SPARSITY_KEEP) -> torch.Tensor:
    """Construct the 370x370 adjacency matrix from DTI structural connectivity.

    Steps:
        1. Load ENIGMA HCP group-average SC (Glasser360 + 14 subcortical).
        2. Expand subcortical from 14 regions to 10 nodes.
        3. Assemble full 370x370 matrix.
        4. Log-normalize: A = log(1 + A).
        5. Sparsity threshold: keep top `sparsity_keep` fraction of edges.
        6. Zero out connections not in the DTI data.

    Returns:
        adj: (370, 370) dense tensor, symmetric, non-negative.
    """
    sc_ctx, sc_sctx = _load_enigma_sc()
    sctx_to_ctx, sctx_to_sctx = _expand_subcortical_connectivity(sc_sctx)

    # Assemble full matrix
    adj = np.zeros((N_TOTAL_NODES, N_TOTAL_NODES), dtype=np.float64)
    # Cortical-cortical block (top-left 360x360)
    adj[:N_GLASSER_PARCELS, :N_GLASSER_PARCELS] = sc_ctx
    # Subcortical-cortical blocks
    adj[N_GLASSER_PARCELS:, :N_GLASSER_PARCELS] = sctx_to_ctx
    adj[:N_GLASSER_PARCELS, N_GLASSER_PARCELS:] = sctx_to_ctx.T
    # Subcortical-subcortical block
    adj[N_GLASSER_PARCELS:, N_GLASSER_PARCELS:] = sctx_to_sctx

    # Symmetrize (should already be, but ensure)
    adj = (adj + adj.T) / 2.0

    # Log normalization
    adj = np.log1p(adj)

    # Sparsity thresholding — keep top fraction of non-zero edges
    nonzero_vals = adj[adj > 0]
    if len(nonzero_vals) > 0:
        threshold = np.percentile(nonzero_vals, (1 - sparsity_keep) * 100)
        adj[adj < threshold] = 0.0

    # Normalize edge weights to [0, 1]
    max_val = adj.max()
    if max_val > 0:
        adj = adj / max_val

    return torch.tensor(adj, dtype=torch.float32)


def adjacency_to_edge_index(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense adjacency matrix to PyG edge_index + edge_weight.

    Args:
        adj: (N, N) dense adjacency.

    Returns:
        edge_index: (2, E) long tensor.
        edge_weight: (E,) float tensor.
    """
    rows, cols = torch.where(adj > 0)
    edge_index = torch.stack([rows, cols], dim=0).long()
    edge_weight = adj[rows, cols]
    return edge_index, edge_weight


# ===================================================================
# Stage 3 — Temporal Processing
# ===================================================================

def upsample_bold(bold_parcellated: np.ndarray,
                  source_sr: int = BOLD_SR,
                  target_sr: int = TARGET_SR) -> np.ndarray:
    """Upsample BOLD from 1Hz to 250Hz via cubic spline interpolation.

    Args:
        bold_parcellated: (n_timesteps, 370) at source_sr Hz.

    Returns:
        (n_timesteps * target_sr / source_sr, 370) at target_sr Hz.
    """
    n_t, n_nodes = bold_parcellated.shape
    t_source = np.arange(n_t) / source_sr  # seconds
    n_target = int(n_t * target_sr / source_sr)
    t_target = np.arange(n_target) / target_sr

    upsampled = np.zeros((n_target, n_nodes), dtype=np.float32)
    for node in range(n_nodes):
        cs = interp.CubicSpline(t_source, bold_parcellated[:, node])
        upsampled[:, node] = cs(t_target)

    return upsampled


# ---------------------------------------------------------------------------
# Stimulus feature extraction
# ---------------------------------------------------------------------------

def extract_audio_features(audio_path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Extract mel spectrogram features from audio at target sample rate.

    Args:
        audio_path: path to audio file (wav, mp3, etc.)

    Returns:
        (n_samples, 128) mel spectrogram features at target_sr Hz.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=target_sr)
    # Mel spectrogram with hop_length=1 sample for 250Hz resolution
    # Use hop_length = sr // target_sr to get one frame per target sample
    hop_length = max(1, sr // target_sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, hop_length=hop_length, n_fft=1024
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # (n_frames, 128)


def extract_video_features(video_path: str | Path,
                           target_sr: int = TARGET_SR) -> np.ndarray:
    """Extract per-frame features from video and resample to target_sr.

    Uses a lightweight frame-level representation (pixel mean per channel
    as a baseline — swap for a real feature extractor in production).

    Args:
        video_path: path to video file.

    Returns:
        (n_samples, feat_dim) features at target_sr Hz.
    """
    import torchvision.io as vio

    video, _, info = vio.read_video(str(video_path), pts_unit="sec")
    # video: (n_frames, H, W, C) uint8
    fps = info["video_fps"]
    n_frames = video.shape[0]

    # Simple spatial average per frame → (n_frames, 3)
    frame_features = video.float().mean(dim=(1, 2)).numpy()  # (n_frames, 3)

    # Resample from fps → target_sr
    t_source = np.arange(n_frames) / fps
    duration = t_source[-1]
    n_target = int(duration * target_sr)
    t_target = np.arange(n_target) / target_sr

    resampled = np.zeros((n_target, frame_features.shape[1]), dtype=np.float32)
    for ch in range(frame_features.shape[1]):
        cs = interp.CubicSpline(t_source, frame_features[:, ch])
        resampled[:, ch] = cs(t_target)

    return resampled


def extract_text_features(word_onsets: list[float],
                          word_embeddings: np.ndarray,
                          duration_s: float,
                          target_sr: int = TARGET_SR) -> np.ndarray:
    """Spread word embeddings across time at target_sr.

    Each word's embedding is placed at its onset time and held until the
    next word onset.

    Args:
        word_onsets: list of onset times in seconds.
        word_embeddings: (n_words, embed_dim) pre-computed embeddings.
        duration_s: total duration in seconds.

    Returns:
        (n_samples, embed_dim) at target_sr Hz.
    """
    n_samples = int(duration_s * target_sr)
    embed_dim = word_embeddings.shape[1]
    features = np.zeros((n_samples, embed_dim), dtype=np.float32)

    for i, onset in enumerate(word_onsets):
        start = int(onset * target_sr)
        end = int(word_onsets[i + 1] * target_sr) if i + 1 < len(word_onsets) else n_samples
        start = max(0, min(start, n_samples))
        end = max(0, min(end, n_samples))
        features[start:end, :] = word_embeddings[i]

    return features


# ---------------------------------------------------------------------------
# Stimulus embedding projection
# ---------------------------------------------------------------------------

class StimulusProjector(torch.nn.Module):
    """Project heterogeneous stimulus features to a shared embedding space."""

    def __init__(self, input_dims: dict[str, int], embed_dim: int = 64):
        super().__init__()
        self.projectors = torch.nn.ModuleDict({
            name: torch.nn.Linear(dim, embed_dim)
            for name, dim in input_dims.items()
        })
        self.embed_dim = embed_dim

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Project and sum all modality features.

        Args:
            features: {modality_name: (T, input_dim)} tensors.

        Returns:
            (T, embed_dim) combined stimulus embedding.
        """
        projected = []
        for name, feat in features.items():
            if name in self.projectors:
                projected.append(self.projectors[name](feat))
        if not projected:
            raise ValueError("No matching modality projectors found")
        return torch.stack(projected, dim=0).sum(dim=0)


# ---------------------------------------------------------------------------
# Hybrid temporal gating
# ---------------------------------------------------------------------------

class HybridTemporalGate(torch.nn.Module):
    """Fuse upsampled BOLD (spatial prior) with stimulus embeddings (temporal driver).

    Gate: output_t = sigma(W_bold * bold_t) * (W_stim * stim_t)

    BOLD controls *which* nodes are active (spatial envelope).
    Stimulus controls *when* and *how* they activate (temporal dynamics).
    """

    def __init__(self, n_nodes: int = N_TOTAL_NODES, stim_dim: int = 64,
                 hidden_dim: int = 64):
        super().__init__()
        self.bold_gate = torch.nn.Linear(1, hidden_dim)   # per-node scalar → gate
        self.stim_proj = torch.nn.Linear(stim_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

    def forward(self, bold: torch.Tensor, stim: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bold: (T, 370) upsampled BOLD signal.
            stim: (T, stim_dim) stimulus embedding.

        Returns:
            (T, 370, hidden_dim) gated node features.
        """
        T = bold.shape[0]
        # BOLD gate: (T, 370, 1) → (T, 370, hidden_dim)
        bold_expanded = bold.unsqueeze(-1)                          # (T, 370, 1)
        gate = torch.sigmoid(self.bold_gate(bold_expanded))         # (T, 370, H)

        # Stimulus: (T, stim_dim) → (T, hidden_dim) → broadcast to (T, 370, H)
        stim_h = self.stim_proj(stim)                               # (T, H)
        stim_h = stim_h.unsqueeze(1).expand(T, self.n_nodes, -1)    # (T, 370, H)

        # Gated fusion
        fused = gate * stim_h                                       # (T, 370, H)
        return self.out_proj(fused)                                  # (T, 370, H)


# ===================================================================
# Stage 4 — Windowing
# ===================================================================

def window_signal(signal: torch.Tensor, window_size: int = 250,
                  stride: int = 125) -> torch.Tensor:
    """Chop a time-series into overlapping windows for batched training.

    Args:
        signal: (T, N_NODES, FEAT) continuous graph signal.
        window_size: samples per window (default 250 = 1 second at 250Hz).
        stride: hop between windows (default 125 = 50% overlap).

    Returns:
        (n_windows, window_size, N_NODES, FEAT)
    """
    T = signal.shape[0]
    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(signal[start : start + window_size])
    return torch.stack(windows, dim=0)


# ===================================================================
# Main entry point
# ===================================================================

def run_prepare(
    bold_ctx: np.ndarray | None = None,
    bold_sctx: np.ndarray | None = None,
    stimulus_features: dict[str, np.ndarray] | None = None,
    window_size: int = 250,
    window_stride: int = 125,
    stim_embed_dim: int = 64,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> GraphSignal:
    """Run the full preprocessing pipeline.

    Args:
        bold_ctx: (n_timesteps, 20484) TRIBE v2 cortical output.
        bold_sctx: (n_timesteps, 8802) TRIBE v2 subcortical output.
        stimulus_features: dict of {modality: (n_samples_250hz, feat_dim)} arrays.
        window_size: samples per training window.
        window_stride: hop between windows.
        stim_embed_dim: shared stimulus embedding dimension.
        hidden_dim: output feature dimension per node.
        device: torch device.

    Returns:
        GraphSignal with model-ready tensors.
    """
    # --- Demo / test mode with synthetic data ---
    if bold_ctx is None:
        logger.info("No input data provided — generating synthetic demo data")
        n_t_demo = 60  # 60 seconds
        bold_ctx = np.random.randn(n_t_demo, N_CORTICAL_VERTICES).astype(np.float32)
        bold_sctx = np.random.randn(n_t_demo, N_SUBCORTICAL_VOXELS).astype(np.float32)
        stimulus_features = {
            "audio": np.random.randn(n_t_demo * TARGET_SR, 128).astype(np.float32),
        }

    # Stage 1: Spatial parcellation
    logger.info("Stage 1: Spatial parcellation → 370 nodes")
    parcellated = parcellate(bold_ctx, bold_sctx)  # (T_bold, 370)

    # Stage 2: Graph construction
    logger.info("Stage 2: Building DTI adjacency matrix (370×370)")
    adj = build_adjacency()  # (370, 370)
    edge_index, edge_weight = adjacency_to_edge_index(adj)

    # Stage 3a: Upsample BOLD
    logger.info("Stage 3a: Upsampling BOLD 1Hz → 250Hz")
    bold_upsampled = upsample_bold(parcellated)  # (T_bold * 250, 370)
    bold_tensor = torch.tensor(bold_upsampled, dtype=torch.float32, device=device)

    # Stage 3b: Stimulus embeddings
    logger.info("Stage 3b: Projecting stimulus features")
    stim_tensors = {}
    input_dims = {}
    for name, feat in stimulus_features.items():
        t_feat = torch.tensor(feat, dtype=torch.float32, device=device)
        stim_tensors[name] = t_feat
        input_dims[name] = feat.shape[1]

    projector = StimulusProjector(input_dims, embed_dim=stim_embed_dim).to(device)
    stim_embedded = projector(stim_tensors)  # (T_250, stim_embed_dim)

    # Align lengths (truncate to shorter)
    min_len = min(bold_tensor.shape[0], stim_embedded.shape[0])
    bold_tensor = bold_tensor[:min_len]
    stim_embedded = stim_embedded[:min_len]

    # Stage 3c: Hybrid temporal gating
    logger.info("Stage 3c: Hybrid temporal gating")
    gate = HybridTemporalGate(
        n_nodes=N_TOTAL_NODES,
        stim_dim=stim_embed_dim,
        hidden_dim=hidden_dim
    ).to(device)
    gated_signal = gate(bold_tensor, stim_embedded)  # (T, 370, hidden_dim)

    # Stage 4: Windowing
    logger.info("Stage 4: Windowing signal (size=%d, stride=%d)", window_size, window_stride)
    windowed = window_signal(gated_signal, window_size, window_stride)
    # (n_windows, window_size, 370, hidden_dim)

    logger.info(
        "Prepare complete: %d windows of shape (%d, %d, %d)",
        windowed.shape[0], windowed.shape[1], windowed.shape[2], windowed.shape[3]
    )

    return GraphSignal(
        node_features=windowed,
        adjacency=adj,
        edge_index=edge_index.to(device),
        edge_weight=edge_weight.to(device),
        meta={
            "n_windows": windowed.shape[0],
            "window_size": window_size,
            "window_stride": window_stride,
            "n_nodes": N_TOTAL_NODES,
            "feature_dim": hidden_dim,
            "target_sr": TARGET_SR,
        },
    )
