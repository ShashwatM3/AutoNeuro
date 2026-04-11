"""
Core model definition and training loop.

Three swappable graph neural network architectures for fMRI→EEG synthesis:
  - GCN_GAT: GCN layers + GAT attention heads + temporal convolutions
  - ST_GCN: Spatio-Temporal Graph Convolutional Network
  - GraphWaveNet: Adaptive adjacency + dilated causal temporal convolutions

All models share the same I/O contract:
  Input:  (batch, window_samples, 370, features)
  Output: (batch, window_samples, 64)  — 64-channel EEG at 250Hz

Prints METRIC= and VRAM_MB= lines for wrapper.sh to append to results.tsv.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from prepare import (
    N_TOTAL_NODES,
    TARGET_SR,
    GraphSignal,
    run_prepare,
)

logger = logging.getLogger(__name__)

N_EEG_CHANNELS = 64
MODEL_REGISTRY: dict[str, type] = {}


def register_model(cls: type) -> type:
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


# ===================================================================
# Shared: Graph-to-EEG spatial projection
# ===================================================================

class GraphToEEG(nn.Module):
    """Learnable spatial mixing: 370 graph nodes → 64 EEG channels."""

    def __init__(self, n_nodes: int = N_TOTAL_NODES, n_channels: int = N_EEG_CHANNELS,
                 hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_nodes * hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_channels),
        )
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, n_nodes, hidden_dim)
        Returns:
            (batch, T, 64)
        """
        B, T, N, H = x.shape
        x = x.reshape(B, T, N * H)
        return self.proj(x)


# ===================================================================
# Model 1: GCN_GAT
# ===================================================================

@register_model
class GCN_GAT(nn.Module):
    """GCN for local aggregation + GAT for adaptive attention + temporal conv."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 3,
                 temporal_kernel: int = 5):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.temporal_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GCNConv(dim_in, hidden_dim))
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // n_heads,
                                           heads=n_heads, concat=True))
            self.temporal_convs.append(nn.Conv1d(
                hidden_dim, hidden_dim, temporal_kernel,
                padding=temporal_kernel // 2
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.eeg_head = GraphToEEG(N_TOTAL_NODES, N_EEG_CHANNELS, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F) windowed graph signal.
            edge_index: (2, E) graph topology.
            edge_weight: (E,) edge weights.
        Returns:
            (B, T, 64) predicted EEG.
        """
        B, T, N, F = x.shape

        # Process each time-node slice through spatial graph layers
        x = x.reshape(B * T, N, F)
        for gcn, gat, norm in zip(self.gcn_layers, self.gat_layers, self.norms):
            # GCN
            x_flat = x.reshape(B * T * N, -1)
            # Expand edge_index for batched graphs
            ei_batch, ew_batch = _batch_edge_index(edge_index, edge_weight, B * T, N)
            h = gcn(x_flat, ei_batch, ew_batch)
            h = F.gelu(h)
            # GAT
            h = gat(h, ei_batch)
            h = h.reshape(B * T, N, -1)
            x = norm(h + x[..., :h.shape[-1]])  # residual

        # Temporal convolution: (B, T, N, H) → per-node 1D conv over time
        x = x.reshape(B, T, N, -1)
        H = x.shape[-1]
        # Reshape to (B*N, H, T) for Conv1d
        x = x.permute(0, 2, 3, 1).reshape(B * N, H, T)
        for tconv in self.temporal_convs:
            x = F.gelu(tconv(x)) + x  # residual
        x = x.reshape(B, N, H, T).permute(0, 3, 1, 2)  # (B, T, N, H)

        return self.eeg_head(x)


# ===================================================================
# Model 2: ST_GCN (Spatio-Temporal Graph Convolutional Network)
# ===================================================================

class STConvBlock(nn.Module):
    """Single spatio-temporal block: spatial GCN + temporal Conv1D."""

    def __init__(self, in_dim: int, out_dim: int, temporal_kernel: int = 3):
        super().__init__()
        self.spatial = GCNConv(in_dim, out_dim)
        self.temporal = nn.Conv1d(out_dim, out_dim, temporal_kernel,
                                  padding=temporal_kernel // 2)
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F)
            edge_index, edge_weight: graph structure.
        Returns:
            (B, T, N, out_dim)
        """
        B, T, N, F = x.shape
        res = self.residual(x)

        # Spatial: apply GCN per timestep
        x_flat = x.reshape(B * T * N, F)
        ei_batch, ew_batch = _batch_edge_index(edge_index, edge_weight, B * T, N)
        h = self.spatial(x_flat, ei_batch, ew_batch)
        h = F.gelu(h)
        h = h.reshape(B, T, N, -1)

        # Temporal: Conv1D per node across time
        H = h.shape[-1]
        h = h.permute(0, 2, 3, 1).reshape(B * N, H, T)  # (B*N, H, T)
        h = F.gelu(self.temporal(h))
        h = h.reshape(B, N, H, T).permute(0, 3, 1, 2)  # (B, T, N, H)

        return self.norm(h + res)


@register_model
class ST_GCN(nn.Module):
    """Spatio-Temporal GCN: interleaved spatial + temporal conv blocks."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64,
                 n_blocks: int = 4, temporal_kernel: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dim_in = in_dim if i == 0 else hidden_dim
            self.blocks.append(STConvBlock(dim_in, hidden_dim, temporal_kernel))
        self.eeg_head = GraphToEEG(N_TOTAL_NODES, N_EEG_CHANNELS, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index, edge_weight)
        return self.eeg_head(x)


# ===================================================================
# Model 3: GraphWaveNet
# ===================================================================

class DilatedCausalConv(nn.Module):
    """Dilated causal 1D convolution for temporal multi-scale modeling."""

    def __init__(self, channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal padding: pad only on the left
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class GraphWaveNetLayer(nn.Module):
    """One GraphWaveNet layer: adaptive GCN + dilated causal temporal conv."""

    def __init__(self, channels: int, n_nodes: int, dilation: int,
                 kernel_size: int = 2, dropout: float = 0.1):
        super().__init__()
        # Fixed graph conv (using DTI adjacency)
        self.gcn_fixed = GCNConv(channels, channels)
        # Adaptive graph: learnable node embeddings for self-discovered edges
        self.node_emb_src = nn.Parameter(torch.randn(n_nodes, 16))
        self.node_emb_tgt = nn.Parameter(torch.randn(n_nodes, 16))

        self.gcn_adaptive = GCNConv(channels, channels)

        # Temporal: gated dilated causal conv
        self.filter_conv = DilatedCausalConv(channels, kernel_size, dilation)
        self.gate_conv = DilatedCausalConv(channels, kernel_size, dilation)

        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.n_nodes = n_nodes

    def _adaptive_adj(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive adjacency from learned node embeddings."""
        adj = F.softmax(F.relu(self.node_emb_src @ self.node_emb_tgt.T), dim=-1)
        # Sparsify: keep top-k per row
        k = max(1, self.n_nodes // 10)
        topk_vals, topk_idx = adj.topk(k, dim=-1)
        sparse_adj = torch.zeros_like(adj)
        sparse_adj.scatter_(1, topk_idx, topk_vals)
        rows, cols = torch.where(sparse_adj > 0)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_weight = sparse_adj[rows, cols]
        return edge_index, edge_weight

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, N, H)
        Returns:
            residual: (B, T, N, H)
            skip: (B, T, N, H)
        """
        B, T, N, H = x.shape

        # --- Spatial: fixed DTI + adaptive ---
        x_flat = x.reshape(B * T * N, H)
        ei_fixed, ew_fixed = _batch_edge_index(edge_index, edge_weight, B * T, N)
        h_fixed = self.gcn_fixed(x_flat, ei_fixed, ew_fixed)

        # Adaptive adjacency (shared across batch)
        ei_adapt, ew_adapt = self._adaptive_adj()
        ei_adapt_b, ew_adapt_b = _batch_edge_index(ei_adapt, ew_adapt, B * T, N)
        h_adapt = self.gcn_adaptive(x_flat, ei_adapt_b, ew_adapt_b)

        h = (h_fixed + h_adapt).reshape(B, T, N, H)

        # --- Temporal: gated dilated causal conv ---
        h = h.permute(0, 2, 3, 1).reshape(B * N, H, T)  # (B*N, H, T)
        h_filter = torch.tanh(self.filter_conv(h))
        h_gate = torch.sigmoid(self.gate_conv(h))
        h = h_filter * h_gate
        h = self.dropout(h)

        skip = self.skip_proj(h).reshape(B, N, H, T).permute(0, 3, 1, 2)
        residual = self.residual_proj(h).reshape(B, N, H, T).permute(0, 3, 1, 2)

        residual = self.norm(residual + x)
        return residual, skip


@register_model
class GraphWaveNet(nn.Module):
    """GraphWaveNet: adaptive adjacency + dilated causal temporal convolutions.

    Combines DTI structural prior with learnable adaptive connections and
    multi-scale temporal receptive fields via exponentially dilated convolutions.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64,
                 n_layers: int = 8, kernel_size: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # cycle: 1, 2, 4, 8, 1, 2, 4, 8
            self.layers.append(GraphWaveNetLayer(
                hidden_dim, N_TOTAL_NODES, dilation, kernel_size, dropout
            ))

        self.end_conv = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eeg_head = GraphToEEG(N_TOTAL_NODES, N_EEG_CHANNELS, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        skip_sum = 0
        for layer in self.layers:
            x, skip = layer(x, edge_index, edge_weight)
            skip_sum = skip_sum + skip
        x = self.end_conv(skip_sum)
        return self.eeg_head(x)


# ===================================================================
# Utility: batched edge index for PyG
# ===================================================================

def _batch_edge_index(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                      batch_size: int, n_nodes: int
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate edge_index for a batch of independent graphs.

    Each graph in the batch gets its own copy of the adjacency with
    node indices offset by graph_idx * n_nodes.
    """
    device = edge_index.device
    offsets = torch.arange(batch_size, device=device).unsqueeze(1) * n_nodes  # (B, 1)
    # (B, 1) + (1, E) → (B, E) for both src and dst
    ei_src = (edge_index[0].unsqueeze(0) + offsets).reshape(-1)
    ei_dst = (edge_index[1].unsqueeze(0) + offsets).reshape(-1)
    ei_batched = torch.stack([ei_src, ei_dst], dim=0)
    ew_batched = edge_weight.repeat(batch_size)
    return ei_batched, ew_batched


# ===================================================================
# Training loop
# ===================================================================

def get_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def train_epoch(model: nn.Module, data: GraphSignal,
                optimizer: torch.optim.Optimizer,
                target_eeg: torch.Tensor) -> float:
    """Run one training epoch over windowed data.

    Args:
        model: one of the three GNN models.
        data: GraphSignal from prepare.py.
        optimizer: torch optimizer.
        target_eeg: (n_windows, window_size, 64) ground truth EEG.

    Returns:
        Mean loss for the epoch.
    """
    model.train()
    x = data.node_features  # (n_windows, T, 370, F)
    n_windows = x.shape[0]
    batch_size = 4
    losses = []

    indices = torch.randperm(n_windows)
    for start in range(0, n_windows, batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = x[batch_idx]
        y_batch = target_eeg[batch_idx]

        optimizer.zero_grad()
        pred = model(x_batch, data.edge_index, data.edge_weight)
        loss = F.mse_loss(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def run_train(
    model_name: str = "GraphWaveNet",
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Full training run: prepare data → train model → report metrics.

    Args:
        model_name: one of "GCN_GAT", "ST_GCN", "GraphWaveNet".
        n_epochs: number of training epochs.
        lr: learning rate.
        device: torch device string.
    """
    logger.info("Preparing data...")
    data = run_prepare(device=device)

    # Synthetic target EEG for now — replaced with real data in production
    n_windows = data.node_features.shape[0]
    window_size = data.node_features.shape[1]
    target_eeg = torch.randn(n_windows, window_size, N_EEG_CHANNELS, device=device)

    logger.info("Initializing model: %s", model_name)
    in_dim = data.node_features.shape[-1]
    model = get_model(model_name, in_dim=in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, data, optimizer, target_eeg)
        scheduler.step()
        elapsed = time.time() - t0

        if loss < best_loss:
            best_loss = loss

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d — loss: %.6f — best: %.6f — %.1fs",
                epoch, n_epochs, loss, best_loss, elapsed
            )

    # --- Report metrics for wrapper.sh ---
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)

    # Run final evaluation pass
    model.eval()
    with torch.no_grad():
        pred = model(data.node_features[:1], data.edge_index, data.edge_weight)
        final_loss = F.mse_loss(pred, target_eeg[:1]).item()

    # Pearson correlation as primary metric (higher is better)
    pred_np = pred.squeeze(0).cpu().numpy()
    target_np = target_eeg[0].cpu().numpy()
    correlations = []
    for ch in range(N_EEG_CHANNELS):
        r = np.corrcoef(pred_np[:, ch], target_np[:, ch])[0, 1]
        if not np.isnan(r):
            correlations.append(r)
    metric = float(np.mean(correlations)) if correlations else 0.0

    print(f"METRIC={metric:.6f}", flush=True)
    print(f"VRAM_MB={vram_mb}", flush=True)


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train fMRI→EEG model")
    parser.add_argument("--model", type=str, default="GraphWaveNet",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_train(
        model_name=args.model,
        n_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
