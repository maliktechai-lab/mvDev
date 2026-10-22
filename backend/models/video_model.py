"""
DeepGuard-Video Model
=====================
Four-branch architecture for detecting fake videos:

  Branch 1 — Spatial:    EfficientNet-B4 per frame (pixel artifacts)
  Branch 2 — Frequency:  FFT CNN per frame (GAN fingerprints over time)
  Branch 3 — Temporal:   LSTM over frame embedding sequence (motion/blink/pose)
  Branch 4 — Biological: rPPG signal extraction (blood flow patterns)

All four branch outputs are fused via a learned MLP head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Spatial branch (per-frame EfficientNet) ───────────────────────────────

class VideoSpatialBranch(nn.Module):
    """
    Applies EfficientNet-B4 frame-by-frame.
    Returns per-frame feature vectors AND a temporal average pooling.
    """

    def __init__(self, out_features: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features   # 1792 for B4
        self.project = nn.Sequential(
            nn.Linear(feat_dim, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: (B, T, C, H, W) — batch of frame sequences

        Returns:
            frame_feats: (B, T, out_features) — per-frame embeddings
            pooled:      (B, out_features)    — temporal average
        """
        B, T, C, H, W = frames.shape
        # Flatten batch+time for efficient backbone pass
        flat = frames.view(B * T, C, H, W)
        feat = self.backbone(flat)             # (B*T, 1792)
        feat = self.project(feat)              # (B*T, out_features)
        frame_feats = feat.view(B, T, -1)      # (B, T, out_features)
        pooled = frame_feats.mean(dim=1)       # (B, out_features)
        return frame_feats, pooled


# ── Frequency branch (per-frame FFT) ─────────────────────────────────────

class VideoFrequencyBranch(nn.Module):
    """
    Computes FFT per frame then uses a lightweight CNN + GRU to capture
    how frequency artifacts evolve over the sequence.
    """

    def __init__(self, in_channels: int = 3, out_features: int = 512):
        super().__init__()
        # Shared frame-level CNN for frequency features
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        frame_feat_dim = 128 * 4 * 4   # 2048

        self.project = nn.Linear(frame_feat_dim, 256)

        # GRU to capture temporal frequency patterns
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=out_features // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        # bidirectional: output is out_features

    @staticmethod
    def _fft_magnitude(x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(x, norm="ortho")
        return torch.log1p(torch.abs(torch.fft.fftshift(fft)))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)
        Returns:
            (B, out_features)
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)

        freq_maps = self._fft_magnitude(flat)          # (B*T, C, H, W)
        frame_feat = self.frame_cnn(freq_maps)          # (B*T, 2048)
        frame_feat = F.relu(self.project(frame_feat))   # (B*T, 256)
        frame_seq  = frame_feat.view(B, T, 256)         # (B, T, 256)

        gru_out, _ = self.gru(frame_seq)               # (B, T, out_features)
        # Take last hidden state from both directions
        return gru_out[:, -1, :]                        # (B, out_features)


# ── Temporal branch (LSTM over spatial embeddings) ────────────────────────

class TemporalBranch(nn.Module):
    """
    Bidirectional LSTM over the sequence of spatial frame embeddings.
    Captures temporal patterns: blink rate, head pose jitter, expression jumps.
    """

    def __init__(self, input_size: int = 512, hidden_size: int = 256, out_features: int = 512):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        # Attention pooling over LSTM outputs
        self.attn = nn.Linear(hidden_size * 2, 1)

        self.project = nn.Sequential(
            nn.Linear(hidden_size * 2, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_feats: (B, T, input_size) — per-frame spatial embeddings

        Returns:
            (B, out_features)
        """
        lstm_out, _ = self.lstm(frame_feats)    # (B, T, hidden*2)

        # Attention-weighted pooling over time steps
        scores = self.attn(lstm_out)            # (B, T, 1)
        weights = F.softmax(scores, dim=1)      # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)  # (B, hidden*2)

        return self.project(context)            # (B, out_features)


# ── Biological branch (rPPG signal) ──────────────────────────────────────

class BiologicalBranch(nn.Module):
    """
    Remote Photoplethysmography (rPPG) branch.
    Extracts subtle skin color variations caused by blood flow.
    Real faces have rhythmic pulse signals; deepfakes often lack them.

    Implementation:
      1. Extract mean RGB from forehead ROI (top-center of face) per frame
      2. Compute temporal differences to get pulse-like signal
      3. 1D CNN + GRU to classify signal as real/fake
    """

    def __init__(self, out_features: int = 256):
        super().__init__()
        # 1D CNN over the rPPG time series (3 channels: R, G, B)
        self.signal_cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=out_features // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

    @staticmethod
    def _extract_rppg_signal(frames: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate rPPG signal from frame sequence.
        Uses mean RGB of the forehead region (top-center 20% of face crop).

        Args:
            frames: (B, T, C, H, W) — face crops normalized to [0,1] approx

        Returns:
            signal: (B, 3, T) — RGB time series
        """
        B, T, C, H, W = frames.shape
        # Forehead ROI: top 20% vertically, middle 40% horizontally
        h_start, h_end = 0, max(1, int(H * 0.20))
        w_start, w_end = int(W * 0.30), int(W * 0.70)

        roi = frames[:, :, :, h_start:h_end, w_start:w_end]  # (B, T, C, roi_H, roi_W)
        # Mean over spatial dimensions → (B, T, C)
        signal = roi.mean(dim=[-1, -2])                        # (B, T, 3)

        # Temporal difference to capture pulse (derivative)
        diff = signal[:, 1:, :] - signal[:, :-1, :]           # (B, T-1, 3)

        # Pad to restore T length
        pad = torch.zeros(B, 1, 3, device=frames.device)
        diff = torch.cat([pad, diff], dim=1)                   # (B, T, 3)

        return diff.permute(0, 2, 1)                           # (B, 3, T)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)
        Returns:
            (B, out_features)
        """
        signal = self._extract_rppg_signal(frames)             # (B, 3, T)

        cnn_out = self.signal_cnn(signal)                      # (B, 64, T)
        cnn_out = cnn_out.permute(0, 2, 1)                     # (B, T, 64)

        gru_out, _ = self.gru(cnn_out)                         # (B, T, out_features)
        return gru_out[:, -1, :]                               # (B, out_features)


# ── Fusion head ────────────────────────────────────────────────────────────

class VideoFusionHead(nn.Module):
    """MLP fusion of all 4 branch outputs."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Main DeepGuard-Video model ────────────────────────────────────────────

class DeepGuardVideoModel(nn.Module):
    """
    Full 4-branch deepfake detection model for videos.

    Input:  (B, T, 3, 224, 224) — sequence of T face frame tensors
    Output: {
        'logits':        (B, 2)
        'probs':         (B, 2)   — [real_prob, fake_prob]
        'branch_scores': dict
        'frame_scores':  (B, T)   — per-frame fake probability
    }
    """

    SPATIAL_DIM    = 512
    FREQ_DIM       = 512
    TEMPORAL_DIM   = 512
    BIOLOGICAL_DIM = 256

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.spatial_branch    = VideoSpatialBranch(out_features=self.SPATIAL_DIM, pretrained=pretrained)
        self.frequency_branch  = VideoFrequencyBranch(out_features=self.FREQ_DIM)
        self.temporal_branch   = TemporalBranch(input_size=self.SPATIAL_DIM, out_features=self.TEMPORAL_DIM)
        self.biological_branch = BiologicalBranch(out_features=self.BIOLOGICAL_DIM)

        total_dim = self.SPATIAL_DIM + self.FREQ_DIM + self.TEMPORAL_DIM + self.BIOLOGICAL_DIM
        self.fusion_head = VideoFusionHead(input_dim=total_dim, num_classes=2)

        # Per-branch auxiliary heads
        self.aux_spatial    = nn.Linear(self.SPATIAL_DIM, 2)
        self.aux_frequency  = nn.Linear(self.FREQ_DIM, 2)
        self.aux_temporal   = nn.Linear(self.TEMPORAL_DIM, 2)
        self.aux_biological = nn.Linear(self.BIOLOGICAL_DIM, 2)

        # Per-frame classifier for frame-level scores
        self.frame_classifier = nn.Linear(self.SPATIAL_DIM, 2)

    def forward(
        self,
        frames: torch.Tensor,           # (B, T, 3, 224, 224)
        return_branch_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, 224, 224) batch of frame sequences
        """
        # ── Extract features ──
        frame_feats, spatial_feat    = self.spatial_branch(frames)    # (B,T,512), (B,512)
        freq_feat                    = self.frequency_branch(frames)   # (B, 512)
        temporal_feat                = self.temporal_branch(frame_feats)  # (B, 512)
        biological_feat              = self.biological_branch(frames)  # (B, 256)

        # ── Fuse branches ──
        combined = torch.cat([spatial_feat, freq_feat, temporal_feat, biological_feat], dim=1)
        logits   = self.fusion_head(combined)
        probs    = F.softmax(logits, dim=1)

        # ── Per-frame scores ──
        frame_logits = self.frame_classifier(frame_feats)   # (B, T, 2)
        frame_probs  = F.softmax(frame_logits, dim=2)       # (B, T, 2)
        frame_scores = frame_probs[:, :, 1]                 # (B, T) — fake prob per frame

        out: Dict = {
            "logits":       logits,
            "probs":        probs,
            "frame_scores": frame_scores,
            "aux_spatial":    self.aux_spatial(spatial_feat),
            "aux_frequency":  self.aux_frequency(freq_feat),
            "aux_temporal":   self.aux_temporal(temporal_feat),
            "aux_biological": self.aux_biological(biological_feat),
        }

        if return_branch_scores:
            out["branch_scores"] = {
                "spatial":    float(F.softmax(out["aux_spatial"],    dim=1)[0, 1]),
                "frequency":  float(F.softmax(out["aux_frequency"],  dim=1)[0, 1]),
                "temporal":   float(F.softmax(out["aux_temporal"],   dim=1)[0, 1]),
                "biological": float(F.softmax(out["aux_biological"], dim=1)[0, 1]),
            }

        return out

    def predict(self, frames: torch.Tensor) -> Dict:
        """Single-pass inference."""
        self.eval()
        with torch.no_grad():
            out = self.forward(frames, return_branch_scores=True)

        probs       = out["probs"]
        fake_prob   = float(probs[0, 1])
        frame_scores = out["frame_scores"][0].tolist()

        return {
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "verdict":          "fake" if fake_prob >= 0.5 else "real",
            "confidence":       max(fake_prob, 1.0 - fake_prob),
            "branch_scores":    out.get("branch_scores", {}),
            "frame_scores":     frame_scores,
        }


def build_video_model(pretrained: bool = True, device: str = "cpu") -> DeepGuardVideoModel:
    model = DeepGuardVideoModel(pretrained=pretrained)
    return model.to(device)


def load_video_model(checkpoint_path: str, device: str = "cpu") -> DeepGuardVideoModel:
    model = DeepGuardVideoModel(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()
