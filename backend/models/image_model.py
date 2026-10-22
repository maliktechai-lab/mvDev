"""
DeepGuard-Image Model
=====================
Three-branch architecture for detecting fake images:

  Branch 1 — Spatial:    EfficientNet-B4 backbone operating on pixel domain
  Branch 2 — Frequency:  CNN operating on FFT + DCT feature maps
  Branch 3 — Attention:  Cross-attention between spatial and frequency features

All three branch outputs are concatenated and fed into a fusion MLP head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import Dict, Tuple


# ── Frequency feature extractor ────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Converts input image to frequency domain (FFT + DCT),
    then learns features from frequency artifacts left by GAN/diffusion models.
    """

    def __init__(self, in_channels: int = 3, out_features: int = 512):
        super().__init__()

        # Separate streams for FFT magnitude and DCT
        self.fft_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dct_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Merge FFT + DCT streams
        self.merge = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    @staticmethod
    def _fft_features(x: torch.Tensor) -> torch.Tensor:
        """Compute log-magnitude spectrum of FFT."""
        # x: (B, C, H, W)
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        log_magnitude = torch.log1p(magnitude)
        return log_magnitude

    @staticmethod
    def _dct_features(x: torch.Tensor) -> torch.Tensor:
        """Approximate DCT via DFT of mirrored signal (type-2 DCT)."""
        B, C, H, W = x.shape
        # Mirror signal for type-2 DCT approximation
        x_mirror = torch.cat([x, x.flip(-1)], dim=-1)   # mirror W
        dct_approx = torch.fft.rfft(x_mirror, norm="ortho")[..., :W]
        return torch.abs(dct_approx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_feat = self._fft_features(x)       # (B, C, H, W)
        dct_feat = self._dct_features(x)       # (B, C, H, W)

        f = self.fft_conv(fft_feat)            # (B, 64, H/2, W/2)
        d = self.dct_conv(dct_feat)            # (B, 64, H/2, W/2)

        merged = self.merge(torch.cat([f, d], dim=1))   # (B, 256, 4, 4)
        return self.fc(merged)                           # (B, out_features)


# ── Spatial branch ─────────────────────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    EfficientNet-B4 backbone for pixel-domain artifact detection.
    Pre-trained on ImageNet, final classifier replaced with feature extractor.
    """

    def __init__(self, out_features: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,       # Remove final classifier → returns feature vector
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features   # 1792 for B4

        self.project = nn.Sequential(
            nn.Linear(feat_dim, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)        # (B, 1792)
        return self.project(feat)      # (B, out_features)


# ── Attention branch ───────────────────────────────────────────────────────

class AttentionBranch(nn.Module):
    """
    Cross-attention module that fuses spatial and frequency features.
    Detects structural / semantic inconsistencies by attending across domains.
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8, out_features: int = 512):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(feature_dim)
        self.norm_freq    = nn.LayerNorm(feature_dim)

        # Spatial attends to frequency (cross-attention)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Self-attention on spatial features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, out_features),
        )
        self.norm_out = nn.LayerNorm(out_features)

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        # Reshape to sequence: (B, 1, D)
        s = spatial_feat.unsqueeze(1)     # query
        f = freq_feat.unsqueeze(1)        # key/value

        # Cross-attention: spatial queries attend to frequency features
        attn_out, _ = self.cross_attn(
            query=self.norm_spatial(s),
            key=self.norm_freq(f),
            value=self.norm_freq(f),
        )
        s = s + attn_out                  # residual

        # Self-attention on enriched spatial features
        self_out, _ = self.self_attn(s, s, s)
        s = s + self_out

        # FFN + norm
        out = self.ffn(s.squeeze(1))      # (B, out_features)
        return self.norm_out(out)


# ── Fusion head ────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    MLP fusion head that combines all 3 branch features into a final prediction.
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Main DeepGuard-Image model ─────────────────────────────────────────────

class DeepGuardImageModel(nn.Module):
    """
    Full 3-branch deepfake detection model for images.

    Input:  (B, 3, 224, 224) normalized face crop
    Output: {
        'logits':        (B, 2)   — raw logits
        'probs':         (B, 2)   — softmax probabilities [real, fake]
        'branch_scores': dict     — per-branch fake probability
    }
    """

    BRANCH_DIM = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.spatial_branch   = SpatialBranch(out_features=self.BRANCH_DIM, pretrained=pretrained)
        self.frequency_branch = FrequencyBranch(in_channels=3, out_features=self.BRANCH_DIM)
        self.attention_branch = AttentionBranch(
            feature_dim=self.BRANCH_DIM,
            num_heads=8,
            out_features=self.BRANCH_DIM,
        )

        # 3 branches × BRANCH_DIM
        self.fusion_head = FusionHead(input_dim=self.BRANCH_DIM * 3, num_classes=2)

        # Per-branch auxiliary heads (used during training for deep supervision)
        self.aux_spatial   = nn.Linear(self.BRANCH_DIM, 2)
        self.aux_frequency = nn.Linear(self.BRANCH_DIM, 2)
        self.aux_attention = nn.Linear(self.BRANCH_DIM, 2)

    def forward(
        self,
        x: torch.Tensor,
        return_branch_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # ── Extract features from each branch ──
        spatial_feat   = self.spatial_branch(x)     # (B, 512)
        frequency_feat = self.frequency_branch(x)   # (B, 512)
        attention_feat = self.attention_branch(spatial_feat, frequency_feat)  # (B, 512)

        # ── Fuse all branches ──
        combined = torch.cat([spatial_feat, frequency_feat, attention_feat], dim=1)  # (B, 1536)
        logits   = self.fusion_head(combined)                                         # (B, 2)
        probs    = F.softmax(logits, dim=1)                                           # (B, 2)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs":  probs,
        }

        # Auxiliary logits (training only)
        out["aux_spatial"]   = self.aux_spatial(spatial_feat)
        out["aux_frequency"] = self.aux_frequency(frequency_feat)
        out["aux_attention"] = self.aux_attention(attention_feat)

        if return_branch_scores:
            out["branch_scores"] = {
                "spatial":   float(F.softmax(out["aux_spatial"],   dim=1)[0, 1]),
                "frequency": float(F.softmax(out["aux_frequency"], dim=1)[0, 1]),
                "attention": float(F.softmax(out["aux_attention"], dim=1)[0, 1]),
            }

        return out

    def predict(self, x: torch.Tensor) -> Dict:
        """
        Single-pass inference. Returns verdict + confidence + per-branch scores.
        x: (B, 3, 224, 224)
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_branch_scores=True)

        probs = out["probs"]  # (B, 2)
        fake_prob = probs[:, 1]

        results = []
        for i in range(probs.shape[0]):
            score = float(fake_prob[i])
            results.append({
                "fake_probability": score,
                "real_probability": 1.0 - score,
                "verdict": "fake" if score >= 0.5 else "real",
                "confidence": max(score, 1.0 - score),
                "branch_scores": {
                    k: v for k, v in out["branch_scores"].items()
                } if "branch_scores" in out else {},
            })

        return results[0] if len(results) == 1 else results


def build_image_model(pretrained: bool = True, device: str = "cpu") -> DeepGuardImageModel:
    """Factory function — creates and moves model to device."""
    model = DeepGuardImageModel(pretrained=pretrained)
    model = model.to(device)
    return model


def load_image_model(checkpoint_path: str, device: str = "cpu") -> DeepGuardImageModel:
    """Load a trained model from checkpoint."""
    model = DeepGuardImageModel(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model
