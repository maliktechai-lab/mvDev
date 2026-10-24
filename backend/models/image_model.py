"""
DeepGuard-Image Model
=====================
Three-branch architecture for detecting fake images.
ALL components built from scratch — no pretrained weights, no external backbones.

  Branch 1 — Spatial:    Custom deep CNN (depthwise-separable + SE blocks + residuals)
  Branch 2 — Frequency:  CNN on FFT magnitude + DCT approximation
  Branch 3 — Attention:  Cross-attention between spatial and frequency features

All three branch outputs are concatenated and fed into a fusion MLP head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ── Building blocks ────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    """Conv2d → BatchNorm2d → activation (default: SiLU)."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """Channel-wise SE block — recalibrates feature maps by channel importance."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise-separable conv + SE + residual.
    Used as the primary building block in the spatial branch.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise  = ConvBnAct(in_ch, in_ch,  3, stride, 1, groups=in_ch)
        self.pointwise  = ConvBnAct(in_ch, out_ch, 1, 1,      0, act=True)
        self.se         = SqueezeExcitation(out_ch, reduction=4)

        self.shortcut = (
            ConvBnAct(in_ch, out_ch, 1, stride, 0, act=False)
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.se(out)
        return F.silu(out + self.shortcut(x))


class ResBlock(nn.Module):
    """Standard residual block with two 3×3 convs + optional stride."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBnAct(in_ch,  out_ch, 3, stride, 1)
        self.conv2 = ConvBnAct(out_ch, out_ch, 3, 1,      1, act=False)
        self.shortcut = (
            ConvBnAct(in_ch, out_ch, 1, stride, 0, act=False)
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv2(self.conv1(x)) + self.shortcut(x))


# ── Spatial branch ─────────────────────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    Custom deep CNN built entirely from scratch.

    Architecture:
        stem       : 3→32, 3×3, stride 2  (112×112)
        stage 1    : 32→64,  2× DSepBlock, stride 2  (56×56)
        stage 2    : 64→128, 2× DSepBlock, stride 2  (28×28)
        stage 3    : 128→256, 3× DSepBlock, stride 2  (14×14)
        stage 4    : 256→512, 3× ResBlock,  stride 2  (7×7)
        stage 5    : 512→512, 2× ResBlock,  stride 1  (7×7)
        head       : AdaptiveAvgPool → Dropout → Linear → output

    Total ~5.2M parameters — efficient enough for CPU inference,
    large enough to learn complex forgery patterns.
    """

    def __init__(self, out_features: int = 512):
        super().__init__()

        # Stem: rapid spatial downsampling
        self.stem = nn.Sequential(
            ConvBnAct(3,  32, 3, stride=2, padding=1),   # 112×112
            ConvBnAct(32, 32, 3, stride=1, padding=1),
        )

        # Stage 1 — 56×56
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 64, stride=1),
        )

        # Stage 2 — 28×28
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 128, stride=1),
        )

        # Stage 3 — 14×14
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(128, 256, stride=2),
            DepthwiseSeparableBlock(256, 256, stride=1),
            DepthwiseSeparableBlock(256, 256, stride=1),
        )

        # Stage 4 — 7×7 (standard residuals for deeper feature extraction)
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
            ResBlock(512, 512, stride=1),
        )

        # Stage 5 — 7×7 (additional refinement)
        self.stage5 = nn.Sequential(
            ResBlock(512, 512, stride=1),
            ResBlock(512, 512, stride=1),
        )

        # Global pooling + projection head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, out_features),
            nn.SiLU(inplace=True),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # (B, 32,  112, 112)
        x = self.stage1(x)   # (B, 64,   56,  56)
        x = self.stage2(x)   # (B, 128,  28,  28)
        x = self.stage3(x)   # (B, 256,  14,  14)
        x = self.stage4(x)   # (B, 512,   7,   7)
        x = self.stage5(x)   # (B, 512,   7,   7)
        x = self.pool(x)     # (B, 512,   1,   1)
        return self.head(x)  # (B, out_features)


# ── Frequency branch ───────────────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Converts input image to frequency domain (FFT + DCT),
    then learns features from frequency artifacts left by GAN/diffusion models.
    All convolutions built from scratch — no pretrained components.
    """

    def __init__(self, in_channels: int = 3, out_features: int = 512):
        super().__init__()

        # FFT stream
        self.fft_conv = nn.Sequential(
            ConvBnAct(in_channels, 32, 3, 1, 1),
            ConvBnAct(32, 64, 3, stride=2, padding=1),
            SqueezeExcitation(64),
        )

        # DCT stream
        self.dct_conv = nn.Sequential(
            ConvBnAct(in_channels, 32, 3, 1, 1),
            ConvBnAct(32, 64, 3, stride=2, padding=1),
            SqueezeExcitation(64),
        )

        # Merge FFT + DCT → deeper refinement
        self.merge = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_features),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _fft_features(x: torch.Tensor) -> torch.Tensor:
        fft       = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft)
        return torch.log1p(torch.abs(fft_shift))

    @staticmethod
    def _dct_features(x: torch.Tensor) -> torch.Tensor:
        _, _, _, W = x.shape
        x_mirror = torch.cat([x, x.flip(-1)], dim=-1)
        dct_approx = torch.fft.rfft(x_mirror, norm="ortho")[..., :W]
        return torch.abs(dct_approx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_feat = self._fft_features(x)
        dct_feat = self._dct_features(x)

        f = self.fft_conv(fft_feat)
        d = self.dct_conv(dct_feat)

        merged = self.merge(torch.cat([f, d], dim=1))
        return self.head(merged)


# ── Attention branch ───────────────────────────────────────────────────────

class AttentionBranch(nn.Module):
    """
    Cross-attention between spatial and frequency features.
    Detects structural / semantic inconsistencies by attending across domains.
    Built from scratch using nn.MultiheadAttention.
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8, out_features: int = 512):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(feature_dim)
        self.norm_freq    = nn.LayerNorm(feature_dim)

        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.1, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.1, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, out_features),
        )
        self.norm_out = nn.LayerNorm(out_features)

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        s = spatial_feat.unsqueeze(1)
        f = freq_feat.unsqueeze(1)

        attn_out, _ = self.cross_attn(
            query=self.norm_spatial(s),
            key=self.norm_freq(f),
            value=self.norm_freq(f),
        )
        s = s + attn_out

        self_out, _ = self.self_attn(s, s, s)
        s = s + self_out

        out = self.ffn(s.squeeze(1))
        return self.norm_out(out)


# ── Fusion head ────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Main model ─────────────────────────────────────────────────────────────

class DeepGuardImageModel(nn.Module):
    """
    Full 3-branch deepfake detection model — trained entirely from scratch.

    Input:  (B, 3, 224, 224) normalized face crop
    Output: {
        'logits':        (B, 2)
        'probs':         (B, 2)  — [real_prob, fake_prob]
        'branch_scores': dict    — per-branch fake probability (when requested)
    }
    """

    BRANCH_DIM = 512

    def __init__(self):
        super().__init__()
        self.spatial_branch   = SpatialBranch(out_features=self.BRANCH_DIM)
        self.frequency_branch = FrequencyBranch(in_channels=3, out_features=self.BRANCH_DIM)
        self.attention_branch = AttentionBranch(
            feature_dim=self.BRANCH_DIM, num_heads=8, out_features=self.BRANCH_DIM,
        )
        self.fusion_head = FusionHead(input_dim=self.BRANCH_DIM * 3, num_classes=2)

        # Per-branch auxiliary heads for deep supervision during training
        self.aux_spatial   = nn.Linear(self.BRANCH_DIM, 2)
        self.aux_frequency = nn.Linear(self.BRANCH_DIM, 2)
        self.aux_attention = nn.Linear(self.BRANCH_DIM, 2)

    def forward(
        self,
        x: torch.Tensor,
        return_branch_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        spatial_feat   = self.spatial_branch(x)
        frequency_feat = self.frequency_branch(x)
        attention_feat = self.attention_branch(spatial_feat, frequency_feat)

        combined = torch.cat([spatial_feat, frequency_feat, attention_feat], dim=1)
        logits   = self.fusion_head(combined)
        probs    = F.softmax(logits, dim=1)

        out: Dict[str, torch.Tensor] = {
            "logits":        logits,
            "probs":         probs,
            "aux_spatial":   self.aux_spatial(spatial_feat),
            "aux_frequency": self.aux_frequency(frequency_feat),
            "aux_attention": self.aux_attention(attention_feat),
        }

        if return_branch_scores:
            out["branch_scores"] = {
                "spatial":   float(F.softmax(out["aux_spatial"],   dim=1)[0, 1]),
                "frequency": float(F.softmax(out["aux_frequency"], dim=1)[0, 1]),
                "attention": float(F.softmax(out["aux_attention"], dim=1)[0, 1]),
            }

        return out

    def predict(self, x: torch.Tensor) -> Dict:
        self.eval()
        with torch.no_grad():
            out = self.forward(x, return_branch_scores=True)
        probs     = out["probs"]
        fake_prob = float(probs[0, 1])
        return {
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "verdict":          "fake" if fake_prob >= 0.5 else "real",
            "confidence":       max(fake_prob, 1.0 - fake_prob),
            "branch_scores":    out.get("branch_scores", {}),
        }


# ── Factory functions ──────────────────────────────────────────────────────

def build_image_model(pretrained: bool = False, device: str = "cpu") -> DeepGuardImageModel:
    """
    Build DeepGuardImageModel from scratch.
    The `pretrained` argument is kept for API compatibility but is ignored —
    this model has no pretrained weights by design.
    """
    model = DeepGuardImageModel()
    return model.to(device)


def load_image_model(checkpoint_path: str, device: str = "cpu") -> DeepGuardImageModel:
    """Load a trained model from a saved checkpoint."""
    model = DeepGuardImageModel()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()
