"""
DeepGuard-Audio Model
======================
Three-branch architecture for detecting synthetic/cloned voice:

  Branch 1 — Raw Waveform: RawNet2-style 1D CNN on raw audio samples
  Branch 2 — Spectral:     2D CNN on Mel-spectrogram + MFCC features
  Branch 3 — Prosody:      Pitch, energy, speaking-rate features through MLP

All three branch outputs fused through an MLP head.

Reference: RawNet2 — "End-to-End Anti-Spoofing with RawNet2"
           (Tak et al., ICASSP 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ── Sinc filter for RawNet2 ────────────────────────────────────────────────

class SincConv(nn.Module):
    """
    Sinc-based convolution layer (from SincNet / RawNet2).
    Learns band-pass filters directly on the raw waveform.
    More interpretable than standard Conv1d — each filter has a specific
    frequency band defined by two learnable parameters: low_freq, band_width.
    """

    def __init__(
        self,
        out_channels: int = 128,
        kernel_size: int = 1024,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # must be odd
        self.sample_rate  = sample_rate
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz

        # Initialize filterbank with mel-scale spacing
        low_freq_mel  = 0
        high_freq_mel = 2595 * torch.log10(torch.tensor(1 + (sample_rate / 2) / 700.0))
        mel_points    = torch.linspace(low_freq_mel, high_freq_mel, out_channels + 2)
        hz_points     = 700 * (10 ** (mel_points / 2595) - 1)

        self.low_hz_   = nn.Parameter(hz_points[:-2].unsqueeze(1))
        self.band_hz_  = nn.Parameter((hz_points[1:-1] - hz_points[:-2]).unsqueeze(1))

        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size - 1) / 2, steps=int((self.kernel_size - 1) / 2 + 1))
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n_lin / self.kernel_size)
        self.register_buffer("window_", window.unsqueeze(0))

        # Time axis
        n = (self.kernel_size - 1) / 2
        self.register_buffer("n_", 2 * torch.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (B, 1, T) raw audio signal

        Returns:
            (B, out_channels, T')
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low  = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low  = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left  = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        band_pass_left  = band_pass_left * self.window_

        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right  = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        # (out_channels, 1, kernel_size)
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, filters, padding=self.kernel_size // 2)


# ── Raw Waveform Branch (RawNet2-style) ────────────────────────────────────

class RawWaveformBranch(nn.Module):
    """
    RawNet2-inspired architecture operating directly on raw audio samples.
    Uses SincConv as the first layer for interpretable frequency filtering.
    """

    def __init__(self, out_features: int = 512, sample_rate: int = 16000):
        super().__init__()

        # SincConv front-end
        self.sinc_conv = SincConv(out_channels=128, kernel_size=1024, sample_rate=sample_rate)
        self.bn1   = nn.BatchNorm1d(128)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            self._res_block(128, 256, stride=2),
            self._res_block(256, 256, stride=2),
            self._res_block(256, 512, stride=2),
            self._res_block(512, 512, stride=2),
        )

        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=out_features // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

    @staticmethod
    def _res_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, 1, T) raw audio at 16kHz

        Returns:
            (B, out_features)
        """
        x = self.sinc_conv(waveform)                   # (B, 128, T)
        x = F.leaky_relu(self.bn1(x), 0.1)

        x = self.res_blocks(x)                         # (B, 512, T//16)

        x = x.permute(0, 2, 1)                        # (B, T', 512)
        out, _ = self.gru(x)                           # (B, T', out_features)
        return out[:, -1, :]                            # (B, out_features)


# ── Spectral Branch ────────────────────────────────────────────────────────

class SpectralBranch(nn.Module):
    """
    2D CNN operating on stacked Mel-spectrogram + MFCC features.
    Detects unnatural harmonic patterns and TTS-specific artifacts.
    """

    def __init__(self, out_features: int = 512, n_mels: int = 80, n_mfcc: int = 40):
        super().__init__()
        # Input channels: n_mels (mel-spec) + n_mfcc (MFCC) = 120 → treated as multi-channel
        in_channels = 2  # mel-spec and mfcc stacked as 2 "images"

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, mel_spec: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: (B, n_mels, T) Mel-spectrogram
            mfcc:     (B, n_mfcc, T) MFCC features

        Returns:
            (B, out_features)
        """
        # Resize MFCC to match mel-spec height
        mfcc_resized = F.interpolate(mfcc.unsqueeze(1), size=mel_spec.shape[1:], mode="bilinear", align_corners=False).squeeze(1)

        # Stack as 2-channel 2D input
        x = torch.stack([mel_spec, mfcc_resized], dim=1)  # (B, 2, n_mels, T)

        x = self.cnn(x)    # (B, 256, 4, 4)
        return self.fc(x)  # (B, out_features)


# ── Prosody Branch ────────────────────────────────────────────────────────

class ProsodyBranch(nn.Module):
    """
    MLP operating on hand-crafted prosody features:
    - Pitch (F0) mean, std, range, contour smoothness
    - Energy mean, std, dynamic range
    - Speaking rate (syllable rate approximation)
    - Voiced/unvoiced ratio

    Detects robotic rhythm, unnatural intonation, and prosodic flatness
    common in TTS systems and voice clones.
    """

    PROSODY_FEATURES = 16   # number of hand-crafted features

    def __init__(self, out_features: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.PROSODY_FEATURES, 64),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, prosody_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prosody_features: (B, PROSODY_FEATURES)

        Returns:
            (B, out_features)
        """
        return self.mlp(prosody_features)


# ── Fusion head ────────────────────────────────────────────────────────────

class AudioFusionHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Main DeepGuard-Audio model ────────────────────────────────────────────

class DeepGuardAudioModel(nn.Module):
    """
    Full 3-branch deepfake detection model for audio.

    Input:
        waveform:        (B, 1, T)              raw audio at 16kHz
        mel_spec:        (B, n_mels, T_frames)  Mel-spectrogram
        mfcc:            (B, n_mfcc, T_frames)  MFCC features
        prosody_feats:   (B, 16)                hand-crafted prosody vector

    Output:
        {
            'logits':        (B, 2)
            'probs':         (B, 2)  — [real_prob, fake_prob]
            'branch_scores': dict
        }
    """

    RAW_DIM     = 512
    SPECTRAL_DIM = 512
    PROSODY_DIM  = 256

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

        self.raw_branch      = RawWaveformBranch(out_features=self.RAW_DIM, sample_rate=sample_rate)
        self.spectral_branch = SpectralBranch(out_features=self.SPECTRAL_DIM)
        self.prosody_branch  = ProsodyBranch(out_features=self.PROSODY_DIM)

        total_dim = self.RAW_DIM + self.SPECTRAL_DIM + self.PROSODY_DIM
        self.fusion_head = AudioFusionHead(input_dim=total_dim, num_classes=2)

        # Auxiliary heads for deep supervision
        self.aux_raw      = nn.Linear(self.RAW_DIM, 2)
        self.aux_spectral = nn.Linear(self.SPECTRAL_DIM, 2)
        self.aux_prosody  = nn.Linear(self.PROSODY_DIM, 2)

    def forward(
        self,
        waveform: torch.Tensor,
        mel_spec: torch.Tensor,
        mfcc: torch.Tensor,
        prosody_feats: torch.Tensor,
        return_branch_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:

        raw_feat      = self.raw_branch(waveform)                    # (B, 512)
        spectral_feat = self.spectral_branch(mel_spec, mfcc)         # (B, 512)
        prosody_feat  = self.prosody_branch(prosody_feats)           # (B, 256)

        combined = torch.cat([raw_feat, spectral_feat, prosody_feat], dim=1)  # (B, 1280)
        logits   = self.fusion_head(combined)
        probs    = F.softmax(logits, dim=1)

        out: Dict = {
            "logits":        logits,
            "probs":         probs,
            "aux_raw":       self.aux_raw(raw_feat),
            "aux_spectral":  self.aux_spectral(spectral_feat),
            "aux_prosody":   self.aux_prosody(prosody_feat),
        }

        if return_branch_scores:
            out["branch_scores"] = {
                "raw_waveform": float(F.softmax(out["aux_raw"],      dim=1)[0, 1]),
                "spectral":     float(F.softmax(out["aux_spectral"], dim=1)[0, 1]),
                "prosody":      float(F.softmax(out["aux_prosody"],  dim=1)[0, 1]),
            }

        return out

    def predict(
        self,
        waveform: torch.Tensor,
        mel_spec: torch.Tensor,
        mfcc: torch.Tensor,
        prosody_feats: torch.Tensor,
    ) -> Dict:
        self.eval()
        with torch.no_grad():
            out = self.forward(waveform, mel_spec, mfcc, prosody_feats, return_branch_scores=True)

        probs     = out["probs"]
        fake_prob = float(probs[0, 1])

        return {
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "verdict":          "fake" if fake_prob >= 0.5 else "real",
            "confidence":       max(fake_prob, 1.0 - fake_prob),
            "branch_scores":    out.get("branch_scores", {}),
        }


def build_audio_model(device: str = "cpu") -> DeepGuardAudioModel:
    model = DeepGuardAudioModel(sample_rate=16000)
    return model.to(device)


def load_audio_model(checkpoint_path: str, device: str = "cpu") -> DeepGuardAudioModel:
    model = DeepGuardAudioModel()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()
