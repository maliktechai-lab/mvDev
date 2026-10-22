"""
Audio Preprocessing Pipeline
==============================
Handles all preprocessing from raw uploaded audio → model-ready tensors.

Steps:
  1. Load audio (supports MP3, WAV, FLAC, M4A, OGG)
  2. Convert to mono, resample to 16kHz
  3. Segment into overlapping 3-second chunks
  4. For each chunk:
     a. Raw waveform tensor (1, T)
     b. Mel-spectrogram (n_mels, T_frames)
     c. MFCC features (n_mfcc, T_frames)
     d. Prosody features (16-dim vector)
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as AT
from loguru import logger

from core.config import settings


# ── Constants ─────────────────────────────────────────────────────────────

TARGET_SR       = 16000    # 16kHz
CHUNK_DURATION  = settings.AUDIO_CHUNK_DURATION_SEC   # 3 seconds
CHUNK_SAMPLES   = TARGET_SR * CHUNK_DURATION           # 48000 samples
OVERLAP_RATIO   = 0.5                                  # 50% overlap between chunks
N_MELS          = 80
N_MFCC          = 40
N_FFT           = 1024
HOP_LENGTH      = 256
WIN_LENGTH      = 1024


# ── Transforms ─────────────────────────────────────────────────────────────

def get_mel_transform(sample_rate: int = TARGET_SR) -> AT.MelSpectrogram:
    return AT.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )


def get_mfcc_transform(sample_rate: int = TARGET_SR) -> AT.MFCC:
    return AT.MFCC(
        sample_rate=sample_rate,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "win_length": WIN_LENGTH,
        },
    )


# ── Audio chunk dataclass ─────────────────────────────────────────────────

@dataclass
class AudioChunk:
    chunk_index: int
    start_sec:   float
    end_sec:     float
    waveform:    torch.Tensor    # (1, T)
    mel_spec:    torch.Tensor    # (n_mels, T_frames)
    mfcc:        torch.Tensor    # (n_mfcc, T_frames)
    prosody:     torch.Tensor    # (16,)


# ── Prosody feature extraction ────────────────────────────────────────────

def extract_prosody_features(waveform_np: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract 16 prosody features from a waveform chunk.

    Features:
      0-3:  F0 (pitch) — mean, std, min, max
      4-7:  Energy (RMS) — mean, std, min, max
      8-11: Spectral centroid — mean, std, min, max
      12:   Voiced frame ratio
      13:   Zero crossing rate mean
      14:   Spectral rolloff mean
      15:   Tempo (beats per minute / 200 for normalization)
    """
    features = np.zeros(16, dtype=np.float32)

    try:
        # F0 (pitch) using librosa YIN
        f0, voiced_flag, _ = librosa.pyin(
            waveform_np,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            fill_na=0.0,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
        features[0] = float(np.mean(f0_voiced))
        features[1] = float(np.std(f0_voiced))
        features[2] = float(np.min(f0_voiced))
        features[3] = float(np.max(f0_voiced))

        # Energy (RMS)
        rms = librosa.feature.rms(y=waveform_np, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        features[4] = float(np.mean(rms))
        features[5] = float(np.std(rms))
        features[6] = float(np.min(rms))
        features[7] = float(np.max(rms))

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=waveform_np, sr=sr, hop_length=HOP_LENGTH)[0]
        features[8]  = float(np.mean(centroid))
        features[9]  = float(np.std(centroid))
        features[10] = float(np.min(centroid))
        features[11] = float(np.max(centroid))

        # Voiced ratio
        features[12] = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.5

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(waveform_np, hop_length=HOP_LENGTH)[0]
        features[13] = float(np.mean(zcr))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=waveform_np, sr=sr, hop_length=HOP_LENGTH)[0]
        features[14] = float(np.mean(rolloff))

        # Tempo (normalized)
        tempo, _ = librosa.beat.beat_track(y=waveform_np, sr=sr)
        features[15] = float(tempo) / 200.0 if tempo else 0.5

    except Exception as e:
        logger.warning(f"Prosody extraction partial failure: {e}")

    # Normalize to [0, 1] range (rough normalization)
    features = np.clip(features / (np.abs(features).max() + 1e-8), -3.0, 3.0)
    return features


# ── Main preprocessor ─────────────────────────────────────────────────────

class AudioPreprocessor:
    """
    End-to-end audio preprocessing pipeline.
    Loads audio, resamples, segments into chunks, extracts all features.
    """

    SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    MAX_SIZE_BYTES     = settings.max_audio_size_bytes

    def __init__(self, device: str = "cpu"):
        self.device          = device
        self.mel_transform   = get_mel_transform().to(device)
        self.mfcc_transform  = get_mfcc_transform().to(device)
        self.amplitude_to_db = AT.AmplitudeToDB(stype="power", top_db=80).to(device)

    def load_audio(self, source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[torch.Tensor, int]:
        """
        Load audio from path or bytes.

        Returns:
            (waveform, sample_rate) where waveform is (channels, T)
        """
        try:
            if isinstance(source, (bytes, io.BytesIO)):
                data = source if isinstance(source, io.BytesIO) else io.BytesIO(source)
                waveform, sr = torchaudio.load(data)
            else:
                waveform, sr = torchaudio.load(str(source))
            return waveform, sr
        except Exception as e:
            raise ValueError(f"Cannot load audio file: {e}") from e

    def preprocess(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
    ) -> Tuple[List[AudioChunk], float]:
        """
        Full preprocessing pipeline.

        Returns:
            (chunks, duration_sec)
        """
        # ── Load ───────────────────────────────────────────────────────────
        waveform, sr = self.load_audio(source)

        # ── To mono ────────────────────────────────────────────────────────
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # (1, T)

        # ── Resample to 16kHz ──────────────────────────────────────────────
        if sr != TARGET_SR:
            resampler = AT.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform  = resampler(waveform)

        duration_sec = waveform.shape[1] / TARGET_SR
        logger.info(f"Audio loaded: {duration_sec:.1f}s, {waveform.shape[1]} samples")

        # ── Segment into chunks ────────────────────────────────────────────
        waveform = waveform.to(self.device)
        chunks   = self._segment(waveform, duration_sec)

        logger.info(f"Audio segmented: {len(chunks)} chunks ({CHUNK_DURATION}s each)")
        return chunks, duration_sec

    def _segment(self, waveform: torch.Tensor, duration_sec: float) -> List[AudioChunk]:
        """Segment waveform into overlapping CHUNK_DURATION-second windows."""
        total_samples = waveform.shape[1]
        stride        = int(CHUNK_SAMPLES * (1 - OVERLAP_RATIO))
        chunks: List[AudioChunk] = []

        start = 0
        idx   = 0
        while start < total_samples:
            end   = start + CHUNK_SAMPLES
            chunk = waveform[:, start:end]

            # Pad last chunk if shorter than CHUNK_SAMPLES
            if chunk.shape[1] < CHUNK_SAMPLES:
                pad = torch.zeros(1, CHUNK_SAMPLES - chunk.shape[1], device=self.device)
                chunk = torch.cat([chunk, pad], dim=1)

            # Extract features
            mel_spec      = self._get_mel_spec(chunk)       # (n_mels, T_frames)
            mfcc          = self._get_mfcc(chunk)            # (n_mfcc, T_frames)
            prosody_np    = extract_prosody_features(
                chunk[0].cpu().numpy(), sr=TARGET_SR
            )
            prosody_feat  = torch.tensor(prosody_np, dtype=torch.float32).to(self.device)

            start_sec = start / TARGET_SR
            end_sec   = min(end / TARGET_SR, duration_sec)

            chunks.append(AudioChunk(
                chunk_index=idx,
                start_sec=start_sec,
                end_sec=end_sec,
                waveform=chunk,
                mel_spec=mel_spec,
                mfcc=mfcc,
                prosody=prosody_feat,
            ))

            idx   += 1
            start += stride

            # Safety cap
            if idx >= 100:
                logger.warning("Audio too long — capping at 100 chunks")
                break

        return chunks

    def _get_mel_spec(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log Mel-spectrogram."""
        with torch.no_grad():
            mel = self.mel_transform(waveform)              # (1, n_mels, T)
            log_mel = self.amplitude_to_db(mel)[0]          # (n_mels, T)
        return log_mel

    def _get_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute MFCC features."""
        with torch.no_grad():
            mfcc = self.mfcc_transform(waveform)[0]         # (n_mfcc, T)
        return mfcc


# ── Singleton ──────────────────────────────────────────────────────────────

_audio_preprocessor: Optional[AudioPreprocessor] = None


def get_audio_preprocessor(device: str = "cpu") -> AudioPreprocessor:
    global _audio_preprocessor
    if _audio_preprocessor is None:
        _audio_preprocessor = AudioPreprocessor(device=device)
    return _audio_preprocessor
