"""
Frame Extractor
===============
Extracts frames from video files using ffmpeg (fast, reliable).
Falls back to OpenCV if ffmpeg is not available.

Produces:
  - Extracted frame images as PIL Images or temp files
  - Metadata: fps, duration, total frames, resolution
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger


@dataclass
class VideoMetadata:
    duration_sec: float
    fps: float
    total_frames: int
    width: int
    height: int
    codec: str = ""

    @property
    def resolution(self) -> str:
        return f"{self.width}×{self.height}"


@dataclass
class ExtractedFrame:
    index: int              # Frame index in extracted sequence (0-based)
    original_frame_num: int # Actual frame number in source video
    timestamp_sec: float    # Timestamp in source video
    image: Image.Image      # PIL Image (RGB)


class FrameExtractor:
    """
    Extracts frames from video at a configurable rate.

    Uses ffmpeg for extraction (fast, handles all codecs).
    Falls back to OpenCV if ffmpeg not found.
    """

    def __init__(
        self,
        target_fps: float = 1.0,          # Frames per second to extract
        max_frames: int = 300,            # Hard cap to avoid OOM
        target_size: Tuple[int, int] = (224, 224),
        face_only: bool = True,           # Apply face detector to each frame
    ):
        self.target_fps  = target_fps
        self.max_frames  = max_frames
        self.target_size = target_size
        self.face_only   = face_only
        self._has_ffmpeg = shutil.which("ffmpeg") is not None

        if not self._has_ffmpeg:
            logger.warning("ffmpeg not found — falling back to OpenCV extraction")

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec  = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        return VideoMetadata(
            duration_sec=duration_sec,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            codec=codec.strip(),
        )

    def extract(self, video_path: str) -> Tuple[List[ExtractedFrame], VideoMetadata]:
        """
        Extract frames from video.

        Returns:
            (frames, metadata)
        """
        meta = self.get_metadata(video_path)
        logger.info(
            f"Extracting from {Path(video_path).name}: "
            f"{meta.duration_sec:.1f}s, {meta.fps:.1f}fps, {meta.resolution}"
        )

        if self._has_ffmpeg:
            frames = self._extract_ffmpeg(video_path, meta)
        else:
            frames = self._extract_opencv(video_path, meta)

        # Cap at max_frames
        if len(frames) > self.max_frames:
            # Sample evenly across the video
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames  = [frames[i] for i in indices]

        logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
        return frames, meta

    def _extract_ffmpeg(self, video_path: str, meta: VideoMetadata) -> List[ExtractedFrame]:
        """Use ffmpeg for fast frame extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract frames at target_fps
            pattern = os.path.join(tmpdir, "frame_%06d.jpg")
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps={self.target_fps}",
                "-q:v", "2",                      # High quality JPEG
                "-frames:v", str(self.max_frames),
                pattern,
                "-y", "-loglevel", "error",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"ffmpeg error: {result.stderr}, falling back to OpenCV")
                return self._extract_opencv(video_path, meta)

            # Load extracted frames
            frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))
            frames: List[ExtractedFrame] = []

            frame_interval = meta.fps / self.target_fps if self.target_fps > 0 else 1
            for i, frame_path in enumerate(frame_files):
                try:
                    img = Image.open(frame_path).convert("RGB")
                    if self.target_size:
                        img = img.resize(self.target_size, Image.LANCZOS)
                    timestamp = i * (1.0 / self.target_fps)
                    frames.append(ExtractedFrame(
                        index=i,
                        original_frame_num=int(i * frame_interval),
                        timestamp_sec=timestamp,
                        image=img,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load frame {frame_path}: {e}")

            return frames

    def _extract_opencv(self, video_path: str, meta: VideoMetadata) -> List[ExtractedFrame]:
        """OpenCV fallback frame extraction."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_interval = max(1, int(meta.fps / self.target_fps))
        frames: List[ExtractedFrame] = []
        frame_num = 0
        extracted = 0

        while extracted < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                if self.target_size:
                    img = img.resize(self.target_size, Image.LANCZOS)

                frames.append(ExtractedFrame(
                    index=extracted,
                    original_frame_num=frame_num,
                    timestamp_sec=frame_num / meta.fps,
                    image=img,
                ))
                extracted += 1

            frame_num += 1

        cap.release()
        return frames


# ── Singleton ──────────────────────────────────────────────────────────────

_extractor: Optional[FrameExtractor] = None


def get_frame_extractor(target_fps: float = 1.0) -> FrameExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FrameExtractor(target_fps=target_fps)
    return _extractor
