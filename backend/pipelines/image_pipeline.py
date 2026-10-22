"""
Image Inference Pipeline
========================
Orchestrates the full detection flow for a single image:

  1. Load & validate image
  2. Detect faces (MTCNN)
  3. Run DeepGuard-Image model on each face
  4. Aggregate results (if multiple faces)
  5. Generate GradCAM heatmap for most suspicious face
  6. Return structured result

This runs synchronously (fast enough for <2s on CPU for typical images).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from loguru import logger
from PIL import Image

from models.registry import get_image_model, get_gradcam
from preprocessing.face_detector import DetectedFace
from preprocessing.image_preprocessor import get_image_preprocessor
from core.config import settings


# ── Result types ───────────────────────────────────────────────────────────

@dataclass
class FaceResult:
    face_index: int
    verdict: str                        # "real" | "fake"
    fake_probability: float
    real_probability: float
    confidence: float
    branch_scores: dict
    heatmap_path: Optional[str]
    box: tuple                          # bounding box in original image


@dataclass
class ImageDetectionResult:
    verdict: str                        # Overall verdict (most suspicious face wins)
    confidence: float
    fake_probability: float
    real_probability: float
    faces_detected: int
    face_results: List[FaceResult]
    heatmap_path: Optional[str]         # Heatmap for the most suspicious face
    heatmap_url: Optional[str]
    branch_scores: dict                 # Averaged across faces
    error: Optional[str] = None


# ── Pipeline ───────────────────────────────────────────────────────────────

class ImageInferencePipeline:
    """
    Stateless inference pipeline. Dependencies (model, preprocessor, gradcam)
    are loaded lazily from the registry singleton on first use.
    """

    def __init__(self):
        self.device = settings.DEVICE

    def run(
        self,
        image_source: Union[str, Path, bytes],
        generate_heatmap: bool = True,
    ) -> ImageDetectionResult:
        """
        Run the full detection pipeline on an image.

        Args:
            image_source: Path to image file, or raw bytes
            generate_heatmap: Whether to generate GradCAM overlay

        Returns:
            ImageDetectionResult with verdict, confidence, heatmaps, branch scores
        """
        try:
            return self._run_pipeline(image_source, generate_heatmap)
        except Exception as e:
            logger.exception(f"Image inference pipeline error: {e}")
            return ImageDetectionResult(
                verdict="unknown",
                confidence=0.0,
                fake_probability=0.0,
                real_probability=0.0,
                faces_detected=0,
                face_results=[],
                heatmap_path=None,
                heatmap_url=None,
                branch_scores={},
                error=str(e),
            )

    def _run_pipeline(self, image_source, generate_heatmap: bool) -> ImageDetectionResult:
        model       = get_image_model()
        preprocessor = get_image_preprocessor(self.device)

        # ── 1. Preprocess ──────────────────────────────────────────────────
        tensors, faces, original_image = preprocessor.preprocess(image_source)
        logger.info(f"Image preprocessed: {len(faces)} face(s) detected")

        # ── 2. Run model on each face ──────────────────────────────────────
        face_results: List[FaceResult] = []

        for i, (tensor, face) in enumerate(zip(tensors, faces if faces else [None] * len(tensors))):
            with torch.no_grad():
                out = model(tensor, return_branch_scores=True)

            probs       = out["probs"][0]
            fake_prob   = float(probs[1])
            real_prob   = float(probs[0])
            verdict     = "fake" if fake_prob >= settings.IMAGE_CONFIDENCE_THRESHOLD else "real"
            confidence  = max(fake_prob, real_prob)
            branch_scores = out.get("branch_scores", {})

            # ── 3. GradCAM for this face ───────────────────────────────────
            heatmap_path = None
            if generate_heatmap:
                try:
                    gradcam = get_gradcam()
                    face_img = face.crop if face else original_image
                    _, heatmap_path = gradcam.generate(tensor, face_img, target_class=1)
                    logger.debug(f"Heatmap saved: {heatmap_path}")
                except Exception as e:
                    logger.warning(f"GradCAM failed for face {i}: {e}")

            face_results.append(FaceResult(
                face_index=i,
                verdict=verdict,
                fake_probability=fake_prob,
                real_probability=real_prob,
                confidence=confidence,
                branch_scores=branch_scores,
                heatmap_path=heatmap_path,
                box=face.box if face else (0, 0, 0, 0),
            ))

        # ── 4. Aggregate across faces ──────────────────────────────────────
        if not face_results:
            return ImageDetectionResult(
                verdict="unknown",
                confidence=0.0,
                fake_probability=0.0,
                real_probability=0.0,
                faces_detected=0,
                face_results=[],
                heatmap_path=None,
                heatmap_url=None,
                branch_scores={},
                error="No faces detected in image",
            )

        # Use most suspicious face as the overall verdict
        # (a single fake face in the image = image is fake)
        most_suspicious = max(face_results, key=lambda r: r.fake_probability)

        # Average branch scores across all faces
        all_branch_keys = set()
        for r in face_results:
            all_branch_keys.update(r.branch_scores.keys())
        avg_branch_scores = {}
        for key in all_branch_keys:
            vals = [r.branch_scores[key] for r in face_results if key in r.branch_scores]
            avg_branch_scores[key] = sum(vals) / len(vals) if vals else 0.0

        # Build heatmap URL from path
        heatmap_url = None
        if most_suspicious.heatmap_path:
            fname = Path(most_suspicious.heatmap_path).name
            heatmap_url = f"/heatmaps/{fname}"

        return ImageDetectionResult(
            verdict=most_suspicious.verdict,
            confidence=most_suspicious.confidence,
            fake_probability=most_suspicious.fake_probability,
            real_probability=most_suspicious.real_probability,
            faces_detected=len(faces) if faces else 1,
            face_results=face_results,
            heatmap_path=most_suspicious.heatmap_path,
            heatmap_url=heatmap_url,
            branch_scores=avg_branch_scores,
        )


# ── Singleton ──────────────────────────────────────────────────────────────

_pipeline: Optional[ImageInferencePipeline] = None


def get_image_pipeline() -> ImageInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ImageInferencePipeline()
    return _pipeline
