"""
Face Detector
=============
MTCNN-based face detection pipeline.
Detects faces in an image, returns aligned and cropped face tensors.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image

# facenet-pytorch MTCNN
from facenet_pytorch import MTCNN


@dataclass
class DetectedFace:
    """A single detected face with its metadata."""
    crop: Image.Image            # Cropped face (PIL Image, 224×224)
    box: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in original image coords
    probability: float           # MTCNN detection confidence
    face_index: int              # Index if multiple faces in same image


class FaceDetector:
    """
    Wraps MTCNN for production face detection.

    Usage:
        detector = FaceDetector(device="cpu")
        faces = detector.detect(image)
        # faces: List[DetectedFace]
    """

    TARGET_SIZE = (224, 224)

    def __init__(
        self,
        device: str = "cpu",
        min_face_size: int = 40,
        detection_threshold: float = 0.9,
        margin: int = 20,
        keep_all: bool = True,
    ):
        self.device = device
        self.detection_threshold = detection_threshold

        self.mtcnn = MTCNN(
            image_size=self.TARGET_SIZE[0],
            margin=margin,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.9],     # P, R, O-Net thresholds
            factor=0.709,
            post_process=False,              # Return raw pixels, not normalized
            device=device,
            keep_all=keep_all,               # Detect all faces, not just largest
            select_largest=False,
        )

    def detect(self, image: Image.Image) -> List[DetectedFace]:
        """
        Detect all faces in image.

        Args:
            image: PIL Image (RGB)

        Returns:
            List of DetectedFace (empty if no face found)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # MTCNN returns (crops, probs, boxes) or (crop, prob, box) for single
        try:
            crops, probs, boxes = self.mtcnn.detect(image, landmarks=False)
            face_crops = self.mtcnn(image)   # Returns aligned crops
        except Exception:
            return []

        if crops is None or probs is None:
            return []

        # Ensure lists for consistent handling
        if not isinstance(face_crops, list):
            face_crops = [face_crops]
        if not isinstance(probs, (list, np.ndarray)):
            probs = [probs]
        if boxes is not None and len(boxes.shape) == 1:
            boxes = [boxes]

        faces: List[DetectedFace] = []
        for i, (crop, prob, box) in enumerate(zip(face_crops, probs, boxes)):
            if prob is None or prob < self.detection_threshold:
                continue
            if crop is None:
                continue

            # crop is a torch tensor (C, H, W) uint8 — convert to PIL
            if isinstance(crop, torch.Tensor):
                crop_pil = Image.fromarray(
                    crop.permute(1, 2, 0).byte().numpy()
                ).resize(self.TARGET_SIZE, Image.LANCZOS)
            else:
                crop_pil = Image.fromarray(crop).resize(self.TARGET_SIZE, Image.LANCZOS)

            x1, y1, x2, y2 = [int(v) for v in box]
            faces.append(DetectedFace(
                crop=crop_pil,
                box=(x1, y1, x2, y2),
                probability=float(prob),
                face_index=i,
            ))

        return faces

    def detect_from_path(self, image_path: str) -> List[DetectedFace]:
        image = Image.open(image_path).convert("RGB")
        return self.detect(image)


# ── Singleton instance (loaded once at startup) ────────────────────────────

_detector: Optional[FaceDetector] = None


def get_face_detector(device: str = "cpu") -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector(device=device)
    return _detector
