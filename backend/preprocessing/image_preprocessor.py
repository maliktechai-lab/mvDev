"""
Image Preprocessing Pipeline
=============================
Handles all preprocessing from raw uploaded file → model-ready tensor.

Steps:
  1. Load image (supports bytes, path, PIL)
  2. Validate (format, size, not corrupt)
  3. Run face detection (MTCNN)
  4. Crop and align faces
  5. Apply normalization transforms
  6. Return tensor(s) ready for model input
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from preprocessing.face_detector import FaceDetector, DetectedFace, get_face_detector
from core.config import settings


# ── ImageNet normalization constants ───────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224


# ── Transforms ────────────────────────────────────────────────────────────

def get_inference_transform() -> T.Compose:
    """Transform for inference — no augmentation."""
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform(augment: bool = True) -> T.Compose:
    """Transform for training — includes augmentation to improve robustness."""
    if not augment:
        return get_inference_transform()

    return T.Compose([
        T.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
        T.RandomCrop(INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomGrayscale(p=0.05),
        # Simulate compression artifacts (common in real-world deepfakes)
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Random erasing simulates occlusion
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


# ── Image preprocessing ────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    End-to-end preprocessing: raw image → face crops → normalized tensors.
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    MAX_SIZE_BYTES     = settings.max_image_size_bytes

    def __init__(self, device: str = "cpu"):
        self.device    = device
        self.transform = get_inference_transform()
        self.detector  = get_face_detector(device=device)

    def load_image(self, source: Union[str, Path, bytes, io.BytesIO]) -> Image.Image:
        """Load image from path or bytes."""
        try:
            if isinstance(source, (str, Path)):
                img = Image.open(source)
            elif isinstance(source, (bytes, io.BytesIO)):
                data = source if isinstance(source, io.BytesIO) else io.BytesIO(source)
                img = Image.open(data)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")

            return img.convert("RGB")
        except UnidentifiedImageError as e:
            raise ValueError("File is not a valid image or is corrupt") from e

    def validate_image(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate image before processing.
        Returns (is_valid, error_message).
        """
        # Check file size for paths
        if isinstance(source, (str, Path)):
            size = Path(source).stat().st_size
            if size > self.MAX_SIZE_BYTES:
                return False, f"Image too large: {size / 1e6:.1f} MB (max {settings.MAX_IMAGE_SIZE_MB} MB)"

        # Try loading
        try:
            img = self.load_image(source)
        except ValueError as e:
            return False, str(e)

        # Check minimum dimensions
        w, h = img.size
        if w < 64 or h < 64:
            return False, f"Image too small: {w}×{h} (min 64×64)"

        return True, None

    def preprocess(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
        use_face_detection: bool = True,
    ) -> Tuple[List[torch.Tensor], List[DetectedFace], Image.Image]:
        """
        Full preprocessing pipeline.

        Args:
            source: Image source (path, bytes, BytesIO)
            use_face_detection: If True, crop faces; if False, use whole image

        Returns:
            (tensors, faces, original_image)
            - tensors: List of (3, 224, 224) float tensors ready for model
            - faces:   List of DetectedFace metadata (empty if no faces found)
            - original_image: Raw PIL image (for GradCAM overlay)
        """
        image = self.load_image(source)

        if use_face_detection:
            faces = self.detector.detect(image)

            if not faces:
                # Fallback: use whole image if no face detected
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                return [tensor], [], image

            tensors = []
            for face in faces:
                t = self.transform(face.crop).unsqueeze(0).to(self.device)
                tensors.append(t)

            return tensors, faces, image
        else:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            return [tensor], [], image

    def batch_preprocess(
        self,
        sources: List[Union[str, Path, bytes]],
    ) -> torch.Tensor:
        """
        Preprocess a list of face crops into a single batch tensor.
        Used during training.

        Returns: (N, 3, 224, 224) tensor
        """
        tensors = []
        for src in sources:
            img = self.load_image(src)
            t = self.transform(img)
            tensors.append(t)
        return torch.stack(tensors).to(self.device)


# ── Singleton ──────────────────────────────────────────────────────────────

_preprocessor: Optional[ImagePreprocessor] = None


def get_image_preprocessor(device: str = "cpu") -> ImagePreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor(device=device)
    return _preprocessor
