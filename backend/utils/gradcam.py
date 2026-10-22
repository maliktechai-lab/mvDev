"""
GradCAM Heatmap Generator
==========================
Generates class activation maps overlaid on the original image
to explain which regions the model focused on.

Uses the pytorch-grad-cam library targeting EfficientNet-B4's
last convolutional layer in the spatial branch.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from core.config import settings


class GradCAMGenerator:
    """
    Generates GradCAM heatmaps for the DeepGuard-Image model.

    The target layer is the last conv block of EfficientNet-B4 in the spatial branch.
    This gives the most semantically meaningful activations.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model  = model
        self.device = device

        # Target layer: last block of EfficientNet-B4 backbone
        target_layer = model.spatial_branch.backbone.blocks[-1]
        self.cam = GradCAM(
            model=self._wrapped_model(),
            target_layers=[target_layer],
        )

    def _wrapped_model(self):
        """
        GradCAM needs a model that returns raw logits as a tensor.
        Wrap DeepGuardImageModel to return only main logits.
        """

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x)
                return out["logits"]

        return Wrapper(self.model)

    def generate(
        self,
        input_tensor: torch.Tensor,     # (1, 3, 224, 224)
        original_image: Image.Image,    # Original (unresized) PIL image
        target_class: int = 1,          # 1 = FAKE class
    ) -> Tuple[np.ndarray, str]:
        """
        Generate heatmap and save to disk.

        Args:
            input_tensor:   Preprocessed face tensor (1, 3, 224, 224)
            original_image: Original PIL image for overlay
            target_class:   1 = highlight regions that indicate FAKE

        Returns:
            (heatmap_array, saved_file_path)
        """
        targets = [ClassifierOutputTarget(target_class)]

        # Compute GradCAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]    # (H, W)

        # Resize original image to match input
        orig_resized = np.array(
            original_image.resize((224, 224), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0

        # Overlay heatmap on image
        heatmap_overlay = show_cam_on_image(
            img=orig_resized,
            mask=grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.5,
        )

        # Save to heatmaps directory
        heatmap_path = self._save_heatmap(heatmap_overlay)

        return heatmap_overlay, heatmap_path

    def _save_heatmap(self, heatmap: np.ndarray) -> str:
        """Save heatmap array as PNG and return the file path."""
        heatmap_dir = Path(settings.HEATMAP_DIR)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        filename   = f"{uuid.uuid4().hex}.png"
        filepath   = heatmap_dir / filename

        img = Image.fromarray(heatmap.astype(np.uint8))
        img.save(str(filepath), format="PNG", optimize=True)

        return str(filepath)

    def generate_to_bytes(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        target_class: int = 1,
    ) -> bytes:
        """Generate heatmap and return as PNG bytes (for API response)."""
        heatmap_array, _ = self.generate(input_tensor, original_image, target_class)
        img = Image.fromarray(heatmap_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
