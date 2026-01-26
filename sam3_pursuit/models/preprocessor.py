"""Background isolation for improving embedding quality."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class IsolationConfig:
    """Configuration for background isolation."""
    mode: str = "solid"  # "none", "solid", "blur"
    background_color: tuple[int, int, int] = (128, 128, 128)  # RGB for solid mode
    blur_radius: int = 25  # for blur mode

    def __post_init__(self):
        if self.mode not in ("none", "solid", "blur"):
            raise ValueError(f"Invalid isolation mode: {self.mode}")


class BackgroundIsolator:
    """Isolates subject from background using segmentation mask."""

    def __init__(self, config: Optional[IsolationConfig] = None):
        self.config = config or IsolationConfig()

    def isolate(self, crop: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply mask to isolate subject, fill background according to config.

        Args:
            crop: Cropped image of the detected subject
            mask: Binary mask (same size as crop), 1 = subject, 0 = background

        Returns:
            Image with background isolated according to config mode
        """
        if self.config.mode == "none":
            return crop

        # Ensure mask matches crop size
        mask = self._resize_mask(mask, crop.size)

        if self.config.mode == "solid":
            return self._apply_solid_background(crop, mask)
        elif self.config.mode == "blur":
            return self._apply_blurred_background(crop, mask)

        return crop

    def _resize_mask(self, mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize mask to match target size if needed.

        Args:
            mask: Binary mask array
            target_size: (width, height) tuple

        Returns:
            Resized mask array
        """
        target_w, target_h = target_size
        mask_h, mask_w = mask.shape[:2]

        if mask_w == target_w and mask_h == target_h:
            return mask

        # Convert to PIL for resizing, then back to numpy
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
        return (np.array(mask_img) > 127).astype(np.uint8)

    def _apply_solid_background(self, crop: Image.Image, mask: np.ndarray) -> Image.Image:
        """Composite subject on solid color background.

        Args:
            crop: Source image
            mask: Binary mask

        Returns:
            Image with solid color background
        """
        # Convert to RGBA for compositing
        crop_rgba = crop.convert("RGBA")

        # Create solid background
        background = Image.new("RGBA", crop.size, self.config.background_color + (255,))

        # Create alpha mask from segmentation mask
        alpha = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        # Composite: subject over background using mask as alpha
        result = Image.composite(crop_rgba, background, alpha)

        return result.convert("RGB")

    def _apply_blurred_background(self, crop: Image.Image, mask: np.ndarray) -> Image.Image:
        """Composite subject on blurred version of itself.

        Args:
            crop: Source image
            mask: Binary mask

        Returns:
            Image with blurred background
        """
        # Convert to RGBA for compositing
        crop_rgba = crop.convert("RGBA")

        # Create blurred background
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=self.config.blur_radius))
        blurred_rgba = blurred.convert("RGBA")

        # Create alpha mask from segmentation mask
        alpha = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        # Composite: subject over blurred background using mask as alpha
        result = Image.composite(crop_rgba, blurred_rgba, alpha)

        return result.convert("RGB")
