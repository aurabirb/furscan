"""SAM3-based fursuit detection and segmentation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from ultralytics import SAM

from sam3_pursuit.config import Config


@dataclass
class SegmentationResult:
    """Result of fursuit segmentation."""
    crop: Image.Image  # Cropped region of the detected fursuit
    mask: np.ndarray  # Binary mask of the segmentation
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # Detection confidence score


class FursuitSegmentor:
    """SAM3-based fursuit detection and segmentation.

    Uses Meta's Segment Anything Model 3 (via ultralytics) to detect
    and segment fursuit characters in images.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = Config.SAM3_MODEL,
        confidence_threshold: float = Config.DETECTION_CONFIDENCE,
        max_detections: int = Config.MAX_DETECTIONS
    ):
        """Initialize the segmentor.

        Args:
            device: Device to run inference on (cuda/mps/cpu). Auto-detected if None.
            model_name: SAM3 model variant name.
            confidence_threshold: Minimum confidence for detections.
            max_detections: Maximum number of detections to return.
        """
        self.device = device or Config.get_device()
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections

        print(f"Loading SAM3 model: {model_name} on {self.device}")
        self.model = SAM(f"{model_name}.pt")
        print("SAM3 model loaded")

    def segment(self, image: Image.Image) -> list[SegmentationResult]:
        """Detect and segment fursuit instances in an image.

        Args:
            image: PIL Image to process.

        Returns:
            List of SegmentationResult objects for each detected fursuit.
        """
        # Convert PIL to numpy for ultralytics
        image_np = np.array(image.convert("RGB"))

        # Run SAM3 segmentation
        # Use automatic mask generation for full image segmentation
        results = self.model(image_np, device=self.device, verbose=False)

        segmentation_results = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i, mask in enumerate(masks):
                if i >= self.max_detections:
                    break

                # Get confidence score
                confidence = 1.0  # SAM doesn't provide confidence scores directly
                if boxes is not None and len(boxes.conf) > i:
                    confidence = float(boxes.conf[i])

                if confidence < self.confidence_threshold:
                    continue

                # Get bounding box from mask
                bbox = self._mask_to_bbox(mask)
                if bbox is None:
                    continue

                x, y, w, h = bbox

                # Create masked crop
                crop = self._create_crop(image, mask, bbox)

                segmentation_results.append(SegmentationResult(
                    crop=crop,
                    mask=mask.astype(np.uint8),
                    bbox=bbox,
                    confidence=confidence
                ))

        # If no masks found, return the whole image as a single result
        if not segmentation_results:
            w, h = image.size
            full_mask = np.ones((h, w), dtype=np.uint8)
            segmentation_results.append(SegmentationResult(
                crop=image.copy(),
                mask=full_mask,
                bbox=(0, 0, w, h),
                confidence=1.0
            ))

        return segmentation_results

    def segment_with_points(
        self,
        image: Image.Image,
        points: list[tuple[int, int]],
        labels: list[int]
    ) -> list[SegmentationResult]:
        """Segment using point prompts.

        Args:
            image: PIL Image to process.
            points: List of (x, y) point coordinates.
            labels: List of labels (1 for foreground, 0 for background).

        Returns:
            List of SegmentationResult objects.
        """
        image_np = np.array(image.convert("RGB"))

        results = self.model(
            image_np,
            points=points,
            labels=labels,
            device=self.device,
            verbose=False
        )

        segmentation_results = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()

            for mask in masks:
                bbox = self._mask_to_bbox(mask)
                if bbox is None:
                    continue

                crop = self._create_crop(image, mask, bbox)

                segmentation_results.append(SegmentationResult(
                    crop=crop,
                    mask=mask.astype(np.uint8),
                    bbox=bbox,
                    confidence=1.0
                ))

        return segmentation_results

    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Convert binary mask to bounding box.

        Returns:
            Tuple of (x, y, width, height) or None if mask is empty.
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))

    def _create_crop(
        self,
        image: Image.Image,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int]
    ) -> Image.Image:
        """Create a cropped image from the mask region.

        Args:
            image: Original PIL Image.
            mask: Binary segmentation mask.
            bbox: Bounding box (x, y, width, height).

        Returns:
            Cropped PIL Image.
        """
        x, y, w, h = bbox

        # Crop the image to bounding box
        crop_box = (x, y, x + w, y + h)
        cropped = image.crop(crop_box)

        return cropped
