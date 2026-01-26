"""Processing pipeline combining SAM3 segmentation and DINOv2 embedding."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.models.segmentor import FursuitSegmentor, SegmentationResult
from sam3_pursuit.models.embedder import FursuitEmbedder
from sam3_pursuit.models.preprocessor import BackgroundIsolator, IsolationConfig


@dataclass
class ProcessingResult:
    segmentation: SegmentationResult
    embedding: np.ndarray
    isolated_crop: Optional[Image.Image] = None  # Crop with background isolation applied
    segmentor_model: str = "unknown"


class ProcessingPipeline:
    """SAM3 segmentation + DINOv2 embedding pipeline."""

    def __init__(
        self,
        device: Optional[str] = None,
        sam_model: Optional[str] = None,
        dinov2_model: str = Config.DINOV2_MODEL,
        isolation_config: Optional[IsolationConfig] = None
    ):
        self.device = device or Config.get_device()
        print("Initializing processing pipeline...")
        self.segmentor = FursuitSegmentor(device=self.device, model_name=sam_model)
        self.embedder = FursuitEmbedder(device=self.device, model_name=dinov2_model)
        self.isolator = BackgroundIsolator(isolation_config)
        print("Pipeline ready")

    def process(self, image: Image.Image, concept: str = "fursuiter") -> list[ProcessingResult]:
        """Segment image and generate embeddings for each detection."""
        segmentations = self.segmentor.segment(image, concept=concept)

        results = []
        for seg in segmentations:
            # Apply background isolation before embedding
            isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
            embedding = self.embedder.embed(isolated)
            results.append(ProcessingResult(
                segmentation=seg,
                embedding=embedding,
                isolated_crop=isolated,
                segmentor_model=self.segmentor.model_name
            ))

        return results

    def process_full_image(self, image: Image.Image) -> ProcessingResult:
        """Process full image without segmentation (for single-character images)."""
        w, h = image.size
        full_mask = np.ones((h, w), dtype=np.uint8)
        segmentation = SegmentationResult(
            crop=image.copy(),
            mask=full_mask,
            crop_mask=full_mask,
            bbox=(0, 0, w, h),
            confidence=1.0
        )
        # Apply background isolation (no-op for full mask)
        isolated = self.isolator.isolate(image, full_mask)
        embedding = self.embedder.embed(isolated)

        return ProcessingResult(
            segmentation=segmentation,
            embedding=embedding,
            isolated_crop=isolated,
            segmentor_model=self.segmentor.model_name
        )

    def embed_only(self, image: Image.Image) -> np.ndarray:
        """Generate embedding without segmentation."""
        return self.embedder.embed(image)

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Generate embeddings for a batch of images."""
        return self.embedder.embed_batch(images)

    @property
    def segmentor_model_name(self) -> str:
        return self.segmentor.model_name

    @property
    def embedder_model_name(self) -> str:
        return self.embedder.model_name

    @property
    def isolation_config(self) -> IsolationConfig:
        return self.isolator.config
