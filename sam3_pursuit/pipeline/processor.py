"""Main processing pipeline combining segmentation and embedding."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.models.segmentor import FursuitSegmentor, SegmentationResult
from sam3_pursuit.models.embedder import FursuitEmbedder


@dataclass
class ProcessingResult:
    """Result of processing an image through the pipeline."""
    segmentation: SegmentationResult
    embedding: np.ndarray


class ProcessingPipeline:
    """Main processing pipeline for fursuit recognition.

    Combines SAM3 segmentation with DINOv2 embedding generation
    to process images for indexing or identification.

    When SAM3 is available, uses text prompts ("fursuit") for targeted detection.
    Falls back to generic segmentation with SAM2.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sam_model: Optional[str] = None,
        dinov2_model: str = Config.DINOV2_MODEL
    ):
        """Initialize the processing pipeline.

        Args:
            device: Device for inference. Auto-detected if None.
            sam_model: SAM model name. If None, auto-selects SAM3 or SAM2.
            dinov2_model: DINOv2 model name.
        """
        self.device = device or Config.get_device()

        print("Initializing processing pipeline...")
        self.segmentor = FursuitSegmentor(device=self.device, model_name=sam_model)
        self.embedder = FursuitEmbedder(device=self.device, model_name=dinov2_model)

        if self.segmentor.supports_text_prompts:
            print("Pipeline initialized with SAM3 - text prompts enabled!")
        else:
            print("Pipeline initialized with SAM2 - using generic segmentation")

    def process(
        self,
        image: Image.Image,
        concept: str = "fursuit"
    ) -> list[ProcessingResult]:
        """Process an image through the full pipeline.

        Uses SAM3 text prompts when available for targeted fursuit detection.
        Falls back to generic segmentation with SAM2.

        Args:
            image: PIL Image to process.
            concept: Text concept for SAM3 (default: "fursuit").

        Returns:
            List of ProcessingResult objects, one per detected fursuit.
        """
        # Use fursuit-specific segmentation (SAM3 text prompt or SAM2 fallback)
        segmentations = self.segmentor.segment_by_concept(image, concept=concept)

        results = []
        for seg in segmentations:
            # Generate embedding for the cropped region
            embedding = self.embedder.embed(seg.crop)

            results.append(ProcessingResult(
                segmentation=seg,
                embedding=embedding
            ))

        return results

    def process_fursuits(self, image: Image.Image) -> list[ProcessingResult]:
        """Process image specifically looking for fursuits.

        Convenience method that uses "fursuit" as the concept.

        Args:
            image: PIL Image to process.

        Returns:
            List of ProcessingResult objects, one per detected fursuit.
        """
        return self.process(image, concept="fursuit")

    def process_full_image(self, image: Image.Image) -> ProcessingResult:
        """Process full image without segmentation.

        Useful for images known to contain a single fursuit (e.g., badge photos).

        Args:
            image: PIL Image to process.

        Returns:
            Single ProcessingResult for the full image.
        """
        w, h = image.size

        # Create a dummy segmentation result for the full image
        segmentation = SegmentationResult(
            crop=image.copy(),
            mask=np.ones((h, w), dtype=np.uint8),
            bbox=(0, 0, w, h),
            confidence=1.0
        )

        # Generate embedding
        embedding = self.embedder.embed(image)

        return ProcessingResult(
            segmentation=segmentation,
            embedding=embedding
        )

    def embed_only(self, image: Image.Image) -> np.ndarray:
        """Generate embedding without segmentation.

        Args:
            image: PIL Image to embed.

        Returns:
            Embedding array.
        """
        return self.embedder.embed(image)

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Generate embeddings for a batch of images.

        Args:
            images: List of PIL Images.

        Returns:
            Array of embeddings, shape (n, embedding_dim).
        """
        return self.embedder.embed_batch(images)

    @property
    def supports_text_prompts(self) -> bool:
        """Check if the pipeline supports SAM3 text prompts."""
        return self.segmentor.supports_text_prompts
