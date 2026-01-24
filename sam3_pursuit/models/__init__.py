"""Model components for the SAM3 fursuit recognition system."""

from sam3_pursuit.models.segmentor import FursuitSegmentor, SegmentationResult
from sam3_pursuit.models.embedder import FursuitEmbedder

__all__ = ["FursuitSegmentor", "SegmentationResult", "FursuitEmbedder"]
