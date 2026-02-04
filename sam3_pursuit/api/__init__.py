"""API components for the SAM3 fursuit recognition system."""

from sam3_pursuit.api.annotator import annotate_image
from sam3_pursuit.api.identifier import FursuitIdentifier, IdentificationResult

__all__ = ["FursuitIdentifier", "IdentificationResult", "annotate_image"]
