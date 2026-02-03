"""API components for the SAM3 fursuit recognition system."""

from sam3_pursuit.api.annotator import annotate_image
from sam3_pursuit.api.identifier import SAM3FursuitIdentifier, IdentificationResult

__all__ = ["SAM3FursuitIdentifier", "IdentificationResult", "annotate_image"]
