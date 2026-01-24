"""Configuration constants for the SAM3 fursuit recognition system."""

import os
import torch


class Config:
    """Configuration settings for the SAM3 system."""

    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Database paths
    DB_PATH = os.path.join(BASE_DIR, "furtrack_sam3.db")
    INDEX_PATH = os.path.join(BASE_DIR, "faiss_sam3.index")
    IMAGES_DIR = os.path.join(BASE_DIR, "furtrack_images")

    # Old system paths for migration
    OLD_DB_PATH = os.path.join(BASE_DIR, "furtrack.db")

    # Model settings
    SAM3_MODEL = "sam2.1_s"  # SAM3 model variant (sam2.1_s, sam2.1_b, sam2.1_l)
    DINOV2_MODEL = "facebook/dinov2-base"  # DINOv2 model name
    EMBEDDING_DIM = 768  # DINOv2 base output dimension

    # Detection settings
    DETECTION_CONFIDENCE = 0.5  # Minimum confidence for SAM3 detections
    MAX_DETECTIONS = 10  # Maximum detections per image

    # Search settings
    DEFAULT_TOP_K = 5
    HNSW_M = 32  # HNSW index parameter
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 50

    # Batch processing
    DEFAULT_BATCH_SIZE = 16

    # Device selection
    @staticmethod
    def get_device() -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            # Set environment variables for MacOS compatibility
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['FAISS_OPT_LEVEL'] = ''
            return "mps"
        return "cpu"

    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        """Convert relative path to absolute based on BASE_DIR."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(cls.BASE_DIR, relative_path)
