"""Storage for segmentation masks."""

import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config


def _normalize_concept(concept: str) -> str:
    """Normalize concept string for use in path (replace non-alphanumeric with _)."""
    return re.sub(r'[^a-zA-Z0-9]', '_', concept).strip('_') or "default"


class MaskStorage:
    """Handles saving and loading segmentation masks."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(Config.MASKS_DIR)

    def get_mask_dir(self, source: str, model: str, concept: str) -> Path:
        """Get directory for masks: {base}/{source}/{model}/{concept}/"""
        return self.base_dir / (source or "unknown") / model / _normalize_concept(concept)

    def save_mask(
        self,
        mask: np.ndarray,
        name: str,
        source: str,
        model: str,
        concept: str,
    ) -> str:
        """Save a segmentation mask as PNG.

        Args:
            mask: Binary mask array (H, W) with values 0-255 or 0-1
            name: Base name for the mask file (without extension)
            source: Ingestion source (e.g., "tgbot", "furtrack")
            model: Segmentor model name (e.g., "sam3")
            concept: Segmentation concept (e.g., "fursuiter head")

        Returns:
            Path to the saved mask file
        """
        target_dir = self.get_mask_dir(source, model, concept)
        target_dir.mkdir(parents=True, exist_ok=True)

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        path = target_dir / f"{name}.png"
        Image.fromarray(mask, mode="L").save(path, optimize=True)
        return str(path)

    def find_masks_for_post(self, post_id: str, source: str, model: str, concept: str) -> list[Path]:
        """Find all segment masks for a post_id ({post_id}_seg_*.png)."""
        mask_dir = self.get_mask_dir(source, model, concept)
        if not mask_dir.exists():
            return []
        return sorted(mask_dir.glob(f"{post_id}_seg_*.png"), key=lambda p: int(p.stem.split("_seg_")[-1]))

    def load_masks_for_post(self, post_id: str, source: str, model: str, concept: str) -> list[np.ndarray]:
        results = []
        for i, path in enumerate(self.find_masks_for_post(post_id, source, model, concept)):
            name = path.stem
            seg_idx = int(name.split("_seg_")[-1])
            assert seg_idx == i, f"Missing segment index {i} in mask files"
            results.append(self.load_mask(name, source, model, concept))
        return results

    def save_masks_for_post(self, post_id: str, source: str, model: str, concept: str, masks: list[np.ndarray]) -> list[str]:
        paths = []
        for i, mask in enumerate(masks):
            name = f"{post_id}_seg_{i}"
            path = self.save_mask(mask, name, source, model, concept)
            paths.append(path)
        return paths

    def save_no_segments_marker(self, post_id: str, source: str, model: str, concept: str) -> str:
        """Save a marker indicating the segmentor found no segments for this post."""
        target_dir = self.get_mask_dir(source, model, concept)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{post_id}.noseg"
        path.touch()
        return str(path)

    def has_no_segments_marker(self, post_id: str, source: str, model: str, concept: str) -> bool:
        """Check if a no-segments marker exists for this post."""
        return (self.get_mask_dir(source, model, concept) / f"{post_id}.noseg").exists()

    def load_mask(self, name: str, source: str, model: str, concept: str) -> Optional[np.ndarray]:
        path = self.get_mask_dir(source, model, concept) / f"{name}.png"
        if not path.exists():
            return None
        return np.array(Image.open(path).convert("L"))
