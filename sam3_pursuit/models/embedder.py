"""DINOv2-based embedding generation for fursuit recognition."""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from sam3_pursuit.config import Config


class FursuitEmbedder:
    """DINOv2-based embedding generation.

    Uses Meta's DINOv2 model to generate visual embeddings suitable
    for similarity search. DINOv2 produces self-supervised embeddings
    that excel at fine-grained visual similarity.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = Config.DINOV2_MODEL
    ):
        """Initialize the embedder.

        Args:
            device: Device to run inference on (cuda/mps/cpu). Auto-detected if None.
            model_name: DINOv2 model name from HuggingFace.
        """
        self.device = device or Config.get_device()
        self.model_name = model_name

        print(f"Loading DINOv2 model: {model_name} on {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"DINOv2 model loaded. Embedding dimension: {self.embedding_dim}")

    def embed(self, image: Image.Image) -> np.ndarray:
        """Generate L2-normalized embedding for an image.

        Args:
            image: PIL Image to embed.

        Returns:
            L2-normalized embedding as numpy array.
        """
        # Ensure RGB format
        image = image.convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :]

            # L2 normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten()

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Generate embeddings for a batch of images.

        Args:
            images: List of PIL Images to embed.

        Returns:
            Array of L2-normalized embeddings, shape (n_images, embedding_dim).
        """
        if not images:
            return np.array([], dtype=np.float32)

        # Convert all images to RGB
        images = [img.convert("RGB") for img in images]

        # Process batch
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :]

            # L2 normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy().astype(np.float32)
