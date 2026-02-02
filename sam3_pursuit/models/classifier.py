from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from sam3_pursuit.config import Config

class ImageClassifier:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = Config.CLIP_MODEL,
    ):
        self.device = device or Config.get_device()
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def classify(self, image: Image.Image) -> dict[str, float]:
        inputs = self.processor(
            text=Config.CLASSIFY_LABELS,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)[0]

        return {label: prob.item() for label, prob in zip(Config.CLASSIFY_LABELS, probs)}

    def is_fursuit(
        self,
        image: Image.Image,
        threshold: float = Config.DEFAULT_CLASSIFY_THRESHOLD,
    ) -> bool:
        scores = self.classify(image)
        return any(scores[l] >= threshold for l in Config.CLASSIFY_FURSUIT_LABELS)
