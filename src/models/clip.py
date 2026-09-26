import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


class FrozenCLIP:
    def __init__(self, device: torch.device) -> None:
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model = CLIPModel.from_pretrained(MODEL_NAME)

        # Baseline = completely frozen.
        self.model.requires_grad_(False)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _extract_features(output) -> torch.Tensor:
        """
        Transformers versions may return either a tensor
        or a BaseModelOutputWithPooling object.
        """
        if isinstance(output, torch.Tensor):
            return output

        if hasattr(output, "pooler_output"):
            return output.pooler_output

        raise TypeError(
            f"Unsupported CLIP output type: {type(output)}"
        )

    @torch.inference_mode()
    def encode_images(self, images) -> torch.Tensor:
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        output = self.model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        features = self._extract_features(output)

        # CLIP similarity is computed in normalized embedding space.
        return F.normalize(features, p=2, dim=-1)

    @torch.inference_mode()
    def encode_texts(self, texts) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        output = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        features = self._extract_features(output)

        return F.normalize(features, p=2, dim=-1)