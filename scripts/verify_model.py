import torch
import transformers
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Transformers: {transformers.__version__}")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    # Load processor and pretrained CLIP.
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Phase 3 baseline: completely frozen model.
    model.requires_grad_(False)
    model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Simple text input.
    text = ["A dog running in a field."]

    # Simple synthetic image for model sanity checking.
    image = Image.new("RGB", (224, 224), color="white")

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        text_output = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        image_output = model.get_image_features(
            pixel_values=inputs["pixel_values"],
        )

    # Current Transformers returns BaseModelOutputWithPooling.
    text_features = text_output.pooler_output
    image_features = image_output.pooler_output

    print(f"Text embedding shape:  {tuple(text_features.shape)}")
    print(f"Image embedding shape: {tuple(image_features.shape)}")

    # Sanity checks.
    assert not model.training, "Model is still in training mode."
    assert trainable_params == 0, "Some model parameters are still trainable."

    assert text_features.ndim == 2
    assert image_features.ndim == 2

    assert text_features.shape == image_features.shape, (
        "Image and text embeddings must have the same shape."
    )

    assert not torch.isnan(text_features).any(), "NaN in text embeddings."
    assert not torch.isnan(image_features).any(), "NaN in image embeddings."

    print("\nModel verification passed.")


if __name__ == "__main__":
    main()