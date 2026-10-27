import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"


def main() -> None:
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Freeze the complete base model first.
    model.requires_grad_(False)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        task_type="FEATURE_EXTRACTION",
    )

    # Important:
    # We only want LoRA adapters in the vision encoder.
    vision_model = get_peft_model(
        model.vision_model,
        config,
    )

    model.vision_model = vision_model

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    vision_trainable = sum(
        p.numel()
        for p in model.vision_model.parameters()
        if p.requires_grad
    )

    text_trainable = sum(
        p.numel()
        for p in model.text_model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Vision trainable:     {vision_trainable:,}")
    print(f"Text trainable:       {text_trainable:,}")

    print("\nTrainable parameter names:\n")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)

    print("\nLoRA vision-only configuration verified.")


if __name__ == "__main__":
    main()