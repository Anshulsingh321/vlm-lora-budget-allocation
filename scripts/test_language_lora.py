import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"


def main() -> None:
    # --------------------------------------------------
    # 1. Load pretrained CLIP
    # --------------------------------------------------
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # --------------------------------------------------
    # 2. Freeze the complete base model
    # --------------------------------------------------
    model.requires_grad_(False)

    # --------------------------------------------------
    # 3. Define LoRA configuration
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 4. Apply LoRA ONLY to the language encoder
    # --------------------------------------------------
    language_model = get_peft_model(
        model.text_model,
        config,
    )

    model.text_model = language_model

    # --------------------------------------------------
    # 5. Parameter counts
    # --------------------------------------------------
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    vision_trainable = sum(
        parameter.numel()
        for parameter in model.vision_model.parameters()
        if parameter.requires_grad
    )

    text_trainable = sum(
        parameter.numel()
        for parameter in model.text_model.parameters()
        if parameter.requires_grad
    )

    # --------------------------------------------------
    # 6. Print summary
    # --------------------------------------------------
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Vision trainable:     {vision_trainable:,}")
    print(f"Text trainable:       {text_trainable:,}")

    print("\nTrainable parameter names:\n")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)

    # --------------------------------------------------
    # 7. Assertions
    # --------------------------------------------------
    assert trainable_params == text_trainable
    assert vision_trainable == 0
    assert text_trainable > 0

    # Expected for:
    # r = 8
    # q_proj + v_proj
    # 12 transformer layers
    #
    # This should be:
    # 196,608 trainable parameters.
    expected_trainable = 196_608

    assert text_trainable == expected_trainable, (
        f"Unexpected trainable parameter count: "
        f"{text_trainable:,}. "
        f"Expected {expected_trainable:,}."
    )

    # --------------------------------------------------
    # 8. Final confirmation
    # --------------------------------------------------
    print("\nLoRA language-only configuration verified.")


if __name__ == "__main__":
    main()