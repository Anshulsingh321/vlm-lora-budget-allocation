import torch

from transformers import CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"


def main() -> None:
    model = CLIPModel.from_pretrained(MODEL_NAME)

    print("\n=== Vision attention modules ===")

    for name, module in model.named_modules():
        if (
            "vision_model" in name
            and any(
                target in name
                for target in ["q_proj", "k_proj", "v_proj", "out_proj"]
            )
        ):
            print(f"{name}: {module.__class__.__name__}")

    print("\n=== Text attention modules ===")

    for name, module in model.named_modules():
        if (
            "text_model" in name
            and any(
                target in name
                for target in ["q_proj", "k_proj", "v_proj", "out_proj"]
            )
        ):
            print(f"{name}: {module.__class__.__name__}")

    print("\n=== Parameter counts by major component ===")

    components = {
        "vision": model.vision_model,
        "text": model.text_model,
        "visual_projection": model.visual_projection,
        "text_projection": model.text_projection,
    }

    for name, component in components.items():
        params = sum(
            p.numel()
            for p in component.parameters()
        )

        print(f"{name:20s}: {params:,}")


if __name__ == "__main__":
    main()