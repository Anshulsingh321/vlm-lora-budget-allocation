import torch

from src.data.flickr30k import Flickr30kDataset
from src.models.clip import FrozenCLIP


ANNOTATIONS = "data/raw/flickr30k/flickr_annotations_30k.csv"
IMAGE_DIR = "data/processed/flickr30k/images"


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = Flickr30kDataset(
        annotation_path=ANNOTATIONS,
        image_dir=IMAGE_DIR,
        split="test",
    )

    sample = dataset[0]

    model = FrozenCLIP(device=device)

    image_embedding = model.encode_images(
        [sample["image"]]
    )

    text_embedding = model.encode_texts(
        sample["captions"]
    )

    print("Device:", device)

    print(
        "Image embedding shape:",
        tuple(image_embedding.shape),
    )

    print(
        "Text embedding shape:",
        tuple(text_embedding.shape),
    )

    print(
        "Image embedding norm:",
        image_embedding.norm(dim=-1).tolist(),
    )

    print(
        "Text embedding norms:",
        text_embedding.norm(dim=-1).tolist(),
    )

    # Expected:
    # image  -> [1, 512]
    # captions -> [5, 512]

    assert image_embedding.shape == (1, 512)
    assert text_embedding.shape == (5, 512)

    # L2-normalized embeddings should have norm ~1.
    assert torch.allclose(
        image_embedding.norm(dim=-1),
        torch.ones(1, device=device),
        atol=1e-5,
    )

    assert torch.allclose(
        text_embedding.norm(dim=-1),
        torch.ones(5, device=device),
        atol=1e-5,
    )

    assert not torch.isnan(image_embedding).any()
    assert not torch.isnan(text_embedding).any()

    print("\nEmbedding smoke test passed.")


if __name__ == "__main__":
    main()