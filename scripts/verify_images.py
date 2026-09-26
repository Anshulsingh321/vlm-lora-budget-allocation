from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image


CSV_PATH = Path("data/raw/flickr30k/flickr_annotations_30k.csv")
ZIP_PATH = Path("data/raw/flickr30k/flickr30k-images.zip")

SAMPLE_SIZE = 20


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    filenames = df["filename"].astype(str).tolist()

    # Sample deterministically for reproducibility.
    sample = filenames[:SAMPLE_SIZE]

    with ZipFile(ZIP_PATH) as z:
        zip_names = set(z.namelist())

        for filename in sample:
            member = f"flickr30k-images/{filename}"

            assert member in zip_names, (
                f"Image not found in archive: {member}"
            )

            with z.open(member) as file:
                image_bytes = file.read()

            image = Image.open(BytesIO(image_bytes))

            # Force actual decoding.
            image.verify()

            # Re-open because verify() invalidates the image object.
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            width, height = image.size

            assert width > 0 and height > 0
            assert image.mode == "RGB"

            print(
                f"{filename}: "
                f"{width}x{height}, "
                f"mode={image.mode}"
            )

    print(f"\nImage integrity check passed for {len(sample)} images.")


if __name__ == "__main__":
    main()