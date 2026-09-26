from pathlib import Path
from zipfile import ZipFile

import pandas as pd


CSV_PATH = Path("data/raw/flickr30k/flickr_annotations_30k.csv")
ZIP_PATH = Path("data/raw/flickr30k/flickr30k-images.zip")
OUTPUT_DIR = Path("data/processed/flickr30k/images")


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    filenames = set(df["filename"].astype(str))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0

    with ZipFile(ZIP_PATH) as z:
        members = {
            Path(name).name: name
            for name in z.namelist()
            if name.startswith("flickr30k-images/")
            and name.lower().endswith(".jpg")
        }

        for filename in sorted(filenames):
            if filename not in members:
                raise FileNotFoundError(
                    f"Annotated image missing from archive: {filename}"
                )

            output_path = OUTPUT_DIR / filename

            if output_path.exists():
                skipped += 1
                continue

            with z.open(members[filename]) as source:
                output_path.write_bytes(source.read())

            extracted += 1

    print(f"Annotated images: {len(filenames):,}")
    print(f"Extracted: {extracted:,}")
    print(f"Already existed: {skipped:,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()