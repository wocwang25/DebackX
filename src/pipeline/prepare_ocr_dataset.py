import argparse
from pathlib import Path

from PIL import Image

from common import clamp_box, dataset_root, load_config, numeric_jpgs, output_path, parse_box, read_lines


def prepare_split(config, config_path, split):
    root = dataset_root(config, config_path)
    source_lang = config["dataset"]["source_lang"]
    output_root = output_path(config_path, config["ocr"]["train_output_dir"])
    padding = int(config["ocr"].get("crop_padding", 0))

    split_root = root / split / source_lang
    image_dir = split_root / "image"
    subtitle_path = split_root / "subtitle.txt"
    pos_path = split_root / "pos.txt"

    subtitles = read_lines(subtitle_path)
    boxes = [parse_box(line) for line in read_lines(pos_path)]
    images = numeric_jpgs(image_dir)

    if not (len(images) == len(subtitles) == len(boxes)):
        raise ValueError(
            f"{split}: image/subtitle/pos counts differ: {len(images)}, {len(subtitles)}, {len(boxes)}"
        )

    crop_dir = output_root / split / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_root / split / config["ocr"].get("label_file", "labels.tsv")

    with label_path.open("w", encoding="utf-8", newline="\n") as label_file:
        for image_path, subtitle, box in zip(images, subtitles, boxes):
            with Image.open(image_path).convert("RGB") as image:
                crop_box = clamp_box(box, image.width, image.height, padding)
                crop = image.crop(crop_box)
                crop_name = f"{image_path.stem}.jpg"
                crop_path = crop_dir / crop_name
                crop.save(crop_path)
            label_file.write(f"images/{crop_name}\t{subtitle}\n")

    print(f"{split}: wrote {len(images)} crops -> {crop_dir}")
    print(f"{split}: labels -> {label_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare OCR crops and labels from IIMT30k-style data.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    splits = args.splits or config["dataset"]["splits"]
    for split in splits:
        prepare_split(config, config_path, split)


if __name__ == "__main__":
    main()
