import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.append(str(Path(__file__).resolve().parent))
from common import cer, precision_dtype, resolve_from_config, save_json, use_amp, wer, load_config


class OcrCropDataset(Dataset):
    def __init__(self, labels_path):
        self.root = Path(labels_path).parent
        self.samples = []
        with Path(labels_path).open("r", encoding="utf-8") as label_file:
            for line in label_file:
                image_rel, text = line.rstrip("\n").split("\t", 1)
                self.samples.append((self.root / image_rel, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, text = self.samples[index]
        return image_path, text


class OcrCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        image_paths, texts = zip(*batch)
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        return {
            "pixel_values": pixel_values,
            "texts": list(texts),
        }


@torch.no_grad()
def predict(config_path, checkpoint, split, output):
    config, config_file = load_config(config_path)
    ocr_config = config["ocr"]
    labels_root = resolve_from_config(config_file, ocr_config["train_output_dir"])
    label_path = labels_root / split / ocr_config.get("label_file", "labels.tsv")
    if not label_path.exists():
        raise FileNotFoundError("OCR labels not found. Run `sh scripts/prepare-ocr-dataset.sh` first.")

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    processor = TrOCRProcessor.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = OcrCropDataset(label_path)
    dataloader = DataLoader(
        dataset,
        batch_size=ocr_config["eval_batch_size"],
        shuffle=False,
        num_workers=ocr_config["num_workers"],
        collate_fn=OcrCollator(processor),
        pin_memory=torch.cuda.is_available(),
    )

    predictions = []
    references = []
    precision = ocr_config.get("precision", "bf16")
    amp_dtype = precision_dtype(precision)
    for batch in tqdm(dataloader, desc=f"ocr predict {split}"):
        pixel_values = batch["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp(precision)):
            generated = model.generate(
                pixel_values,
                max_length=ocr_config["max_target_length"],
                num_beams=4,
            )
        predictions.extend(processor.batch_decode(generated, skip_special_tokens=True))
        references.extend(batch["texts"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for prediction in predictions:
            output_file.write(prediction + "\n")

    metrics = {
        "split": split,
        "cer": cer(predictions, references),
        "wer": wer(predictions, references),
        "num_samples": len(predictions),
        "predictions": predictions[:20],
        "references": references[:20],
    }
    save_json(output_path.with_suffix(".metrics.json"), metrics)
    print(f"wrote OCR predictions -> {output_path}")
    print(f"cer={metrics['cer']:.4f} wer={metrics['wer']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate English OCR text from prepared crops.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--checkpoint", default="models/ocr-trocr-en/best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/ocr/en/test.pred.en.txt")
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.split, args.output)


if __name__ == "__main__":
    main()
