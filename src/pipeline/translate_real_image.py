import argparse
import json
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from common import clamp_box, load_config, output_path, resolve_from_config
from render_translations import draw_translated_text


def precision_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def use_amp(name):
    return name in {"bf16", "fp16"} and torch.cuda.is_available()


def resolve_path_from_cwd(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def default_output_path(config_path, config, input_path):
    output_dir = output_path(config_path, config["real_image"]["output_dir"])
    return output_dir / f"{input_path.stem}.vi.png"


def polygon_to_box(polygon):
    points = np.array(polygon, dtype=np.float32)
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    return x1, y1, x2, y2


def normalize_polygon(raw_polygon):
    return [[float(point[0]), float(point[1])] for point in raw_polygon]


def detect_text_regions(image_path, real_config):
    reader = easyocr.Reader(
        real_config.get("detector_languages", ["en"]),
        gpu=bool(real_config.get("detector_gpu", True)) and torch.cuda.is_available(),
    )
    detections = reader.readtext(
        str(image_path),
        detail=1,
        paragraph=False,
        text_threshold=float(real_config.get("detector_text_threshold", 0.5)),
        low_text=float(real_config.get("detector_low_text", 0.3)),
        link_threshold=float(real_config.get("detector_link_threshold", 0.4)),
        width_ths=float(real_config.get("detector_width_ths", 0.7)),
    )

    min_confidence = float(real_config.get("min_confidence", 0.0))
    regions = []
    for index, item in enumerate(detections):
        polygon, detector_text, confidence = item
        confidence = float(confidence)
        if confidence < min_confidence:
            continue
        normalized_polygon = normalize_polygon(polygon)
        regions.append(
            {
                "index": index,
                "polygon": normalized_polygon,
                "box": polygon_to_box(normalized_polygon),
                "detector_text": detector_text,
                "detector_confidence": confidence,
            }
        )
    regions.sort(key=lambda region: (region["box"][1], region["box"][0]))
    return regions


class CropDataset(Dataset):
    def __init__(self, image, regions, padding):
        self.image = image
        self.regions = regions
        self.padding = padding

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, index):
        box = clamp_box(self.regions[index]["box"], self.image.width, self.image.height, self.padding)
        return self.image.crop(box), index


class OcrCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        crops, indexes = zip(*batch)
        return {
            "pixel_values": self.processor(images=list(crops), return_tensors="pt").pixel_values,
            "indexes": list(indexes),
        }


@torch.no_grad()
def recognize_regions(image, regions, config, config_path):
    if not regions:
        return []

    real_config = config["real_image"]
    ocr_config = config["ocr"]
    checkpoint = resolve_from_config(config_path, real_config["ocr_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"OCR checkpoint not found: {checkpoint}. Train OCR first.")

    processor = TrOCRProcessor.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = CropDataset(image, regions, int(real_config.get("box_padding", 0)))
    dataloader = DataLoader(
        dataset,
        batch_size=ocr_config["eval_batch_size"],
        shuffle=False,
        collate_fn=OcrCollator(processor),
        pin_memory=torch.cuda.is_available(),
    )

    predictions = [""] * len(regions)
    precision = ocr_config.get("precision", "bf16")
    amp_dtype = precision_dtype(precision)
    for batch in tqdm(dataloader, desc="ocr"):
        pixel_values = batch["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp(precision)):
            generated = model.generate(
                pixel_values,
                max_length=ocr_config["max_target_length"],
                num_beams=4,
            )
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        for index, text in zip(batch["indexes"], decoded):
            predictions[index] = text.strip()
    return predictions


class TextDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]


@torch.no_grad()
def translate_texts(texts, config, config_path):
    if not texts:
        return []

    real_config = config["real_image"]
    mt_config = config["translation"]
    checkpoint = resolve_from_config(config_path, real_config["mt_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"MT checkpoint not found: {checkpoint}. Train translation first.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer.src_lang = mt_config["source_code"]
    tokenizer.tgt_lang = mt_config["target_code"]
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(mt_config["target_code"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataloader = DataLoader(TextDataset(texts), batch_size=mt_config["eval_batch_size"], shuffle=False)
    translations = []
    for batch in tqdm(dataloader, desc="translate"):
        encoded = tokenizer(
            list(batch),
            max_length=mt_config["max_source_length"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            max_length=mt_config["max_target_length"],
            num_beams=mt_config["num_beams"],
            forced_bos_token_id=forced_bos_token_id,
        )
        translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [translation.strip() for translation in translations]


def make_mask(image_size, regions, real_config):
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    padding = int(real_config.get("box_padding", 0))
    for region in regions:
        x1, y1, x2, y2 = clamp_box(region["box"], width, height, padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    dilation = int(real_config.get("mask_dilation", 0))
    if dilation > 0:
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def inpaint_image(image, regions, real_config):
    if not regions:
        return image.copy(), np.zeros((image.height, image.width), dtype=np.uint8)

    mask = make_mask(image.size, regions, real_config)
    radius = float(real_config.get("inpaint_radius", 3))
    backend = real_config.get("inpaint_backend", "opencv_telea")
    method = cv2.INPAINT_NS if backend == "opencv_ns" else cv2.INPAINT_TELEA
    rgb = np.array(image.convert("RGB"))
    inpainted = cv2.inpaint(rgb, mask, radius, method)
    return Image.fromarray(inpainted), mask


def render_regions(image, regions, translations, render_config):
    rendered = image.copy()
    for region, translation in zip(regions, translations):
        box = clamp_box(region["box"], rendered.width, rendered.height, padding=0)
        rendered = draw_translated_text(rendered, box, translation, render_config)
    return rendered


def write_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, ensure_ascii=False, indent=2)


def process_image(config_path, input_path, output, metadata):
    config, config_file = load_config(config_path)
    input_path = resolve_path_from_cwd(input_path)
    output_path = resolve_path_from_cwd(output) if output else default_output_path(config_file, config, input_path)
    metadata_path = (
        resolve_path_from_cwd(metadata)
        if metadata
        else output_path.with_suffix(output_path.suffix + config["real_image"].get("metadata_suffix", ".json"))
    )

    with Image.open(input_path) as opened:
        image = opened.convert("RGB")

    regions = detect_text_regions(input_path, config["real_image"])
    ocr_texts = recognize_regions(image, regions, config, config_file)
    translations = translate_texts(ocr_texts, config, config_file)
    clean_image, mask = inpaint_image(image, regions, config["real_image"])
    rendered = render_regions(clean_image, regions, translations, config["render"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)

    mask_path = output_path.with_suffix(".mask.png")
    Image.fromarray(mask).save(mask_path)

    for region, ocr_text, translation in zip(regions, ocr_texts, translations):
        region["ocr_text"] = ocr_text
        region["translation"] = translation

    write_metadata(
        metadata_path,
        {
            "input": str(input_path),
            "output": str(output_path),
            "mask": str(mask_path),
            "num_regions": len(regions),
            "regions": regions,
        },
    )
    print(f"detected regions: {len(regions)}")
    print(f"wrote image -> {output_path}")
    print(f"wrote mask -> {mask_path}")
    print(f"wrote metadata -> {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate English text in a real image into Vietnamese.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to real_image.output_dir.")
    parser.add_argument("--metadata", default=None, help="Output metadata JSON path.")
    args = parser.parse_args()

    process_image(args.config, args.input, args.output, args.metadata)


if __name__ == "__main__":
    main()
