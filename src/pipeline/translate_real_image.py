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


def polygon_to_box(polygon):
    points = np.array(polygon, dtype=np.float32)
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    return x1, y1, x2, y2


def normalize_polygon(raw_polygon):
    return [[float(point[0]), float(point[1])] for point in raw_polygon]


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


class TextDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]


def make_mask(image_size, regions, real_config):
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    padding = int(real_config.get("box_padding", 0))
    for region in regions:
        if real_config.get("mask_from_polygon", True) and region.get("polygon"):
            points = np.array(region["polygon"], dtype=np.float32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
        else:
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


def render_regions(image, regions, translations, render_config, style_source=None):
    rendered = image.copy()
    for region, translation in zip(regions, translations):
        box = clamp_box(region["box"], rendered.width, rendered.height, padding=0)
        rendered = draw_translated_text(rendered, box, translation, render_config, style_source)
    return rendered


def write_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, ensure_ascii=False, indent=2)


class RealImageTranslator:
    def __init__(self, config_path, lazy=False):
        self.config, self.config_file = load_config(config_path)
        self.real_config = self.config["real_image"]
        self.ocr_config = self.config["ocr"]
        self.mt_config = self.config["translation"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = None
        self.ocr_processor = None
        self.ocr_model = None
        self.mt_tokenizer = None
        self.mt_model = None
        self.mt_forced_bos_token_id = None

        if not lazy:
            self.load()

    @property
    def output_dir(self):
        return output_path(self.config_file, self.real_config["output_dir"])

    @property
    def ocr_checkpoint(self):
        checkpoint = resolve_from_config(self.config_file, self.real_config["ocr_checkpoint"])
        fallback = self.real_config.get("fallback_ocr_checkpoint")
        if checkpoint.exists() or not fallback:
            return checkpoint
        fallback_checkpoint = resolve_from_config(self.config_file, fallback)
        if fallback_checkpoint.exists():
            return fallback_checkpoint
        return checkpoint

    @property
    def mt_checkpoint(self):
        return resolve_from_config(self.config_file, self.real_config["mt_checkpoint"])

    def default_output_path(self, input_path):
        return self.output_dir / f"{input_path.stem}.vi.png"

    def health(self):
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "ocr_checkpoint": str(self.ocr_checkpoint),
            "ocr_checkpoint_exists": self.ocr_checkpoint.exists(),
            "mt_checkpoint": str(self.mt_checkpoint),
            "mt_checkpoint_exists": self.mt_checkpoint.exists(),
            "models_loaded": self.models_loaded,
            "output_dir": str(self.output_dir),
        }

    @property
    def models_loaded(self):
        return all(
            item is not None
            for item in [
                self.detector,
                self.ocr_processor,
                self.ocr_model,
                self.mt_tokenizer,
                self.mt_model,
            ]
        )

    def load(self):
        self.load_detector()
        self.load_ocr()
        self.load_mt()

    def load_detector(self):
        if self.detector is not None:
            return
        self.detector = easyocr.Reader(
            self.real_config.get("detector_languages", ["en"]),
            gpu=bool(self.real_config.get("detector_gpu", True)) and torch.cuda.is_available(),
        )

    def load_ocr(self):
        if self.ocr_model is not None and self.ocr_processor is not None:
            return
        checkpoint = self.ocr_checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"OCR checkpoint not found: {checkpoint}. Train OCR first.")
        self.ocr_processor = TrOCRProcessor.from_pretrained(checkpoint)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
        self.ocr_model.to(self.device).eval()

    def load_mt(self):
        if self.mt_model is not None and self.mt_tokenizer is not None:
            return
        checkpoint = self.mt_checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"MT checkpoint not found: {checkpoint}. Train translation first.")
        self.mt_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.mt_tokenizer.src_lang = self.mt_config["source_code"]
        self.mt_tokenizer.tgt_lang = self.mt_config["target_code"]
        self.mt_forced_bos_token_id = self.mt_tokenizer.convert_tokens_to_ids(self.mt_config["target_code"])
        self.mt_model.to(self.device).eval()

    def detect_text_regions(self, image_path):
        self.load_detector()
        detections = self.detector.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            text_threshold=float(self.real_config.get("detector_text_threshold", 0.5)),
            low_text=float(self.real_config.get("detector_low_text", 0.3)),
            link_threshold=float(self.real_config.get("detector_link_threshold", 0.4)),
            width_ths=float(self.real_config.get("detector_width_ths", 0.7)),
        )

        min_confidence = float(self.real_config.get("min_confidence", 0.0))
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

    @torch.no_grad()
    def recognize_regions(self, image, regions):
        if not regions:
            return []
        self.load_ocr()

        dataset = CropDataset(image, regions, int(self.real_config.get("box_padding", 0)))
        dataloader = DataLoader(
            dataset,
            batch_size=self.ocr_config["eval_batch_size"],
            shuffle=False,
            collate_fn=OcrCollator(self.ocr_processor),
            pin_memory=torch.cuda.is_available(),
        )

        predictions = [""] * len(regions)
        precision = self.ocr_config.get("precision", "bf16")
        amp_dtype = precision_dtype(precision)
        for batch in tqdm(dataloader, desc="ocr"):
            pixel_values = batch["pixel_values"].to(self.device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp(precision)):
                generated = self.ocr_model.generate(
                    pixel_values,
                    max_length=self.ocr_config["max_target_length"],
                    num_beams=4,
                )
            decoded = self.ocr_processor.batch_decode(generated, skip_special_tokens=True)
            for index, text in zip(batch["indexes"], decoded):
                predictions[index] = text.strip()
        return predictions

    @torch.no_grad()
    def translate_texts(self, texts):
        if not texts:
            return []
        self.load_mt()

        dataloader = DataLoader(TextDataset(texts), batch_size=self.mt_config["eval_batch_size"], shuffle=False)
        translations = []
        for batch in tqdm(dataloader, desc="translate"):
            encoded = self.mt_tokenizer(
                list(batch),
                max_length=self.mt_config["max_source_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            generated = self.mt_model.generate(
                **encoded,
                max_length=self.mt_config["max_target_length"],
                num_beams=self.mt_config["num_beams"],
                forced_bos_token_id=self.mt_forced_bos_token_id,
            )
            translations.extend(self.mt_tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [translation.strip() for translation in translations]

    def process_image(self, input_path, output=None, metadata=None):
        input_path = resolve_path_from_cwd(input_path)
        output_path = resolve_path_from_cwd(output) if output else self.default_output_path(input_path)
        metadata_path = (
            resolve_path_from_cwd(metadata)
            if metadata
            else output_path.with_suffix(output_path.suffix + self.real_config.get("metadata_suffix", ".json"))
        )

        with Image.open(input_path) as opened:
            image = opened.convert("RGB")

        regions = self.detect_text_regions(input_path)
        ocr_texts = self.recognize_regions(image, regions)
        translations = self.translate_texts(ocr_texts)
        clean_image, mask = inpaint_image(image, regions, self.real_config)
        rendered = render_regions(clean_image, regions, translations, self.config["render"], style_source=image)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(output_path)

        mask_path = output_path.with_suffix(".mask.png")
        Image.fromarray(mask).save(mask_path)

        for region, ocr_text, translation in zip(regions, ocr_texts, translations):
            region["ocr_text"] = ocr_text
            region["translation"] = translation

        result = {
            "input": str(input_path),
            "output": str(output_path),
            "mask": str(mask_path),
            "metadata": str(metadata_path),
            "num_regions": len(regions),
            "regions": regions,
        }
        write_metadata(metadata_path, result)
        return result


def process_image(config_path, input_path, output, metadata):
    translator = RealImageTranslator(config_path)
    result = translator.process_image(input_path, output, metadata)
    print(f"detected regions: {result['num_regions']}")
    print(f"wrote image -> {result['output']}")
    print(f"wrote mask -> {result['mask']}")
    print(f"wrote metadata -> {result['metadata']}")


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
