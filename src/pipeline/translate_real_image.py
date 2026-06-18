import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
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


def box_to_polygon(box):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def union_box(boxes):
    points = np.array(boxes, dtype=np.float32)
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 2].max())
    y2 = float(points[:, 3].max())
    return x1, y1, x2, y2


def region_height(region):
    x1, y1, x2, y2 = region["box"]
    return max(1.0, float(y2) - float(y1))


def region_text(region):
    return region.get("detector_text", "").strip()


def as_python_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def result_to_dict(result):
    if isinstance(result, dict):
        payload = result
    elif hasattr(result, "json"):
        raw_json = result.json
        payload = raw_json() if callable(raw_json) else raw_json
    elif hasattr(result, "res"):
        payload = result.res
    else:
        payload = vars(result)

    if isinstance(payload, dict) and "res" in payload:
        return payload["res"]
    return payload


def parse_paddleocr_result(result, min_confidence):
    payload = result_to_dict(result)
    if not isinstance(payload, dict):
        return []

    texts = as_python_list(payload.get("rec_texts"))
    scores = as_python_list(payload.get("rec_scores"))
    polygons = payload.get("rec_polys")
    if polygons is None:
        polygons = payload.get("dt_polys")
    polygons = as_python_list(polygons)
    boxes = as_python_list(payload.get("rec_boxes"))

    regions = []
    for index, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue

        confidence = float(scores[index]) if index < len(scores) else 1.0
        if confidence < min_confidence:
            continue

        polygon = None
        if index < len(polygons):
            polygon = normalize_polygon(polygons[index])
        elif index < len(boxes):
            polygon = box_to_polygon(boxes[index])

        if not polygon:
            continue

        regions.append(
            {
                "index": index,
                "polygon": polygon,
                "box": polygon_to_box(polygon),
                "detector_text": text,
                "detector_confidence": confidence,
                "detector_backend": "paddleocr",
            }
        )
    return regions


def parse_legacy_paddleocr_result(result, min_confidence):
    if not result:
        return []
    page = result[0] if len(result) == 1 and isinstance(result[0], list) else result
    regions = []
    for index, item in enumerate(page or []):
        if not item or len(item) < 2:
            continue
        polygon = normalize_polygon(item[0])
        text, confidence = item[1]
        text = str(text).strip()
        confidence = float(confidence)
        if not text or confidence < min_confidence:
            continue
        regions.append(
            {
                "index": index,
                "polygon": polygon,
                "box": polygon_to_box(polygon),
                "detector_text": text,
                "detector_confidence": confidence,
                "detector_backend": "paddleocr_legacy",
            }
        )
    return regions


def resample_bicubic():
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def polygon_crop(image, polygon, padding):
    points = np.array(polygon, dtype=np.float32)
    if points.shape != (4, 2):
        return None

    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[2] - points[3])
    left_height = np.linalg.norm(points[3] - points[0])
    right_height = np.linalg.norm(points[2] - points[1])
    width = max(1, int(round(max(top_width, bottom_width))) + padding * 2)
    height = max(1, int(round(max(left_height, right_height))) + padding * 2)

    target = np.array(
        [
            [padding, padding],
            [width - padding - 1, padding],
            [width - padding - 1, height - padding - 1],
            [padding, height - padding - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        np.array(image.convert("RGB")),
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(crop)


def preprocess_ocr_crop(crop, real_config):
    crop = crop.convert("RGB")
    min_height = int(real_config.get("ocr_crop_min_height", 0))
    max_scale = float(real_config.get("ocr_crop_max_scale", 1.0))
    if min_height > 0 and crop.height > 0 and crop.height < min_height:
        scale = min(min_height / crop.height, max_scale)
        if scale > 1.0:
            new_size = (
                max(1, int(round(crop.width * scale))),
                max(1, int(round(crop.height * scale))),
            )
            crop = crop.resize(new_size, resample=resample_bicubic())

    contrast = float(real_config.get("ocr_crop_contrast", 1.0))
    if contrast != 1.0:
        crop = ImageEnhance.Contrast(crop).enhance(contrast)

    sharpness = float(real_config.get("ocr_crop_sharpness", 1.0))
    if sharpness != 1.0:
        crop = ImageEnhance.Sharpness(crop).enhance(sharpness)

    return crop


def sort_regions_by_reading_order(regions, vertical_overlap_threshold=0.5):
    """
    Sắp xếp các vùng chứa chữ theo thứ tự đọc tự nhiên của con người:
    1. Nhóm các hộp chữ nằm trên cùng một dòng ngang (dựa trên độ chồng lấp chiều dọc).
    2. Sắp xếp các dòng từ trên xuống dưới.
    3. Trong mỗi dòng, sắp xếp các hộp chữ từ trái qua phải.
    """
    if not regions:
        return []

    # Sắp xếp sơ bộ theo y1 để dễ xử lý tuần tự
    sorted_by_y = sorted(regions, key=lambda r: r["box"][1])
    
    lines = []
    current_line = [sorted_by_y[0]]
    
    for r in sorted_by_y[1:]:
        prev_r = current_line[-1]
        prev_box = prev_r["box"]
        curr_box = r["box"]
        
        # Tính toán độ chồng lấp chiều dọc (Vertical Overlap)
        overlap = min(prev_box[3], curr_box[3]) - max(prev_box[1], curr_box[1])
        h_prev = prev_box[3] - prev_box[1]
        h_curr = curr_box[3] - curr_box[1]
        min_h = min(h_prev, h_curr)
        
        if min_h > 0 and overlap / min_h >= vertical_overlap_threshold:
            current_line.append(r)
        else:
            lines.append(current_line)
            current_line = [r]
            
    if current_line:
        lines.append(current_line)
        
    final_regions = []
    for line in lines:
        sorted_line = sorted(line, key=lambda r: r["box"][0])
        final_regions.extend(sorted_line)
        
    for idx, r in enumerate(final_regions):
        r["index"] = idx
        
    return final_regions


def merge_text_regions(regions, real_config):
    if not real_config.get("merge_text_regions", False) or len(regions) < 2:
        return regions

    # Kiểm tra xem các vùng chữ có cấu trúc sentence_id được cung cấp bởi LLM không
    has_sentence_ids = all("sentence_id" in r for r in regions)
    
    groups = []
    if has_sentence_ids:
        # Nhóm theo sentence_id nhưng giữ nguyên thứ tự sắp xếp hiện tại
        current_id = regions[0]["sentence_id"]
        current_group = [regions[0]]
        for region in regions[1:]:
            if region["sentence_id"] == current_id:
                current_group.append(region)
            else:
                groups.append(current_group)
                current_id = region["sentence_id"]
                current_group = [region]
        groups.append(current_group)
    else:
        # Sử dụng thuật toán so sánh khoảng cách dọc (y_gap)
        max_gap_ratio = float(real_config.get("merge_line_gap_ratio", 1.6))
        max_gap_px = float(real_config.get("merge_line_gap_px", 18))
        
        current_group = [regions[0]]
        for region in regions[1:]:
            previous = current_group[-1]
            previous_height = region_height(previous)
            y_gap = float(region["box"][1]) - float(previous["box"][3])
            max_allowed_gap = max(max_gap_px, previous_height * max_gap_ratio)
            if y_gap <= max_allowed_gap:
                current_group.append(region)
            else:
                groups.append(current_group)
                current_group = [region]
        groups.append(current_group)

    merged_regions = []
    for group_index, group in enumerate(groups):
        if len(group) == 1:
            single = dict(group[0])
            single["index"] = group_index
            merged_regions.append(single)
            continue

        # Giữ nguyên thứ tự đọc ban đầu khi vẽ
        box = union_box([region["box"] for region in group])
        text = " ".join(region_text(region) for region in group if region_text(region))
        confidence = min(float(region.get("detector_confidence", 1.0)) for region in group)
        merged_regions.append(
            {
                "index": group_index,
                "polygon": box_to_polygon(box),
                "box": box,
                "detector_text": text,
                "detector_confidence": confidence,
                "detector_backend": group[0].get("detector_backend", "unknown"),
                "merged_from": group,
            }
        )
    return merged_regions


class CropDataset(Dataset):
    def __init__(self, image, regions, real_config):
        self.image = image
        self.regions = regions
        self.real_config = real_config
        self.padding = int(real_config.get("box_padding", 0))

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, index):
        region = self.regions[index]
        crop = None
        if self.real_config.get("ocr_use_polygon_crop", True) and region.get("polygon"):
            crop = polygon_crop(self.image, region["polygon"], self.padding)
        if crop is None:
            box = clamp_box(region["box"], self.image.width, self.image.height, self.padding)
            crop = self.image.crop(box)
        return preprocess_ocr_crop(crop, self.real_config), index


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
    padding = int(real_config.get("mask_padding", real_config.get("box_padding", 0)))
    mask_regions = []
    for region in regions:
        if real_config.get("mask_merged_sources", True) and region.get("merged_from"):
            mask_regions.extend(region["merged_from"])
        else:
            mask_regions.append(region)

    for region in mask_regions:
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
        self.ocr_config = self.config.get("ocr", {})
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
    def detector_backend(self):
        return self.real_config.get("detector", "paddleocr")

    @property
    def recognizer_backend(self):
        return self.real_config.get("recognizer", "trocr")

    @property
    def uses_trocr_recognizer(self):
        return self.recognizer_backend in {"trocr", "hybrid"}

    @property
    def output_dir(self):
        return output_path(self.config_file, self.real_config["output_dir"])

    @property
    def ocr_checkpoint(self):
        raw_checkpoint = self.real_config.get("ocr_checkpoint")
        if not raw_checkpoint:
            return None
        checkpoint = resolve_from_config(self.config_file, raw_checkpoint)
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
        payload = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "detector": self.detector_backend,
            "recognizer": self.recognizer_backend,
            "mt_checkpoint": str(self.mt_checkpoint),
            "mt_checkpoint_exists": self.mt_checkpoint.exists(),
            "models_loaded": self.models_loaded,
            "output_dir": str(self.output_dir),
        }
        if self.uses_trocr_recognizer:
            checkpoint = self.ocr_checkpoint
            payload["ocr_checkpoint"] = str(checkpoint) if checkpoint else None
            payload["ocr_checkpoint_exists"] = bool(checkpoint and checkpoint.exists())
        return payload

    @property
    def models_loaded(self):
        required = [self.detector, self.mt_tokenizer, self.mt_model]
        if self.uses_trocr_recognizer:
            required.extend([self.ocr_processor, self.ocr_model])
        return all(item is not None for item in required)

    def load(self):
        self.load_detector()
        if self.uses_trocr_recognizer:
            self.load_ocr()
        self.load_mt()

    def load_detector(self):
        if self.detector is not None:
            return

        if self.detector_backend == "easyocr":
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError(
                    "EasyOCR backend is enabled but easyocr is not installed. "
                    "Use the PaddleOCR strong config or install easyocr for the legacy backend."
                ) from exc

            self.detector = easyocr.Reader(
                self.real_config.get("detector_languages", ["en"]),
                gpu=bool(self.real_config.get("detector_gpu", True)) and torch.cuda.is_available(),
            )
            return

        if self.detector_backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR backend is enabled but PaddleOCR is not installed. "
                    "Install the optional PaddleOCR dependencies documented in README.md."
                ) from exc

            device = self.real_config.get("paddleocr_device")
            if not device:
                device = "gpu:0" if torch.cuda.is_available() and self.real_config.get("detector_gpu", True) else "cpu"

            self.detector = PaddleOCR(
                use_doc_orientation_classify=bool(self.real_config.get("paddleocr_use_doc_orientation", False)),
                use_doc_unwarping=bool(self.real_config.get("paddleocr_use_doc_unwarping", False)),
                use_textline_orientation=bool(self.real_config.get("paddleocr_use_textline_orientation", False)),
                lang=self.real_config.get("paddleocr_lang", "en"),
                ocr_version=self.real_config.get("paddleocr_version", "PP-OCRv5"),
                text_detection_model_name=self.real_config.get(
                    "paddleocr_text_detection_model_name", "PP-OCRv5_server_det"
                ),
                text_recognition_model_name=self.real_config.get(
                    "paddleocr_text_recognition_model_name", "en_PP-OCRv5_mobile_rec"
                ),
                text_recognition_batch_size=int(self.real_config.get("paddleocr_text_recognition_batch_size", 8)),
                text_det_limit_side_len=int(self.real_config.get("paddleocr_text_det_limit_side_len", 4096)),
                text_det_limit_type=self.real_config.get("paddleocr_text_det_limit_type", "max"),
                text_det_thresh=float(self.real_config.get("paddleocr_text_det_thresh", 0.25)),
                text_det_box_thresh=float(self.real_config.get("paddleocr_text_det_box_thresh", 0.5)),
                text_det_unclip_ratio=float(self.real_config.get("paddleocr_text_det_unclip_ratio", 1.8)),
                text_rec_score_thresh=float(self.real_config.get("paddleocr_text_rec_score_thresh", 0.35)),
                device=device,
            )
            return

        raise ValueError(f"Unsupported detector backend: {self.detector_backend}")

    def load_ocr(self):
        if self.ocr_model is not None and self.ocr_processor is not None:
            return
        checkpoint = self.ocr_checkpoint
        if checkpoint is None:
            raise ValueError("TrOCR recognizer is enabled but real_image.ocr_checkpoint is not configured.")
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

    def reconstruct_layout_and_sort(self, regions):
        if not regions:
            return []

        # Đánh lại index ban đầu để ánh xạ
        for idx, r in enumerate(regions):
            r["index"] = idx

        # Thử sử dụng LLM bổ trợ trước nếu có GOOGLE_API_KEY
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                from vlm_translator import GeminiLayoutHelper
                helper = GeminiLayoutHelper()
                if helper.is_available:
                    layout = helper.reconstruct_layout(regions)
                    if layout:
                        # Tạo ánh xạ từ box_id sang sentence_id
                        box_sentence_map = {b.box_id: b.sentence_id for b in layout.boxes}
                        
                        # Sắp xếp các vùng theo reading_order
                        regions_by_id = {r["index"]: r for r in regions}
                        sorted_regions = []
                        for box_id in layout.reading_order:
                            if box_id in regions_by_id:
                                r = regions_by_id[box_id]
                                r["sentence_id"] = box_sentence_map.get(box_id, 0)
                                sorted_regions.append(r)
                                
                        # Thêm các vùng bị bỏ sót nếu có
                        for r in regions:
                            if r["index"] not in layout.reading_order:
                                r["sentence_id"] = 999
                                sorted_regions.append(r)
                                
                        # Đánh lại index sau khi sắp xếp đúng thứ tự đọc
                        for idx, r in enumerate(sorted_regions):
                            r["index"] = idx
                        return sorted_regions
            except Exception as exc:
                print(f"[reconstruct_layout_and_sort] LLM failed: {exc}. Chuyển sang fallback thuật toán local.")

        # Thuật toán local bổ trợ sắp xếp dọc-ngang
        return sort_regions_by_reading_order(regions)

    def detect_text_regions(self, image_path):
        self.load_detector()

        if self.detector_backend == "paddleocr":
            min_confidence = float(self.real_config.get("min_confidence", 0.0))
            if hasattr(self.detector, "predict"):
                results = self.detector.predict(
                    str(image_path),
                    use_doc_orientation_classify=bool(self.real_config.get("paddleocr_use_doc_orientation", False)),
                    use_doc_unwarping=bool(self.real_config.get("paddleocr_use_doc_unwarping", False)),
                    use_textline_orientation=bool(self.real_config.get("paddleocr_use_textline_orientation", False)),
                    text_det_limit_side_len=int(self.real_config.get("paddleocr_text_det_limit_side_len", 4096)),
                    text_det_limit_type=self.real_config.get("paddleocr_text_det_limit_type", "max"),
                    text_det_thresh=float(self.real_config.get("paddleocr_text_det_thresh", 0.25)),
                    text_det_box_thresh=float(self.real_config.get("paddleocr_text_det_box_thresh", 0.5)),
                    text_det_unclip_ratio=float(self.real_config.get("paddleocr_text_det_unclip_ratio", 1.8)),
                    text_rec_score_thresh=float(self.real_config.get("paddleocr_text_rec_score_thresh", 0.35)),
                )
                regions = []
                for result in results:
                    regions.extend(parse_paddleocr_result(result, min_confidence))
            else:
                raw_result = self.detector.ocr(str(image_path), cls=False)
                regions = parse_legacy_paddleocr_result(raw_result, min_confidence)

            regions = self.reconstruct_layout_and_sort(regions)
            return merge_text_regions(regions, self.real_config)

        detections = self.detector.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            text_threshold=float(self.real_config.get("detector_text_threshold", 0.5)),
            low_text=float(self.real_config.get("detector_low_text", 0.3)),
            link_threshold=float(self.real_config.get("detector_link_threshold", 0.4)),
            width_ths=float(self.real_config.get("detector_width_ths", 0.7)),
            canvas_size=int(self.real_config.get("detector_canvas_size", 2560)),
            mag_ratio=float(self.real_config.get("detector_mag_ratio", 1.0)),
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
        regions = self.reconstruct_layout_and_sort(regions)
        return merge_text_regions(regions, self.real_config)

    @torch.no_grad()
    def recognize_regions(self, image, regions):
        if not regions:
            return []

        recognizer = self.recognizer_backend
        detector_texts = [region.get("detector_text", "").strip() for region in regions]
        if recognizer in {"easyocr", "detector", "paddleocr"}:
            return detector_texts

        self.load_ocr()

        dataset = CropDataset(image, regions, self.real_config)
        dataloader = DataLoader(
            dataset,
            batch_size=int(self.ocr_config.get("eval_batch_size", 8)),
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
                    max_length=int(self.ocr_config.get("max_target_length", 128)),
                    num_beams=4,
                )
            decoded = self.ocr_processor.batch_decode(generated, skip_special_tokens=True)
            for index, text in zip(batch["indexes"], decoded):
                predictions[index] = text.strip()

        if recognizer == "hybrid":
            confidence_threshold = float(self.real_config.get("detector_text_min_confidence", 0.65))
            merged_predictions = []
            for region, detector_text, trocr_text in zip(regions, detector_texts, predictions):
                detector_confidence = float(region.get("detector_confidence", 0.0))
                if detector_text and detector_confidence >= confidence_threshold:
                    merged_predictions.append(detector_text)
                else:
                    merged_predictions.append(trocr_text)
            return merged_predictions

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
    parser.add_argument("--config", default="configs/config-pipeline-strong.json")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to real_image.output_dir.")
    parser.add_argument("--metadata", default=None, help="Output metadata JSON path.")
    args = parser.parse_args()

    process_image(args.config, args.input, args.output, args.metadata)


if __name__ == "__main__":
    main()
