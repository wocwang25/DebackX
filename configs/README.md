# Pipeline Config

The active configuration is:

```text
configs/config-pipeline-strong.json
```

## Sections

`dataset`

Controls the IIMT-style dataset root, split names, language names, and image size.

`ocr`

Controls OCR crop generation and TrOCR fine-tuning. `train_output_dir` receives cropped text regions and `labels.tsv` files. `checkpoint_dir` receives the fine-tuned OCR model.

`translation`

Controls NLLB fine-tuning for English to Vietnamese subtitle translation. `checkpoint_dir` receives the fine-tuned MT model.

`inpainting`

For benchmark/evaluation on `IIMT30k_Vi`, clean backgrounds are available in the dataset. For real images, use an inpainting model such as LaMa with masks from OCR boxes.

`real_image`

Controls the deployable worker path for user-uploaded images. The strong config uses PaddleOCR PP-OCRv5 for scene-text detection and recognition, the fine-tuned NLLB checkpoint for translation, OpenCV inpainting to remove old text, and the render config to insert Vietnamese text.

For subtitle-like images, `merge_text_regions` groups nearby OCR lines into one text block before translation so the MT model receives a complete sentence instead of isolated words or short fragments.

`worker`

Controls the FastAPI backend worker: upload limit, output directory, allowed image extensions, and whether OCR/MT models are loaded during startup.

`render`

Controls font, text color, box color, opacity, padding, and output directory for inserting Vietnamese text back into images.

## Training Defaults

The current defaults target an A100 GPU:

```text
ocr.precision = bf16
translation.precision = bf16
```

Run the preflight check before training:

```bash
sh scripts/check-training-env.sh
```
