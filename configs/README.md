# Pipeline Config

The active application configuration is:

```text
configs/config-pipeline-strong.json
```

## Sections

`dataset`

Controls the IIMT-style dataset root, split names, language names, and image/background locations used for translation training and benchmark rendering.

`translation`

Controls NLLB 1.3B fine-tuning for English to Vietnamese text translation. `checkpoint_dir` receives the trained MT model.

`real_image`

Controls the deployable image translation path. The strong config uses PaddleOCR PP-OCRv5 for scene-text detection/recognition, the fine-tuned NLLB checkpoint for translation, OpenCV inpainting to remove old text, and the render config to insert Vietnamese text.

`worker`

Controls the FastAPI backend worker: upload limit, output directory, allowed image extensions, and whether models are loaded during startup.

`render`

Controls font, adaptive text color, stroke, box opacity, padding, and output directory for inserting Vietnamese text back into images.

## Main Runtime

The production worker uses:

```text
PaddleOCR PP-OCRv5
models/mt-nllb-1p3b-en-vi/best
OpenCV inpainting
adaptive text renderer
```

OCR training is not part of the production flow.
