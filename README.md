# IIMT English-Vietnamese Pipeline

This project follows a modular In-Image Machine Translation pipeline:

```text
image with English text
-> crop or detect text region
-> recognize English text with OCR
-> translate English text to Vietnamese
-> use clean background or inpaint old text
-> render Vietnamese text back into the image
```

The active dataset root is `IIMT30k_Vi/Arial`.

## What Gets Trained

`OCR`

Fine-tune `microsoft/trocr-base-printed` on cropped English text regions from `IIMT30k_Vi/Arial/{train,val}/en`.

TrOCR large was tested but overfit heavily on this dataset, so the dataset OCR benchmark uses the base checkpoint that generalizes better.

`Translation`

Fine-tune `facebook/nllb-200-1.3B` for English to Vietnamese using:

```text
IIMT30k_Vi/Arial/train/en/subtitle.txt
IIMT30k_Vi/Arial/train/vi/subtitle.txt
IIMT30k_Vi/Arial/val/en/subtitle.txt
IIMT30k_Vi/Arial/val/vi/subtitle.txt
```

`Rendering`

Rendering text back into the image is not trained. It is a deterministic post-processing step controlled by `configs/config-pipeline-strong.json`.

## Dataset

Expected structure:

```text
IIMT30k_Vi/Arial/{train,val,test}/background
IIMT30k_Vi/Arial/{train,val,test}/en/image
IIMT30k_Vi/Arial/{train,val,test}/en/text
IIMT30k_Vi/Arial/{train,val,test}/en/subtitle.txt
IIMT30k_Vi/Arial/{train,val,test}/en/pos.txt
IIMT30k_Vi/Arial/{train,val,test}/vi/image
IIMT30k_Vi/Arial/{train,val,test}/vi/text
IIMT30k_Vi/Arial/{train,val,test}/vi/subtitle.txt
IIMT30k_Vi/Arial/{train,val,test}/vi/pos.txt
```

The active training scripts use `subtitle.txt` and `pos.txt`. Tokenized files can stay in the dataset, but they are not required for the current OCR + MT pipeline.

## Configuration

Main application config:

```text
configs/config-pipeline-strong.json
```

The main config uses:

```text
Translation: facebook/nllb-200-1.3B
Dataset OCR: microsoft/trocr-base-printed
Real image OCR: PaddleOCR PP-OCRv5
```

It also enables application-quality rendering:

```text
PaddleOCR PP-OCRv5 for large real images
merge nearby subtitle lines before translation
lower text-box opacity
text stroke/outline
adaptive text color from the original English image
polygon-based inpainting mask for real images
```

Important outputs:

```text
outputs/ocr/en/{train,val,test}/images
outputs/ocr/en/{train,val,test}/labels.tsv
outputs/ocr/en/test.pred.en.txt
models/ocr-trocr-en/best
models/mt-nllb-1p3b-en-vi/best
outputs/mt/test.1p3b.pred.vi.txt
outputs/mt/test.from-ocr.1p3b.pred.vi.txt
outputs/rendered_strong/test/vi/image
outputs/real_images_strong
```

## Complete Workflow

Use this order when training and using the project end to end.

### 1. Setup Environment

Use Python 3.10 or 3.11 for the pinned PyTorch/Transformers stack.

Create a Python environment on the GPU machine:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

For the stronger real-image OCR backend, install PaddleOCR as well:

```bash
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python3 -m pip install -r requirements-paddleocr.txt
```

If the server does not use CUDA 11.8, choose the PaddlePaddle wheel that matches the server CUDA version.

Check whether the machine and dataset are ready:

```bash
sh scripts/check-training-env.sh
```

The main config is already set to use `bf16`, which is suitable for A100.

### 2. Prepare OCR Data

Create text crops and OCR labels from `en/image`, `en/pos.txt`, and `en/subtitle.txt`:

```bash
sh scripts/prepare-ocr-dataset.sh
```

`scripts/train-ocr.sh` also runs this preparation step automatically.

Expected OCR data outputs:

```text
outputs/ocr/en/train/images
outputs/ocr/en/train/labels.tsv
outputs/ocr/en/val/images
outputs/ocr/en/val/labels.tsv
outputs/ocr/en/test/images
outputs/ocr/en/test/labels.tsv
```

### 3. Train Translation

Train the English to Vietnamese translation model first. This is the main quality bottleneck of the system.

Training input:

```text
IIMT30k_Vi/Arial/train/en/subtitle.txt
IIMT30k_Vi/Arial/train/vi/subtitle.txt
IIMT30k_Vi/Arial/val/en/subtitle.txt
IIMT30k_Vi/Arial/val/vi/subtitle.txt
```

Run:

```bash
CONFIG=configs/config-pipeline-strong.json sh scripts/train-translation.sh
```

Outputs:

```text
models/mt-nllb-1p3b-en-vi/best
models/mt-nllb-1p3b-en-vi/last
models/mt-nllb-1p3b-en-vi/metrics.json
```

Use `best` for inference unless visual inspection shows `last` is better.

### 3.1. Translation Quality Check

Predict the test split with the 1.3B checkpoint:

```bash
CONFIG=configs/config-pipeline-strong.json \
CHECKPOINT=models/mt-nllb-1p3b-en-vi/best \
OUTPUT=outputs/mt/test.1p3b.pred.vi.txt \
SPLIT=test \
sh scripts/predict-translation.sh
```

Analyze the 1.3B errors:

```bash
CONFIG=configs/config-pipeline-strong.json \
PREDICTIONS=outputs/mt/test.1p3b.pred.vi.txt \
OUTPUT=outputs/mt/test.1p3b.translation-errors.tsv \
TOP=200 \
sh scripts/analyze-translation-errors.sh
```

If there are still bad sentences, try decode variants without retraining:

```bash
CONFIG=configs/config-pipeline-strong.json \
CHECKPOINT=models/mt-nllb-1p3b-en-vi/best \
TAG=1p3b \
BEAMS="4 5 6" \
LENGTH_PENALTIES="0.9 1.0 1.1" \
SPLIT=test \
sh scripts/predict-translation-variants.sh
```

This writes files such as:

```text
outputs/mt/test.1p3b.beam4.lp1p0.pred.vi.txt
outputs/mt/test.1p3b.beam5.lp1p0.pred.vi.txt
outputs/mt/test.1p3b.beam6.lp1p1.pred.vi.txt
```

Pick the best-looking file by inspecting its `.errors.tsv` file and several rendered images.

Render the selected translation file:

```bash
sh scripts/render-translations.sh \
  --config configs/config-pipeline-strong.json \
  --split test \
  --translations outputs/mt/test.1p3b.pred.vi.txt
```

If a decode variant is better, replace the `--translations` path with that variant file.

### 3.2. Application-Quality Mode

For the final application output, use the main strong config and adaptive renderer.

Recommended production/demo path:

```text
MT:     models/mt-nllb-1p3b-en-vi/best
Real-image OCR: PaddleOCR PP-OCRv5
Dataset OCR:    models/ocr-trocr-en/best
Render: configs/config-pipeline-strong.json adaptive render settings
```

For uploaded images, OCR uses PaddleOCR PP-OCRv5:

```text
PP-OCRv5_server_det detects scene-text boxes
en_PP-OCRv5_mobile_rec recognizes English text
nearby subtitle lines are merged before translation
```

This is more robust for large images because uploaded images are scene-text data, while TrOCR was trained here on clean cropped text regions.

The renderer now improves visual quality with:

```text
box_alpha = 0.25 for less intrusive background
stroke_width = 1 for readable text without a heavy black box
adaptive_text_color = true to reuse the original text color when possible
mask_from_polygon = true to reduce damage to real-image backgrounds
```

Render the best translation output with the strong renderer:

```bash
sh scripts/render-translations.sh \
  --config configs/config-pipeline-strong.json \
  --split test \
  --translations outputs/mt/test.1p3b.pred.vi.txt
```

Run real-image CLI with the strong config:

```bash
CONFIG=configs/config-pipeline-strong.json \
sh scripts/translate-image.sh --input path/to/english-image.jpg
```

### 4. Test Translation And Render Benchmark

Generate Vietnamese translations for the test split using ground-truth English subtitles:

```bash
CONFIG=configs/config-pipeline-strong.json \
CHECKPOINT=models/mt-nllb-1p3b-en-vi/best \
OUTPUT=outputs/mt/test.1p3b.pred.vi.txt \
SPLIT=test \
sh scripts/predict-translation.sh
```

Output:

```text
outputs/mt/test.1p3b.pred.vi.txt
```

Render those translations onto the clean benchmark backgrounds:

```bash
sh scripts/render-translations.sh \
  --config configs/config-pipeline-strong.json \
  --split test \
  --translations outputs/mt/test.1p3b.pred.vi.txt
```

Output:

```text
outputs/rendered_strong/test/vi/image
```

This benchmark checks:

```text
ground-truth English text -> MT model -> Vietnamese text -> render into image
```

It intentionally skips OCR so translation and rendering quality can be inspected first.

For an oracle render benchmark with ground-truth Vietnamese subtitles:

```bash
sh scripts/render-translations.sh --config configs/config-pipeline-strong.json --split test
```

### 5. Train OCR

Train the English OCR model after OCR crops are prepared:

```bash
CONFIG=configs/config-pipeline-strong.json sh scripts/train-ocr.sh
```

Outputs:

```text
models/ocr-trocr-en/best
models/ocr-trocr-en/last
models/ocr-trocr-en/metrics.json
```

Use `best` for inference unless visual inspection shows `last` is better.

### 6. Run Full Dataset Pipeline

Run OCR on the test crops:

```bash
CONFIG=configs/config-pipeline-strong.json CHECKPOINT=models/ocr-trocr-en/best SPLIT=test sh scripts/predict-ocr.sh
```

Translate the OCR text to Vietnamese:

```bash
CONFIG=configs/config-pipeline-strong.json \
INPUT=outputs/ocr/en/test.pred.en.txt \
OUTPUT=outputs/mt/test.from-ocr.1p3b.pred.vi.txt \
CHECKPOINT=models/mt-nllb-1p3b-en-vi/best \
SPLIT=test \
sh scripts/predict-translation.sh
```

Output:

```text
outputs/mt/test.from-ocr.1p3b.pred.vi.txt
```

Render the generated Vietnamese text onto clean test backgrounds:

```bash
sh scripts/render-translations.sh \
  --config configs/config-pipeline-strong.json \
  --split test \
  --translations outputs/mt/test.from-ocr.1p3b.pred.vi.txt
```

Output:

```text
outputs/rendered_strong/test/vi/image
```

This full dataset pipeline checks:

```text
English image crop -> OCR -> MT -> render Vietnamese into clean background
```

### 7. Translate A Real Image With CLI

After both checkpoints exist, run the deployable worker path on a user image:

```bash
CONFIG=configs/config-pipeline-strong.json \
sh scripts/translate-image.sh --input path/to/english-image.jpg
```

Default outputs:

```text
outputs/real_images_strong/english-image.vi.png       # final translated image
outputs/real_images_strong/english-image.vi.mask.png  # text-removal mask
outputs/real_images_strong/english-image.vi.png.json  # boxes, OCR text, translations
```

This path adds the two parts that the dataset benchmark does not need:

```text
PaddleOCR PP-OCRv5 -> finds and reads text in the uploaded image
OpenCV inpainting -> removes the original English text before rendering Vietnamese
```

You can choose the output path manually:

```bash
CONFIG=configs/config-pipeline-strong.json \
sh scripts/translate-image.sh --input path/to/english-image.jpg --output outputs/real_images_strong/result.png
```

### 8. Start Backend Worker

After training, start the FastAPI worker:

```bash
CONFIG=configs/config-pipeline-strong.json sh scripts/run-worker.sh
```

Default server:

```text
http://0.0.0.0:8000
```

The worker loads PaddleOCR PP-OCRv5 and NLLB 1.3B once at startup and keeps them in memory. TrOCR base is still kept for the dataset OCR benchmark, but the real-image worker no longer depends on TrOCR for OCR quality. If you want to start the API before checkpoints exist, set `worker.load_models_on_startup` to `false` in `configs/config-pipeline-strong.json`; the first translation job will load the models lazily.

Health check:

```bash
curl http://localhost:8000/health
```

Create an async translation job:

```bash
curl -X POST http://localhost:8000/jobs \
  -F "file=@path/to/english-image.jpg"
```

Check job status:

```bash
curl http://localhost:8000/jobs/<job_id>
```

Run a synchronous translation request for demos:

```bash
curl -X POST http://localhost:8000/translate \
  -F "file=@path/to/english-image.jpg"
```

Successful job responses include:

```text
result.output_url    # translated image URL
result.mask_url      # inpainting mask URL
result.metadata_url  # JSON with boxes, OCR text, and translations
```

Worker outputs are stored under:

```text
outputs/worker_strong/uploads
outputs/worker_strong/results
outputs/worker_strong/jobs
```

For Docker GPU deployment:

```bash
docker build -f Dockerfile.worker -t iimt-worker .
docker run --gpus all --rm -p 8000:8000 \
  -e CONFIG=configs/config-pipeline-strong.json \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  iimt-worker
```

## Practical Notes

For the graduation project, the trainable parts are OCR and machine translation. The full system still satisfies the IIMT goal because it recognizes text inside images, translates it, and reconstructs a translated image.

For real-world images that do not have clean `background` files, `scripts/translate-image.sh` and the FastAPI worker use OCR boxes as masks and OpenCV inpainting as the baseline text-removal method. A stronger inpainting model such as LaMa can replace this later for better visual quality.

Recommended demo order:

```text
1. Show translation metrics and outputs/mt/test.1p3b.pred.vi.txt examples.
2. Show rendered benchmark images from outputs/rendered_strong/test/vi/image.
3. Show full OCR -> MT -> render examples after OCR training.
4. Show backend worker API if the web app integration is needed.
```
