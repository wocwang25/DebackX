# IIMT English-Vietnamese Worker

This project is an application-focused In-Image Machine Translation worker:

```text
image with English text
-> PaddleOCR PP-OCRv5 detects and reads English text
-> fine-tuned NLLB 1.3B translates English to Vietnamese
-> OpenCV inpaints the old English text
-> adaptive renderer inserts Vietnamese text back into the image
```

The main application config is:

```text
configs/config-pipeline-strong.json
```

## Main Components

`OCR`

Uses PaddleOCR PP-OCRv5 pretrained models for real images:

```text
PP-OCRv5_server_det       # scene-text detection
en_PP-OCRv5_mobile_rec    # English text recognition
```

No OCR training is required for the production worker.

`Translation`

Fine-tunes `facebook/nllb-200-1.3B` for English to Vietnamese subtitle-style text:

```text
IIMT30k_Vi/Arial/train/en/subtitle.txt
IIMT30k_Vi/Arial/train/vi/subtitle.txt
IIMT30k_Vi/Arial/val/en/subtitle.txt
IIMT30k_Vi/Arial/val/vi/subtitle.txt
```

`Rendering`

Rendering is deterministic post-processing. It uses lower box opacity, text stroke, adaptive text color, polygon masks, and subtitle-line merging for cleaner final images.

## Expected Data

The active dataset root is `IIMT30k_Vi/Arial`.

```text
IIMT30k_Vi/Arial/{train,val,test}/background
IIMT30k_Vi/Arial/{train,val,test}/en/image
IIMT30k_Vi/Arial/{train,val,test}/en/subtitle.txt
IIMT30k_Vi/Arial/{train,val,test}/vi/subtitle.txt
```

The main training flow uses the English/Vietnamese `subtitle.txt` files. Benchmark rendering also uses the clean `background` images.

## Outputs

Important outputs:

```text
models/mt-nllb-1p3b-en-vi/best
models/mt-nllb-1p3b-en-vi/last
models/mt-nllb-1p3b-en-vi/metrics.json
outputs/mt/test.1p3b.pred.vi.txt
outputs/mt/test.1p3b.translation-errors.tsv
outputs/rendered_strong/test/vi/image
outputs/real_images_strong
outputs/worker_strong
```

## Complete Flow

### 1. Setup

Use Python 3.10 or 3.11 on the GPU machine.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python3 -m pip install -r requirements-paddleocr.txt
```

If the server is not CUDA 11.8, install the PaddlePaddle wheel that matches the server CUDA version.

Check the environment:

```bash
sh scripts/check-training-env.sh
```

### 2. Train Translation

Train the English to Vietnamese model:

```bash
sh scripts/train-translation.sh
```

Expected checkpoints:

```text
models/mt-nllb-1p3b-en-vi/best
models/mt-nllb-1p3b-en-vi/last
models/mt-nllb-1p3b-en-vi/metrics.json
```

Use `best` for inference unless visual inspection shows `last` is better.

### 3. Check Translation Quality

Predict the test split:

```bash
OUTPUT=outputs/mt/test.1p3b.pred.vi.txt sh scripts/predict-translation.sh
```

Analyze weak translations:

```bash
PREDICTIONS=outputs/mt/test.1p3b.pred.vi.txt sh scripts/analyze-translation-errors.sh
```

Try decoding variants without retraining:

```bash
sh scripts/predict-translation-variants.sh
```

This creates files such as:

```text
outputs/mt/test.1p3b.beam4.lp1p0.pred.vi.txt
outputs/mt/test.1p3b.beam5.lp1p0.pred.vi.txt
outputs/mt/test.1p3b.beam6.lp1p1.pred.vi.txt
```

Pick the best-looking translation file by checking its `.errors.tsv` and rendered images.

### Quantitative Evaluation Status

The repository already contains partial quantitative results for the strong
pipeline, but it does not yet contain a complete final evaluation table for a
thesis-style report. The currently available numbers are:

| Component | Metric | Value | Source |
| --- | ---: | ---: | --- |
| MT fine-tuned NLLB 1.3B | chrF | 52.95 | `models/mt-nllb-1p3b-en-vi/best/eval.json` |
| MT fine-tuned NLLB 600M | chrF | 52.09 | `models/mt-nllb-en-vi/best/eval.json` |
| PaddleOCR on test split | CER | 20.70% | `outputs/ocr/en/test.pred.en.metrics.json` |
| PaddleOCR on test split | WER | 21.89% | `outputs/ocr/en/test.pred.en.metrics.json` |
| MT test predictions | Samples | 1,500 | `outputs/mt/test.1p3b.pred.vi.txt` |
| OCR test predictions | Samples | 1,500 | `outputs/ocr/en/test.pred.en.metrics.json` |

For the final report, the project should still add a dedicated benchmark table
with the following measurements:

| Evaluation item | Required result | Current status |
| --- | --- | --- |
| BLEU / chrF on the official test set | Corpus-level BLEU and chrF for `outputs/mt/test.1p3b.pred.vi.txt` against the official Vietnamese references | Not yet summarized in README |
| Base NLLB vs fine-tuned model | Direct comparison between the original NLLB checkpoint and the fine-tuned checkpoint on the same test set | Not yet recorded |
| OCR accuracy in the strong pipeline | CER/WER for PaddleOCR PP-OCRv5 when used by `configs/config-pipeline-strong.json` | CER/WER available, but not yet presented as a final table |
| Average latency per image | Mean, p50, and p95 processing time for one image from upload to rendered output | Not yet measured |
| VRAM/RAM usage | Peak GPU memory and system RAM during worker inference | Not yet measured |
| Worker throughput | Images per minute or requests per second with the production worker configuration | Not yet measured |
| End-to-end image quality | Human or rubric-based score for OCR correctness, translation adequacy, text removal, and rendered-text readability | Not yet measured |

This means the current model is usable for an application-oriented graduation
project, but the evaluation section should be completed before the final
submission. The strongest report format is to present MT quality, OCR quality,
runtime cost, and visual end-to-end quality in one consolidated table, then show
several representative success and failure cases.

### 4. Render Benchmark Images

Render translations onto the clean benchmark backgrounds:

```bash
sh scripts/render-translations.sh \
  --split test \
  --translations outputs/mt/test.1p3b.pred.vi.txt
```

Output:

```text
outputs/rendered_strong/test/vi/image
```

This benchmark checks:

```text
ground-truth English subtitle -> MT -> Vietnamese text -> render into clean image
```

### 5. Translate A Real Image

Run the actual application path on an uploaded-style image:

```bash
sh scripts/translate-image.sh --input path/to/english-image.jpg
```

Default outputs:

```text
outputs/real_images_strong/english-image.vi.png
outputs/real_images_strong/english-image.vi.mask.png
outputs/real_images_strong/english-image.vi.png.json
```

The metadata JSON contains detected boxes, OCR text, translations, and merged subtitle groups.

### 6. Start Backend Worker

Start the FastAPI worker:

```bash
sh scripts/run-worker.sh
```

If the worker is public or runs on a separate GPU host, protect it with a shared API key:

```bash
IIMT_WORKER_API_KEY="<shared-secret>" sh scripts/run-worker.sh
```

Default server:

```text
http://0.0.0.0:8081
```

Health check:

```bash
curl http://localhost:8081/health
```

Create an async translation job:

```bash
curl -X POST http://localhost:8081/jobs \
  -H "Authorization: Bearer <shared-secret>" \
  -F "file=@path/to/english-image.jpg"
```

Check job status:

```bash
curl http://localhost:8081/jobs/<job_id> \
  -H "Authorization: Bearer <shared-secret>"
```

Run a synchronous translation request for demos:

```bash
curl -X POST http://localhost:8081/translate \
  -H "Authorization: Bearer <shared-secret>" \
  -F "file=@path/to/english-image.jpg"
```

Successful responses include:

```text
result.output_url
result.mask_url
result.metadata_url
```

Worker outputs are stored under:

```text
outputs/worker_strong/uploads
outputs/worker_strong/results
outputs/worker_strong/jobs
```

## Docker

Build and run the GPU worker:

```bash
docker build -f Dockerfile.worker -t iimt-worker .
docker run --gpus all --rm -p 8081:8081 \
  -e CONFIG=configs/config-pipeline-strong.json \
  -e IIMT_WORKER_API_KEY="<shared-secret>" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  iimt-worker
```

## Final Application Flow

For the production/demo web app, the final worker only needs:

```text
PaddleOCR PP-OCRv5 pretrained OCR
models/mt-nllb-1p3b-en-vi/best
configs/config-pipeline-strong.json
FastAPI worker
```

The only model trained by the project for the production flow is the English-to-Vietnamese translation model.
