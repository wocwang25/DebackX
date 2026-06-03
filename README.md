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

Fine-tune TrOCR on cropped English text regions from `IIMT30k_Vi/Arial/{train,val}/en`.

`Translation`

Fine-tune NLLB for English to Vietnamese using:

```text
IIMT30k_Vi/Arial/train/en/subtitle.txt
IIMT30k_Vi/Arial/train/vi/subtitle.txt
IIMT30k_Vi/Arial/val/en/subtitle.txt
IIMT30k_Vi/Arial/val/vi/subtitle.txt
```

`Rendering`

Rendering text back into the image is not trained. It is a deterministic post-processing step controlled by `configs/config-pipeline.json`.

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

Main config:

```text
configs/config-pipeline.json
```

Strong pretrained preset:

```text
configs/config-pipeline-strong.json
```

The strong preset uses:

```text
Translation: facebook/nllb-200-1.3B
OCR:         microsoft/trocr-large-printed
```

Important outputs:

```text
outputs/ocr/en/{train,val,test}/images
outputs/ocr/en/{train,val,test}/labels.tsv
outputs/ocr/en/test.pred.en.txt
models/ocr-trocr-en/best
models/mt-nllb-en-vi/best
outputs/mt/test.pred.vi.txt
outputs/mt/test.from-ocr.pred.vi.txt
outputs/rendered/test/vi/image
outputs/real_images
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

Check whether the machine and dataset are ready:

```bash
sh scripts/check-training-env.sh
```

The config is already set to use `bf16`, which is suitable for A100.

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
sh scripts/train-translation.sh
```

Outputs:

```text
models/mt-nllb-en-vi/best
models/mt-nllb-en-vi/last
models/mt-nllb-en-vi/metrics.json
```

Use `best` for inference unless visual inspection shows `last` is better.

To retrain translation with the stronger NLLB 1.3B preset:

```bash
CONFIG=configs/config-pipeline-strong.json sh scripts/train-translation.sh
```

Strong translation outputs:

```text
models/mt-nllb-1p3b-en-vi/best
models/mt-nllb-1p3b-en-vi/last
models/mt-nllb-1p3b-en-vi/metrics.json
```

### 4. Test Translation And Render Benchmark

Generate Vietnamese translations for the test split using ground-truth English subtitles:

```bash
CHECKPOINT=models/mt-nllb-en-vi/best SPLIT=test sh scripts/predict-translation.sh
```

Output:

```text
outputs/mt/test.pred.vi.txt
```

Render those translations onto the clean benchmark backgrounds:

```bash
sh scripts/render-translations.sh --split test --translations outputs/mt/test.pred.vi.txt
```

Output:

```text
outputs/rendered/test/vi/image
```

This benchmark checks:

```text
ground-truth English text -> MT model -> Vietnamese text -> render into image
```

It intentionally skips OCR so translation and rendering quality can be inspected first.

For an oracle render benchmark with ground-truth Vietnamese subtitles:

```bash
sh scripts/render-translations.sh --split test
```

### 5. Train OCR

Train the English OCR model after OCR crops are prepared:

```bash
sh scripts/train-ocr.sh
```

Outputs:

```text
models/ocr-trocr-en/best
models/ocr-trocr-en/last
models/ocr-trocr-en/metrics.json
```

Use `best` for inference unless visual inspection shows `last` is better.

To retrain OCR with the stronger TrOCR large preset:

```bash
CONFIG=configs/config-pipeline-strong.json sh scripts/train-ocr.sh
```

Strong OCR outputs:

```text
models/ocr-trocr-large-en/best
models/ocr-trocr-large-en/last
models/ocr-trocr-large-en/metrics.json
```

### 6. Run Full Dataset Pipeline

Run OCR on the test crops:

```bash
CHECKPOINT=models/ocr-trocr-en/best SPLIT=test sh scripts/predict-ocr.sh
```

Translate the OCR text to Vietnamese:

```bash
INPUT=outputs/ocr/en/test.pred.en.txt OUTPUT=outputs/mt/test.from-ocr.pred.vi.txt CHECKPOINT=models/mt-nllb-en-vi/best SPLIT=test sh scripts/predict-translation.sh
```

Output:

```text
outputs/mt/test.from-ocr.pred.vi.txt
```

Render the generated Vietnamese text onto clean test backgrounds:

```bash
sh scripts/render-translations.sh --split test --translations outputs/mt/test.from-ocr.pred.vi.txt
```

Output:

```text
outputs/rendered/test/vi/image
```

This full dataset pipeline checks:

```text
English image crop -> OCR -> MT -> render Vietnamese into clean background
```

### 7. Translate A Real Image With CLI

After both checkpoints exist, run the deployable worker path on a user image:

```bash
sh scripts/translate-image.sh --input path/to/english-image.jpg
```

Default outputs:

```text
outputs/real_images/english-image.vi.png       # final translated image
outputs/real_images/english-image.vi.mask.png  # text-removal mask
outputs/real_images/english-image.vi.png.json  # boxes, OCR text, translations
```

This path adds the two parts that the dataset benchmark does not need:

```text
EasyOCR text detection -> finds text boxes in the uploaded image
OpenCV inpainting -> removes the original English text before rendering Vietnamese
```

You can choose the output path manually:

```bash
sh scripts/translate-image.sh --input path/to/english-image.jpg --output outputs/real_images/result.png
```

### 8. Start Backend Worker

After training, start the FastAPI worker:

```bash
sh scripts/run-worker.sh
```

Default server:

```text
http://0.0.0.0:8000
```

The worker loads EasyOCR, TrOCR, and NLLB once at startup and keeps them in memory. If you want to start the API before checkpoints exist, set `worker.load_models_on_startup` to `false` in `configs/config-pipeline.json`; the first translation job will load the models lazily.

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
outputs/worker/uploads
outputs/worker/results
outputs/worker/jobs
```

For Docker GPU deployment:

```bash
docker build -f Dockerfile.worker -t iimt-worker .
docker run --gpus all --rm -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  iimt-worker
```

## Practical Notes

For the graduation project, the trainable parts are OCR and machine translation. The full system still satisfies the IIMT goal because it recognizes text inside images, translates it, and reconstructs a translated image.

For real-world images that do not have clean `background` files, `scripts/translate-image.sh` and the FastAPI worker use OCR boxes as masks and OpenCV inpainting as the baseline text-removal method. A stronger inpainting model such as LaMa can replace this later for better visual quality.

Recommended demo order:

```text
1. Show translation metrics and test.pred.vi.txt examples.
2. Show rendered benchmark images from outputs/rendered/test/vi/image.
3. Show full OCR -> MT -> render examples after OCR training.
4. Show backend worker API if the web app integration is needed.
```
