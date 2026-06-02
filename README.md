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

Important outputs:

```text
outputs/ocr/en/{train,val,test}/images
outputs/ocr/en/{train,val,test}/labels.tsv
outputs/ocr/en/test.pred.en.txt
models/ocr-trocr-en/best
models/mt-nllb-en-vi/best
outputs/mt/test.pred.vi.txt
outputs/rendered/test/vi/image
```

## A100 Setup

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

## Prepare OCR Data

Create text crops and OCR labels from `en/image`, `en/pos.txt`, and `en/subtitle.txt`:

```bash
sh scripts/prepare-ocr-dataset.sh
```

`scripts/train-ocr.sh` also runs this preparation step automatically.

## Train

Train the English to Vietnamese translation model:

```bash
sh scripts/train-translation.sh
```

Train the English OCR model:

```bash
sh scripts/train-ocr.sh
```

Checkpoints and metrics are saved under:

```text
models/mt-nllb-en-vi
models/ocr-trocr-en
```

## Generate Translated Images

Run OCR on the test crops:

```bash
CHECKPOINT=models/ocr-trocr-en/best SPLIT=test sh scripts/predict-ocr.sh
```

Translate the OCR text to Vietnamese:

```bash
INPUT=outputs/ocr/en/test.pred.en.txt CHECKPOINT=models/mt-nllb-en-vi/best SPLIT=test sh scripts/predict-translation.sh
```

Render the generated Vietnamese text onto clean test backgrounds:

```bash
sh scripts/render-translations.sh --split test --translations outputs/mt/test.pred.vi.txt
```

For an oracle render benchmark with ground-truth Vietnamese subtitles:

```bash
sh scripts/render-translations.sh --split test
```

For an MT-only benchmark that skips OCR and translates ground-truth English subtitles:

```bash
CHECKPOINT=models/mt-nllb-en-vi/best SPLIT=test sh scripts/predict-translation.sh
sh scripts/render-translations.sh --split test --translations outputs/mt/test.pred.vi.txt
```

## Practical Notes

For the graduation project, the trainable parts are OCR and machine translation. The full system still satisfies the IIMT goal because it recognizes text inside images, translates it, and reconstructs a translated image.

For real-world images that do not have clean `background` files, add an inpainting model such as LaMa and use OCR boxes as masks before rendering the Vietnamese text.
