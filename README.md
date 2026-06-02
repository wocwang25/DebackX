# Image Text Translation Pipeline

This workspace has been pivoted away from end-to-end DebackX training.

The new direction is a modular pipeline:

```text
image with English text
-> detect/crop text region
-> OCR English text
-> translate English to Vietnamese with an external MT model
-> remove old text / use clean background
-> render Vietnamese text back into the image
```

The current dataset root is `IIMT30k_Vi/Arial`.

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

Vietnamese subtitles have been cleaned and re-tokenized with:

```text
IIMT30k_Vi/Arial/spm/vi.model
IIMT30k_Vi/Arial/spm/vi.vocab
```

Backups are kept with `.bak` suffix.

## Configuration

Main config:

```text
configs/config-pipeline.json
```

It controls dataset paths, OCR crop output, translation backend notes, inpainting mode, and render style.

## Prepare OCR Data

Create text crops and OCR labels from `en/image`, `en/pos.txt`, and `en/subtitle.txt`:

```bash
sh scripts/prepare-ocr-dataset.sh
```

Outputs:

```text
outputs/ocr/en/{train,val,test}/images
outputs/ocr/en/{train,val,test}/labels.tsv
```

These files can be used to fine-tune an OCR recognizer, or you can start with a pretrained OCR engine such as PaddleOCR or EasyOCR.

## Render Benchmark

Render Vietnamese subtitles back onto clean backgrounds. This is an oracle benchmark for the final "insert translated text" stage:

```bash
sh scripts/render-translations.sh --split test
```

Outputs:

```text
outputs/rendered/test/vi/image
```

To render model-generated translations instead of ground truth Vietnamese subtitles:

```bash
sh scripts/render-translations.sh --split test --translations path/to/predicted.vi.txt
```

The translation file must contain one line per image in numeric filename order.

## Practical Training Order

1. Prepare OCR crops with `scripts/prepare-ocr-dataset.sh`.
2. Fine-tune or choose an OCR backend.
3. Use an external MT model for `en -> vi`, optionally fine-tuned on `IIMT30k_Vi/Arial/train/{en,vi}/subtitle.txt`.
4. For benchmark images, use `background` as the clean canvas.
5. For real images, use OCR boxes to create masks and inpaint with a model such as LaMa.
6. Render Vietnamese text with `scripts/render-translations.sh`.

The old DebackX model code is left in `src/` only as reference. The active project flow is controlled by `configs/config-pipeline.json`.
