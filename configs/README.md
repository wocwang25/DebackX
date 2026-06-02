# Pipeline Config

The active configuration is:

```text
configs/config-pipeline.json
```

## Sections

`dataset`

Controls the IIMT-style dataset root, split names, language names, and image size.

`ocr`

Controls OCR crop generation. `train_output_dir` receives cropped text regions and `labels.tsv` files.

`translation`

Documents the external machine-translation backend. The repo no longer trains DebackX's visual translation transformer as the main path.

`inpainting`

For benchmark/evaluation on `IIMT30k_Vi`, clean backgrounds are available in the dataset. For real images, use an inpainting model such as LaMa with masks from OCR boxes.

`render`

Controls font, text color, box color, opacity, padding, and output directory for inserting Vietnamese text back into images.
