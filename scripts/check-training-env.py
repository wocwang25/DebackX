import argparse
import importlib
import json
from pathlib import Path


def resolve(config_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def read_lines(path):
    with Path(path).open("r", encoding="utf-8") as text_file:
        return [line.rstrip("\n") for line in text_file]


def count_lines(path):
    path = Path(path)
    if not path.exists():
        return None
    return len(read_lines(path))


def count_numeric_jpgs(path):
    path = Path(path)
    if not path.exists():
        return None
    return len(list(path.glob("*.jpg")))


def format_count(value):
    return "missing" if value is None else str(value)


def check_module(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, str(exc)
    return True, getattr(module, "__version__", "installed")


def check_cuda():
    ok, version = check_module("torch")
    if not ok:
        print(f"[missing] torch: {version}")
        return

    import torch

    print(f"[ok] torch: {version}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[warn] training will run on CPU. Use the A100 machine before real training.")
        return

    print(f"[info] cuda runtime: {torch.version.cuda}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gb = props.total_memory / (1024**3)
        print(f"[ok] gpu {index}: {props.name}, {memory_gb:.1f} GB")
    print(f"[info] bf16 supported: {torch.cuda.is_bf16_supported()}")


def check_dependencies():
    for name in ["transformers", "PIL", "sentencepiece", "sacrebleu", "cv2", "tqdm"]:
        ok, version = check_module(name)
        status = "ok" if ok else "missing"
        print(f"[{status}] {name}: {version}")


def check_dataset(config, config_path):
    root = resolve(config_path, config["dataset"]["root"])
    source_lang = config["dataset"]["source_lang"]
    target_lang = config["dataset"]["target_lang"]
    print(f"[info] dataset root: {root}")
    if not root.exists():
        print("[missing] dataset root does not exist")
        return

    for split in config["dataset"]["splits"]:
        split_root = root / split
        en_root = split_root / source_lang
        vi_root = split_root / target_lang
        counts = {
            "background": count_numeric_jpgs(split_root / "background"),
            "en_image": count_numeric_jpgs(en_root / "image"),
            "vi_image": count_numeric_jpgs(vi_root / "image"),
            "en_subtitle": count_lines(en_root / "subtitle.txt"),
            "vi_subtitle": count_lines(vi_root / "subtitle.txt"),
            "en_pos": count_lines(en_root / "pos.txt"),
            "vi_pos": count_lines(vi_root / "pos.txt"),
        }
        values = list(counts.values())
        if None not in values and len(set(values)) == 1:
            print(f"[ok] {split}: {values[0]} aligned samples")
        else:
            detail = ", ".join(f"{key}={format_count(value)}" for key, value in counts.items())
            print(f"[warn] {split}: count mismatch: {detail}")


def check_ocr_labels(config, config_path):
    output_root = resolve(config_path, config["ocr"]["train_output_dir"])
    label_name = config["ocr"].get("label_file", "labels.tsv")
    for split in config["dataset"]["splits"]:
        label_path = output_root / split / label_name
        if label_path.exists():
            print(f"[ok] ocr labels {split}: {len(read_lines(label_path))} rows")
        else:
            print(f"[warn] ocr labels {split}: missing. Run `sh scripts/prepare-ocr-dataset.sh`.")


def check_render(config):
    font_path = Path(config["render"]["font_path"])
    if font_path.exists():
        print(f"[ok] render font: {font_path}")
    else:
        print(f"[warn] render font missing: {font_path}")


def main():
    parser = argparse.ArgumentParser(description="Check whether the IIMT training workspace is ready.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    check_cuda()
    check_dependencies()
    check_dataset(config, config_path)
    check_ocr_labels(config, config_path)
    check_render(config)


if __name__ == "__main__":
    main()
