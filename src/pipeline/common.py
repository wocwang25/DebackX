import ast
import json
from pathlib import Path


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config, config_path


def resolve_from_config(config_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def dataset_root(config, config_path):
    return resolve_from_config(config_path, config["dataset"]["root"])


def output_path(config_path, raw_path):
    return resolve_from_config(config_path, raw_path)


def read_lines(path):
    with Path(path).open("r", encoding="utf-8") as text_file:
        return [line.rstrip("\n") for line in text_file]


def parse_box(line):
    x1, y1, x2, y2 = ast.literal_eval(line.strip())
    return float(x1), float(y1), float(x2), float(y2)


def numeric_jpgs(path):
    return sorted(Path(path).glob("*.jpg"), key=lambda item: int(item.stem))


def clamp_box(box, width, height, padding=0):
    x1, y1, x2, y2 = box
    x1 = max(0, int(round(x1)) - padding)
    y1 = max(0, int(round(y1)) - padding)
    x2 = min(width, int(round(x2)) + padding)
    y2 = min(height, int(round(y2)) + padding)
    return x1, y1, x2, y2
