import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import clamp_box, dataset_root, load_config, numeric_jpgs, output_path, parse_box, read_lines


def text_size(draw, text, font):
    x1, y1, x2, y2 = draw.textbbox((0, 0), text, font=font)
    return x2 - x1, y2 - y1


def wrap_text(draw, text, font, max_width):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        if text_size(draw, word, font)[0] <= max_width:
            current = word
            continue

        piece = ""
        for char in word:
            candidate = piece + char
            if text_size(draw, candidate, font)[0] <= max_width:
                piece = candidate
            else:
                if piece:
                    lines.append(piece)
                piece = char
        current = piece

    if current:
        lines.append(current)
    return lines


def fit_text(draw, text, box_width, box_height, render_config):
    font_path = render_config["font_path"]
    min_size = int(render_config["min_font_size"])
    max_size = int(render_config["max_font_size"])
    hpad = int(render_config["horizontal_padding"])
    vpad = int(render_config["vertical_padding"])
    max_text_width = max(1, box_width - 2 * hpad)
    max_text_height = max(1, box_height - 2 * vpad)

    for font_size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        lines = wrap_text(draw, text, font, max_text_width)
        _, line_height = text_size(draw, "Ay", font)
        total_height = line_height * len(lines)
        widest = max(text_size(draw, line, font)[0] for line in lines)
        if widest <= max_text_width and total_height <= max_text_height:
            return font, lines, line_height

    font = ImageFont.truetype(font_path, min_size, encoding="utf-8")
    return font, wrap_text(draw, text, font, max_text_width), text_size(draw, "Ay", font)[1]


def draw_translated_text(image, box, text, render_config):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    box_color = tuple(render_config["box_color"])
    alpha = int(float(render_config["box_alpha"]) * 255)
    overlay_draw.rectangle((x1, y1, x2, y2), fill=box_color + (alpha,))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    font, lines, line_height = fit_text(draw, text, box_width, box_height, render_config)
    text_color = tuple(render_config["text_color"])
    total_height = line_height * len(lines)
    y = y1 + max(0, (box_height - total_height) // 2)
    for line in lines:
        width, _ = text_size(draw, line, font)
        x = x1 + max(0, (box_width - width) // 2)
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height
    return image


def translations_for_split(root, split, target_lang, translations_file):
    if translations_file:
        return read_lines(translations_file)
    return read_lines(root / split / target_lang / "subtitle.txt")


def render_split(config, config_path, split, translations_file=None):
    root = dataset_root(config, config_path)
    target_lang = config["dataset"]["target_lang"]
    render_config = config["render"]
    output_root = output_path(config_path, render_config["output_dir"])

    background_dir = root / split / "background"
    pos_path = root / split / target_lang / "pos.txt"
    backgrounds = numeric_jpgs(background_dir)
    boxes = [parse_box(line) for line in read_lines(pos_path)]
    translations = translations_for_split(root, split, target_lang, translations_file)

    if not (len(backgrounds) == len(boxes) == len(translations)):
        raise ValueError(
            f"{split}: background/pos/translation counts differ: {len(backgrounds)}, {len(boxes)}, {len(translations)}"
        )

    output_dir = output_root / split / target_lang / "image"
    output_dir.mkdir(parents=True, exist_ok=True)

    for background_path, box, text in zip(backgrounds, boxes, translations):
        with Image.open(background_path).convert("RGB") as image:
            draw_box = clamp_box(box, image.width, image.height, padding=0)
            rendered = draw_translated_text(image, draw_box, text, render_config)
            rendered.save(output_dir / background_path.name)

    print(f"{split}: rendered {len(backgrounds)} images -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render translated text back into background images.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--translations", default=None, help="Optional one-line-per-image translated text file.")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    render_split(config, config_path, args.split, args.translations)


if __name__ == "__main__":
    main()
