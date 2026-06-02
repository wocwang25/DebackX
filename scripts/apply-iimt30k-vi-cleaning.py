#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import sentencepiece as spm


SPLITS = ("train", "val", "test")


def backup_once(path):
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def read_lines(path):
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def apply_cleaned_subtitles(root, cleaned_root):
    for split in SPLITS:
        source = cleaned_root / split / "vi" / "subtitle.txt"
        target = root / split / "vi" / "subtitle.txt"
        if not source.exists():
            raise FileNotFoundError(source)
        if not target.exists():
            raise FileNotFoundError(target)

        cleaned_lines = read_lines(source)
        if len(cleaned_lines) != len(read_lines(target)):
            raise ValueError(f"line count mismatch for {split}: {source} -> {target}")

        backup_once(target)
        write_lines(target, cleaned_lines)


def train_sentencepiece(root, vocab_size):
    spm_dir = root / "spm"
    spm_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = spm_dir / "vi"
    model_path = spm_dir / "vi.model"

    if model_path.exists():
        backup_once(model_path)
    vocab_path = spm_dir / "vi.vocab"
    if vocab_path.exists():
        backup_once(vocab_path)

    spm.SentencePieceTrainer.Train(
        input=str(root / "train" / "vi" / "subtitle.txt"),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        pad_piece="<pad>",
        hard_vocab_limit=False,
    )
    return model_path


def regenerate_vi_tokens(root, model_path):
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    for split in SPLITS:
        subtitle_path = root / split / "vi" / "subtitle.txt"
        tok_path = root / split / "vi" / "subtitle.tok.txt"
        id_path = root / split / "vi" / "subtitle.tok.id.txt"
        backup_once(tok_path)
        backup_once(id_path)

        piece_lines = []
        id_lines = []
        for line in read_lines(subtitle_path):
            pieces = processor.encode(line, out_type=str, add_bos=True, add_eos=True)
            ids = processor.encode(line, out_type=int, add_bos=True, add_eos=True)
            piece_lines.append(" ".join(pieces))
            id_lines.append(" ".join(str(token_id) for token_id in ids))

        write_lines(tok_path, piece_lines)
        write_lines(id_path, id_lines)


def main():
    parser = argparse.ArgumentParser(description="Apply cleaned Vietnamese subtitles and regenerate SentencePiece tokens.")
    parser.add_argument("--root", default="IIMT30k_Vi/Arial")
    parser.add_argument("--cleaned-root", default="cleaning/iimt30k_vi/cleaned_text")
    parser.add_argument("--vocab-size", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.root)
    cleaned_root = Path(args.cleaned_root)
    apply_cleaned_subtitles(root, cleaned_root)
    model_path = train_sentencepiece(root, args.vocab_size)
    regenerate_vi_tokens(root, model_path)
    print(f"applied cleaned subtitles under {root}")
    print(f"sentencepiece model: {model_path}")
    print("regenerated vi/subtitle.tok.txt and vi/subtitle.tok.id.txt for train/val/test")


if __name__ == "__main__":
    main()
