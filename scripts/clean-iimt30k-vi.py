#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
import unicodedata
from pathlib import Path


SPLITS = ("train", "val", "test")

VI_TYPO_RULES = [
    (re.compile(r"(?<!\w)đẵ(?!\w)"), "đã", "typo:đẵ->đã"),
    (re.compile(r"(?<!\w)Đẵ(?!\w)"), "Đã", "typo:Đẵ->Đã"),
    (re.compile(r"(?<!\w)sé(?!\w)"), "sẽ", "typo:sé->sẽ"),
    (re.compile(r"(?<!\w)Sé(?!\w)"), "Sẽ", "typo:Sé->Sẽ"),
    (re.compile(r"(?<!\w)phia(?!\w)"), "phía", "typo:phia->phía"),
    (re.compile(r"(?<!\w)Phia(?!\w)"), "Phía", "typo:Phia->Phía"),
    (re.compile(r"(?<!\w)chủ sỡ hữu(?!\w)"), "chủ sở hữu", "typo:chủ sỡ hữu->chủ sở hữu"),
    (re.compile(r"(?<!\w)Chủ sỡ hữu(?!\w)"), "Chủ sở hữu", "typo:Chủ sỡ hữu->Chủ sở hữu"),
    (re.compile(r"(?<!\w)sững sỡ(?!\w)"), "sững sờ", "typo:sững sỡ->sững sờ"),
    (re.compile(r"(?<!\w)Sững sỡ(?!\w)"), "Sững sờ", "typo:Sững sỡ->Sững sờ"),
    (re.compile(r"(?<!\w)khác biết(?!\w)"), "khác biệt", "typo:khác biết->khác biệt"),
    (re.compile(r"(?<!\w)Khác biết(?!\w)"), "Khác biệt", "typo:Khác biết->Khác biệt"),
    (re.compile(r"(?<!\w)quan trong(?= như thế nào| thế nào| đối với| là|\\b)"), "quan trọng", "typo:quan trong->quan trọng"),
    (re.compile(r"(?<!\w)Quan trong(?= như thế nào| thế nào| đối với| là|\\b)"), "Quan trọng", "typo:Quan trong->Quan trọng"),
    (re.compile(r"(?<!\w)(rất|điều) quan trong(?!\w)"), r"\1 quan trọng", "typo:quan trong->quan trọng"),
]


VI_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
EN_STOPWORD_RE = re.compile(r"\b(the|and|you|that|this|with|from|have|for|not|are|was|were|will|can|would|should)\b", re.IGNORECASE)


def normalize_text(text):
    original = text.rstrip("\n")
    text = unicodedata.normalize("NFC", original)
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    text = re.sub(r"\s+/", "/", text)
    text = re.sub(r"/\s+", "/", text)
    text = re.sub(r"\s+%", "%", text)
    return text


def apply_vi_rules(text):
    notes = []
    cleaned = text
    for pattern, replacement, note in VI_TYPO_RULES:
        cleaned, count = pattern.subn(replacement, cleaned)
        if count:
            notes.append(f"{note} x{count}")
    return cleaned, notes


def flag_vi_line(text):
    flags = []
    if "  " in text:
        flags.append("double-space")
    if re.search(r"\s+[,.!?;:%]", text):
        flags.append("space-before-punctuation")
    if re.search(r"[A-Za-z]{4,}", text) and not VI_DIACRITIC_RE.search(text):
        flags.append("no-vietnamese-diacritics")
    if len(EN_STOPWORD_RE.findall(text)) >= 3 and not VI_DIACRITIC_RE.search(text):
        flags.append("maybe-untranslated-en")
    if re.search(r"[đĐ]ẵ|(?<!\w)(sé|phia|khác biết|chủ sỡ hữu|sững sỡ)(?!\w)", text):
        flags.append("known-typo")
    return flags


def read_lines(path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def process_split(root, out_dir, split, apply):
    src = root / split / "vi" / "subtitle.txt"
    if not src.exists():
        raise FileNotFoundError(src)

    original_lines = read_lines(src)
    cleaned_lines = []
    rows = []
    auto_changed = 0
    flagged = 0

    for line_no, raw in enumerate(original_lines, start=1):
        normalized = normalize_text(raw)
        cleaned, notes = apply_vi_rules(normalized)
        flags = flag_vi_line(cleaned)
        changed = cleaned != raw
        if changed:
            auto_changed += 1
        if flags:
            flagged += 1
        cleaned_lines.append(cleaned)
        if changed or flags:
            rows.append(
                {
                    "split": split,
                    "line": line_no,
                    "auto_changed": int(changed),
                    "notes": "; ".join(notes),
                    "flags": "; ".join(flags),
                    "before": raw,
                    "after": cleaned,
                }
            )

    cleaned_path = out_dir / "cleaned_text" / split / "vi" / "subtitle.txt"
    report_path = out_dir / "reports" / f"{split}.vi.cleaning.tsv"
    write_lines(cleaned_path, cleaned_lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("split", "line", "auto_changed", "notes", "flags", "before", "after"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    if apply:
        backup = src.with_suffix(".txt.bak")
        if not backup.exists():
            shutil.copy2(src, backup)
        write_lines(src, cleaned_lines)

    return {
        "split": split,
        "lines": len(original_lines),
        "auto_changed": auto_changed,
        "flagged": flagged,
        "cleaned_path": str(cleaned_path),
        "report_path": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalize and audit Vietnamese subtitles for IIMT30k_Vi without touching image files."
    )
    parser.add_argument("--root", default="IIMT30k_Vi/Arial", help="Dataset font root containing train/val/test.")
    parser.add_argument("--out-dir", default="cleaning/iimt30k_vi", help="Where to write cleaned text and reports.")
    parser.add_argument("--apply", action="store_true", help="Write cleaned subtitle.txt back in place and keep .bak files.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    summaries = [process_split(root, out_dir, split, args.apply) for split in SPLITS]

    summary_path = out_dir / "reports" / "summary.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("split", "lines", "auto_changed", "flagged", "cleaned_path", "report_path"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(
            "{split}: lines={lines}, auto_changed={auto_changed}, flagged={flagged}, report={report_path}".format(
                **summary
            )
        )
    print(f"summary={summary_path}")
    if args.apply:
        print("Applied cleaned subtitle.txt in place. Backups use .txt.bak suffix.")
    else:
        print("Dry run only. Original dataset was not modified.")


if __name__ == "__main__":
    main()
