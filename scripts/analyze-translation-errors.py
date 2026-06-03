import argparse
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


def safe_tsv(text):
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def edit_distance(source, target):
    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, 1):
        current = [i]
        for j, target_item in enumerate(target, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_item != target_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def fallback_score(prediction, reference):
    distance = edit_distance(prediction, reference)
    return max(0.0, 100.0 * (1.0 - distance / max(1, len(reference))))


def sentence_score(prediction, reference):
    try:
        import sacrebleu

        return sacrebleu.sentence_chrf(prediction, [reference]).score
    except Exception:
        return fallback_score(prediction, reference)


def main():
    parser = argparse.ArgumentParser(description="Rank translation predictions by sentence-level chrF.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", default="outputs/mt/test.pred.vi.txt")
    parser.add_argument("--output", default="outputs/mt/test.translation-errors.tsv")
    parser.add_argument("--top", type=int, default=200)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    root = resolve(config_path, config["dataset"]["root"])
    source_path = root / args.split / config["dataset"]["source_lang"] / "subtitle.txt"
    reference_path = root / args.split / config["dataset"]["target_lang"] / "subtitle.txt"
    prediction_path = Path(args.predictions)
    if not prediction_path.is_absolute():
        prediction_path = (Path.cwd() / prediction_path).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    sources = read_lines(source_path)
    references = read_lines(reference_path)
    predictions = read_lines(prediction_path)
    if not (len(sources) == len(references) == len(predictions)):
        raise ValueError(
            "source/reference/prediction line count mismatch: "
            f"{len(sources)}, {len(references)}, {len(predictions)}"
        )

    rows = []
    for index, (source, reference, prediction) in enumerate(zip(sources, references, predictions), 1):
        rows.append((sentence_score(prediction, reference), index, source, prediction, reference))
    rows.sort(key=lambda item: item[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("rank\tindex\tchrf\tsource\tprediction\treference\n")
        for rank, (score, index, source, prediction, reference) in enumerate(rows[: args.top], 1):
            output_file.write(
                f"{rank}\t{index}\t{score:.2f}\t"
                f"{safe_tsv(source)}\t{safe_tsv(prediction)}\t{safe_tsv(reference)}\n"
            )

    average = sum(row[0] for row in rows) / max(1, len(rows))
    print(f"average sentence chrF: {average:.2f}")
    print(f"wrote worst {min(args.top, len(rows))} examples -> {output_path}")


if __name__ == "__main__":
    main()
