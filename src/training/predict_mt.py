import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
from common import resolve_from_config, load_config


class TextDataset(Dataset):
    def __init__(self, path):
        with Path(path).open("r", encoding="utf-8") as text_file:
            self.lines = [line.rstrip("\n") for line in text_file]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]


def collate_text(batch):
    return list(batch)


@torch.no_grad()
def predict(config_path, checkpoint, split, output, input_path):
    config, config_file = load_config(config_path)
    mt_config = config["translation"]
    if input_path:
        source_path = Path(input_path)
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
    else:
        root = resolve_from_config(config_file, config["dataset"]["root"])
        source_path = root / split / config["dataset"]["source_lang"] / "subtitle.txt"
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer.src_lang = mt_config["source_code"]
    tokenizer.tgt_lang = mt_config["target_code"]
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(mt_config["target_code"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = TextDataset(source_path)
    dataloader = DataLoader(dataset, batch_size=mt_config["eval_batch_size"], shuffle=False, collate_fn=collate_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as out_file:
        for lines in tqdm(dataloader, desc=f"predict {split}"):
            encoded = tokenizer(
                lines,
                max_length=mt_config["max_source_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_length=mt_config["max_target_length"],
                num_beams=mt_config["num_beams"],
                forced_bos_token_id=forced_bos_token_id,
            )
            for prediction in tokenizer.batch_decode(generated, skip_special_tokens=True):
                out_file.write(prediction + "\n")
    print(f"wrote translations -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Vietnamese translations with a fine-tuned MT model.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--checkpoint", default="models/mt-nllb-en-vi/best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/mt/test.pred.vi.txt")
    parser.add_argument("--input", default=None, help="Optional English text file, one line per image.")
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.split, args.output, args.input)


if __name__ == "__main__":
    main()
