import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parent))
from common import chrf_score, precision_dtype, resolve_from_config, save_json, set_seed, total_steps, use_amp, load_config


def configure_generation(model, tokenizer, mt_config):
    model.generation_config.max_length = mt_config["max_target_length"]
    model.generation_config.num_beams = mt_config["num_beams"]
    model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(mt_config["target_code"])
    model.generation_config.length_penalty = mt_config.get("length_penalty", 1.0)
    model.generation_config.repetition_penalty = mt_config.get("repetition_penalty", 1.0)
    model.generation_config.early_stopping = mt_config.get("early_stopping", True)
    no_repeat = mt_config.get("no_repeat_ngram_size", 0)
    if no_repeat:
        model.generation_config.no_repeat_ngram_size = no_repeat

    default_generation_config = GenerationConfig()
    for key in list(model.config._get_non_default_generation_parameters()):
        setattr(model.config, key, getattr(default_generation_config, key, None))


class ParallelTextDataset(Dataset):
    def __init__(self, source_path, target_path):
        with Path(source_path).open("r", encoding="utf-8") as source_file:
            self.sources = [line.rstrip("\n") for line in source_file]
        with Path(target_path).open("r", encoding="utf-8") as target_file:
            self.targets = [line.rstrip("\n") for line in target_file]
        if len(self.sources) != len(self.targets):
            raise ValueError(f"source/target line count mismatch: {source_path}, {target_path}")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return self.sources[index], self.targets[index]


class MtCollator:
    def __init__(self, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch):
        sources, targets = zip(*batch)
        encoded = self.tokenizer(
            list(sources),
            max_length=self.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=list(targets),
            max_length=self.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        encoded["targets"] = list(targets)
        return encoded


@torch.no_grad()
def evaluate(model, tokenizer, dataloader, device, mt_config):
    model.eval()
    predictions = []
    references = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(mt_config["target_code"])
    seen = 0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        targets = batch.pop("targets")
        batch = {key: value.to(device) for key, value in batch.items()}
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=mt_config["max_target_length"],
            num_beams=mt_config["num_beams"],
            forced_bos_token_id=forced_bos_token_id,
        )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        references.extend(targets)
        seen += len(targets)
        if mt_config.get("eval_samples", 0) and seen >= mt_config["eval_samples"]:
            break
    if mt_config.get("eval_samples", 0):
        predictions = predictions[: mt_config["eval_samples"]]
        references = references[: mt_config["eval_samples"]]
    return {
        "chrf": chrf_score(predictions, references),
        "predictions": predictions[:20],
        "references": references[:20],
    }


def train(config_path, seed):
    set_seed(seed)
    config, config_file = load_config(config_path)
    mt_config = config["translation"]
    output_dir = resolve_from_config(config_file, mt_config["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(mt_config["pretrained_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(mt_config["pretrained_model"])
    tokenizer.src_lang = mt_config["source_code"]
    tokenizer.tgt_lang = mt_config["target_code"]
    configure_generation(model, tokenizer, mt_config)

    train_dataset = ParallelTextDataset(
        resolve_from_config(config_file, mt_config["train_source"]),
        resolve_from_config(config_file, mt_config["train_target"]),
    )
    valid_dataset = ParallelTextDataset(
        resolve_from_config(config_file, mt_config["valid_source"]),
        resolve_from_config(config_file, mt_config["valid_target"]),
    )
    collator = MtCollator(tokenizer, mt_config["max_source_length"], mt_config["max_target_length"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=mt_config["batch_size"],
        shuffle=True,
        num_workers=mt_config["num_workers"],
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=mt_config["eval_batch_size"],
        shuffle=False,
        num_workers=mt_config["num_workers"],
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mt_config["learning_rate"],
        weight_decay=mt_config["weight_decay"],
    )
    grad_accum = mt_config.get("gradient_accumulation_steps", 1)
    total_train_steps = total_steps(len(train_dataset), mt_config["batch_size"], grad_accum, mt_config["epochs"])
    warmup_steps = int(total_train_steps * mt_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)
    precision = mt_config.get("precision", "bf16")
    amp_dtype = precision_dtype(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and torch.cuda.is_available())

    best_chrf = -1.0
    history = []
    global_step = 0
    for epoch in range(1, mt_config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"mt epoch {epoch}")
        for step, batch in enumerate(progress, 1):
            batch.pop("targets")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp(precision)):
                loss = model(**batch).loss / grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(loss=running_loss / max(1, step))

        metrics = evaluate(model, tokenizer, valid_loader, device, mt_config)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(1, len(train_loader))
        history.append(metrics)
        if metrics["chrf"] > best_chrf:
            best_chrf = metrics["chrf"]
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            save_json(output_dir / "best" / "eval.json", metrics)
        model.save_pretrained(output_dir / "last")
        tokenizer.save_pretrained(output_dir / "last")
        save_json(output_dir / "metrics.json", {"history": history, "best_chrf": best_chrf})
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} chrf={metrics['chrf']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB for English to Vietnamese subtitles.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.config, args.seed)


if __name__ == "__main__":
    main()
