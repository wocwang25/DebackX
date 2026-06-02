import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parent))
from common import cer, precision_dtype, resolve_from_config, save_json, set_seed, use_amp, wer, load_config


class OcrDataset(Dataset):
    def __init__(self, labels_path, processor, max_target_length):
        self.root = Path(labels_path).parent
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples = []
        with Path(labels_path).open("r", encoding="utf-8") as label_file:
            for line in label_file:
                image_rel, text = line.rstrip("\n").split("\t", 1)
                self.samples.append((self.root / image_rel, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, text = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "text": text}


def collate(batch):
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
        "texts": [sample["text"] for sample in batch],
    }


@torch.no_grad()
def evaluate(model, processor, dataloader, device, precision, max_target_length, max_samples):
    model.eval()
    predictions = []
    references = []
    seen = 0
    amp_dtype = precision_dtype(precision)
    for batch in tqdm(dataloader, desc="eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp(precision)):
            generated = model.generate(pixel_values, max_length=max_target_length, num_beams=4)
        predictions.extend(processor.batch_decode(generated, skip_special_tokens=True))
        references.extend(batch["texts"])
        seen += len(batch["texts"])
        if max_samples and seen >= max_samples:
            break
    predictions = predictions[:max_samples] if max_samples else predictions
    references = references[:max_samples] if max_samples else references
    return {
        "cer": cer(predictions, references),
        "wer": wer(predictions, references),
        "predictions": predictions[:20],
        "references": references[:20],
    }


def train(config_path, seed):
    set_seed(seed)
    config, config_file = load_config(config_path)
    ocr_config = config["ocr"]
    output_dir = resolve_from_config(config_file, ocr_config["checkpoint_dir"])
    train_root = resolve_from_config(config_file, ocr_config["train_output_dir"])
    label_file = ocr_config.get("label_file", "labels.tsv")

    train_labels = train_root / "train" / label_file
    valid_labels = train_root / "val" / label_file
    if not train_labels.exists() or not valid_labels.exists():
        raise FileNotFoundError(
            "OCR labels not found. Run `sh scripts/prepare-ocr-dataset.sh` before training."
        )

    processor = TrOCRProcessor.from_pretrained(ocr_config["pretrained_model"])
    model = VisionEncoderDecoderModel.from_pretrained(ocr_config["pretrained_model"])
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = OcrDataset(train_labels, processor, ocr_config["max_target_length"])
    valid_dataset = OcrDataset(valid_labels, processor, ocr_config["max_target_length"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=ocr_config["batch_size"],
        shuffle=True,
        num_workers=ocr_config["num_workers"],
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=ocr_config["eval_batch_size"],
        shuffle=False,
        num_workers=ocr_config["num_workers"],
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ocr_config["learning_rate"],
        weight_decay=ocr_config["weight_decay"],
    )
    total_train_steps = len(train_loader) * ocr_config["epochs"]
    warmup_steps = int(total_train_steps * ocr_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)
    precision = ocr_config.get("precision", "bf16")
    amp_dtype = precision_dtype(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and torch.cuda.is_available())

    output_dir.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    history = []
    global_step = 0
    for epoch in range(1, ocr_config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"ocr epoch {epoch}")
        for step, batch in enumerate(progress, 1):
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp(precision)):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / max(1, step))

        metrics = evaluate(
            model,
            processor,
            valid_loader,
            device,
            precision,
            ocr_config["max_target_length"],
            ocr_config.get("eval_samples", 0),
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(1, len(train_loader))
        history.append(metrics)
        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
            save_json(output_dir / "best" / "eval.json", metrics)
        model.save_pretrained(output_dir / "last")
        processor.save_pretrained(output_dir / "last")
        save_json(output_dir / "metrics.json", {"history": history, "best_cer": best_cer})
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} cer={metrics['cer']:.4f} wer={metrics['wer']:.4f}")

    save_json(output_dir / "metrics.json", {"history": history, "best_cer": best_cer})


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on IIMT30k English text crops.")
    parser.add_argument("--config", default="configs/config-pipeline.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.config, args.seed)


if __name__ == "__main__":
    main()
