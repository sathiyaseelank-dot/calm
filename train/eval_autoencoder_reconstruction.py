#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.configuration_autoencoder import AutoencoderConfig
from models.modeling_autoencoder import Autoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate autoencoder token reconstruction accuracy.")
    parser.add_argument("--ae_path", default="out_ae_semantic")
    parser.add_argument("--file", default="data/train_20.txt")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_blocks", type=int, default=20)
    return parser.parse_args()


def read_records(path):
    file_path = Path(path)
    if file_path.suffix in {".json", ".jsonl"}:
        texts = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                if text:
                    texts.append(text)
        return texts
    return file_path.read_text(encoding="utf-8").splitlines()


def pad_to_multiple(input_ids, multiple, pad_token_id):
    remainder = input_ids.shape[1] % multiple
    if remainder == 0:
        return input_ids
    pad_len = multiple - remainder
    pad = torch.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype)
    return torch.cat([input_ids, pad], dim=1)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.ae_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoencoderConfig.from_pretrained(args.ae_path)
    config.pad_token_id = tokenizer.pad_token_id
    model = Autoencoder.from_pretrained(args.ae_path, config=config)
    model.eval()

    records = read_records(args.file)
    token_chunks = []
    for record in records:
        token_ids = tokenizer(record, return_tensors="pt", add_special_tokens=True)["input_ids"].long()
        eos = torch.tensor([[tokenizer.eos_token_id]], dtype=token_ids.dtype)
        token_ids = torch.cat([token_ids, eos], dim=1)
        token_chunks.append(pad_to_multiple(token_ids, config.patch_size, tokenizer.pad_token_id))
    input_ids = torch.cat(token_chunks, dim=1).long()
    block_size = max(config.patch_size, (args.block_size // config.patch_size) * config.patch_size)

    total = 0
    correct = 0
    examples = []
    with torch.no_grad():
        for block_idx, start in enumerate(range(0, input_ids.shape[1], block_size)):
            if args.max_blocks is not None and block_idx >= args.max_blocks:
                break
            labels = input_ids[:, start : start + block_size]
            labels = pad_to_multiple(labels, config.patch_size, tokenizer.pad_token_id)
            logits = model(input_ids=labels, labels=labels).logits
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != tokenizer.pad_token_id
            total += mask.sum().item()
            correct += ((predictions == labels) & mask).sum().item()
            if len(examples) < 3:
                examples.append(
                    {
                        "input": tokenizer.decode(labels[0][mask[0]], skip_special_tokens=True),
                        "reconstruction": tokenizer.decode(predictions[0][mask[0]], skip_special_tokens=True),
                    }
                )

    token_acc = correct / max(1, total)
    print(f"TOKEN_ACC={token_acc:.6f} correct={correct} total={total}")
    for idx, example in enumerate(examples, start=1):
        print(f"example_{idx}_input={example['input']!r}")
        print(f"example_{idx}_reconstruction={example['reconstruction']!r}")


if __name__ == "__main__":
    main()