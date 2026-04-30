#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from models.configuration_autoencoder import AutoencoderConfig
from models.modeling_autoencoder import Autoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_path", default="out_ae_semantic")
    parser.add_argument("--file", default="data/train_20.txt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoencoderConfig.from_pretrained(args.ae_path)
    config.pad_token_id = tokenizer.pad_token_id
    model = Autoencoder.from_pretrained(args.ae_path, config=config)
    model.eval()

    texts = []
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)

    total = 0
    correct = 0
    examples = []
    patch_size = config.patch_size

    with torch.no_grad():
        for text in texts[:10]:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            if len(encoded[0]) < patch_size:
                continue

            num_patches = len(encoded[0]) // patch_size
            input_patches = encoded[0, :num_patches * patch_size].reshape(-1, patch_size)

            logits = model(input_ids=input_patches, labels=input_patches).logits
            predictions = torch.argmax(logits, dim=-1)

            labels_flat = input_patches.reshape(-1)
            mask = labels_flat != tokenizer.pad_token_id

            total += mask.sum().item()
            correct += ((predictions.reshape(-1) == labels_flat) & mask).sum().item()

            if len(examples) < 3:
                examples.append({
                    "input": text,
                    "pred": tokenizer.decode(predictions[0], skip_special_tokens=True)
                })

    token_acc = correct / max(1, total)
    print(f"TOKEN_ACC={token_acc:.6f} correct={correct} total={total}")
    for idx, ex in enumerate(examples, 1):
        print(f"example_{idx}_input={ex['input']!r}")
        print(f"example_{idx}_pred={ex['pred']!r}")

if __name__ == "__main__":
    main()