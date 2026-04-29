#!/usr/bin/env python
# coding=utf-8
"""
Train autoencoder on Pile-uncopyrighted for semantic latent space.
"""
import logging
import os
import sys
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, default_data_collator

from models.configuration_autoencoder import AutoencoderConfig
from models.modeling_autoencoder import Autoencoder

logger = logging.getLogger(__name__)

def main():
    model_name = "gpt2"
    output_dir = "out_ae_semantic"
    block_size = 128
    max_train_samples = 20000
    num_train_epochs = 5
    per_device_train_batch_size = 1
    learning_rate = 5e-4
    seed = 42

    set_seed(seed)
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoencoderConfig(
        hidden_size=128,
        latent_size=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        patch_size=4,
        kl_weight=0.0,
        ae_dropout=0.0,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = Autoencoder(config)
    model = model.cuda()
    model.train()

    logger.info("Loading monology/pile-uncopyrighted dataset (streaming)...")
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    def is_good_text(text):
        if not text or not isinstance(text, str):
            return False
        text = text.strip()
        if len(text) < 20:
            return False
        if text.count("@") > 5:
            return False
        if text.count("http") > 2:
            return False
        return True

    logger.info("Tokenizing and preparing data...")
    tokenized_texts = []
    count = 0
    for item in dataset:
        text = item.get("text", "")
        if is_good_text(text):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
            )
            if len(encoded["input_ids"]) >= 20:
                tokenized_texts.append(torch.tensor(encoded["input_ids"], dtype=torch.long))
                count += 1
                if count >= max_train_samples:
                    break

    logger.info(f"Tokenized {len(tokenized_texts)} examples")

    patch_size = config.patch_size
    padded_tensors = []
    for t in tokenized_texts:
        if len(t) < block_size:
            pad_len = block_size - len(t)
            t = torch.cat([t, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
        else:
            t = t[:block_size]
        padded_tensors.append(t)

    input_ids = torch.stack(padded_tensors)
    dataset_torch = torch.utils.data.TensorDataset(input_ids)

    train_loader = DataLoader(
        dataset_torch,
        batch_size=per_device_train_batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    logger.info(f"Starting training for {num_train_epochs} epochs...")
    model.train()
    global_step = 0

    for epoch in range(num_train_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (batch,) in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.cuda()

            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"epoch={epoch+1}/{num_train_epochs} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"epoch={epoch+1} avg_loss={avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    logger.info(f"Model saved to {output_dir}")

    logger.info("Running validation...")
    model.eval()
    with torch.no_grad():
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny and warm.",
        ]
        for text in test_texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=block_size)
            labels = encoded["input_ids"]
            outputs = model(input_ids=encoded["input_ids"], labels=labels)
            predictions = torch.argmax(outputs.logits, dim=-1)

            input_decoded = tokenizer.decode(labels[0], skip_special_tokens=True)
            output_decoded = tokenizer.decode(predictions[0], skip_special_tokens=True)

            logger.info(f"Input:  {input_decoded}")
            logger.info(f"Output: {output_decoded}")


if __name__ == "__main__":
    main()