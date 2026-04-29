#!/usr/bin/env python
# coding=utf-8
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
sys.path.insert(0, '/home/inkyank-08/Downloads/calm')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from models.configuration_autoencoder import AutoencoderConfig
from models.modeling_autoencoder import Autoencoder
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Settings
output_dir = "out_ae_semantic"
block_size = 128
num_epochs = 10  # More epochs since we start from existing weights
batch_size = 4
learning_rate = 1e-4  # Lower LR for fine-tuning
seed = 42

torch.manual_seed(seed)

# Use existing llama tokenizer (same as the autoencoder was trained with)
# Actually use GPT-2 tokenizer as requested
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Load Pile samples
logger.info("Loading Pile samples...")
samples = torch.load("/tmp/pile_samples.pt")
logger.info(f"Loaded {len(samples)} samples")

# Pad/truncate to block_size
patch_size = 4
padded = []
for s in samples:
    if len(s) < block_size:
        pad_len = block_size - len(s)
        s = torch.cat([s, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
    else:
        s = s[:block_size]
    padded.append(s)

input_ids = torch.stack(padded)
logger.info(f"Input shape: {input_ids.shape}")

# Create model with improved capacity
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

# Initialize from existing autoencoder if possible, otherwise from scratch
try:
    # Try loading the existing model and adapt it
    base_ae = Autoencoder.from_pretrained("out_untied_lr1e3")
    logger.info("Loaded base autoencoder from out_untied_lr1e3")
    
    # Create new model with larger capacity
    model = Autoencoder(config)
    
    # Copy encoder layers where possible
    for i in range(min(len(base_ae.encoder.encoder_layers), len(model.encoder.encoder_layers))):
        model.encoder.encoder_layers[i].load_state_dict(base_ae.encoder.encoder_layers[i].state_dict())
    model.encoder.squeeze_layer.load_state_dict(base_ae.encoder.squeeze_layer.state_dict())
    model.encoder.embed_tokens.load_state_dict(base_ae.encoder.embed_tokens.state_dict())
    model.encoder.norm.load_state_dict(base_ae.encoder.norm.state_dict())
    
    # Decoder layers
    for i in range(min(len(base_ae.decoder.decoder_layers), len(model.decoder.decoder_layers))):
        model.decoder.decoder_layers[i].load_state_dict(base_ae.decoder.decoder_layers[i].state_dict())
    model.decoder.expand_layer.load_state_dict(base_ae.decoder.expand_layer.state_dict())
    model.decoder.latent_to_hidden.load_state_dict(base_ae.decoder.latent_to_hidden.state_dict())
    model.decoder.norm.load_state_dict(base_ae.decoder.norm.state_dict())
    model.decoder.lm_head.load_state_dict(base_ae.decoder.lm_head.state_dict())
    
    logger.info("Copied weights from base model")
except Exception as e:
    logger.warning(f"Could not load base model: {e}")
    logger.info("Initializing from scratch")
    model = Autoencoder(config)

model.train()
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Dataset and loader
dataset = TensorDataset(input_ids)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
logger.info(f"Training for {num_epochs} epochs...")
best_loss = float('inf')

for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch,) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logger.info(f"epoch={epoch+1}/{num_epochs} batch={batch_idx}/{len(train_loader)} loss={loss.item():.4f}")
    
    avg_loss = total_loss / max(1, num_batches)
    logger.info(f"epoch={epoch+1} avg_loss={avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss

# Save model
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
config.save_pretrained(output_dir)
logger.info(f"Model saved to {output_dir}")

# Validation
logger.info("=" * 60)
logger.info("Validation on training set samples:")
model.eval()
total_correct = 0
total_tokens = 0

with torch.no_grad():
    # Test on first 50 samples from training set
    test_samples = input_ids[:50]
    for i in range(0, len(test_samples), 5):
        batch = test_samples[i:i+5]
        outputs = model(input_ids=batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        for j in range(len(batch)):
            labels = batch[j]
            pred = predictions[j]
            mask = labels != tokenizer.pad_token_id
            total_correct += ((predictions == batch) & mask).sum().item()
            total_tokens += mask.sum().item()
        
        # Show a few examples
        if i == 0:
            for j in range(min(3, len(batch))):
                labels = batch[j]
                pred = predictions[j]
                mask = labels != tokenizer.pad_token_id
                
                input_decoded = tokenizer.decode(labels[mask], skip_special_tokens=True)
                output_decoded = tokenizer.decode(pred[mask], skip_special_tokens=True)
                
                logger.info(f"Example {j+1}:")
                logger.info(f"  Input:  {input_decoded[:100]}")
                logger.info(f"  Output: {output_decoded[:100]}")

token_acc = total_correct / max(1, total_tokens)
logger.info(f"TOKEN_ACC={token_acc:.4f} correct={total_correct} total={total_tokens}")

# Test on unseen sentences
logger.info("=" * 60)
logger.info("Test on unseen sentences:")
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny and warm.",
    "Paris is the capital of France and known for its art museums.",
]

model.eval()
with torch.no_grad():
    for text in test_texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=block_size)
        labels = encoded["input_ids"]
        outputs = model(input_ids=encoded["input_ids"])
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        mask = labels != tokenizer.pad_token_id
        input_decoded = tokenizer.decode(labels[0][mask[0]], skip_special_tokens=True)
        output_decoded = tokenizer.decode(predictions[0][mask[0]], skip_special_tokens=True)
        
        match = input_decoded == output_decoded
        logger.info(f"Input:  {input_decoded}")
        logger.info(f"Output: {output_decoded} {'[MATCH]' if match else ''}")

logger.info("=" * 60)
if token_acc >= 0.85:
    logger.info("SUCCESS: TOKEN_ACC >= 0.85")
else:
    logger.info(f"WARNING: TOKEN_ACC={token_acc:.4f} < 0.85 (may need more training)")

logger.info("Done!")
