#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.configuration_autoencoder import AutoencoderConfig
from models.modeling_autoencoder import Autoencoder


class LatentPredictor(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.net(latent)


class LatentPredictorTransformer(nn.Module):
    def __init__(self, latent_size: int, context_length: int = 3, nhead: int = 2,
                 num_layers: int = 2, dim_feedforward: int = 128):
        super().__init__()
        self.context_length = context_length
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        output = self.transformer(latent_sequence)
        next_latent = output[:, -1, :]
        last_latent = latent_sequence[:, -1, :]
        next_latent = last_latent + next_latent
        return next_latent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roll out a trained latent predictor and decode each predicted latent."
    )
    parser.add_argument("--ae_path", default="out_untied_lr1e3")
    parser.add_argument("--predictor_path", default="out_latent_predictor_transformer/predictor.pt")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument(
        "--sentence",
        default="The evolution of pasta is a strange but interesting concept.",
    )
    return parser.parse_args()


def pad_to_multiple(input_ids: torch.Tensor, multiple: int, pad_token_id: int) -> torch.Tensor:
    remainder = input_ids.shape[1] % multiple
    if remainder == 0:
        return input_ids
    pad_len = multiple - remainder
    pad = torch.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype)
    return torch.cat([input_ids, pad], dim=1)


def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents / (latents.norm(dim=-1, keepdim=True) + 1e-6)


@torch.no_grad()
def encode_text_to_latents(text, tokenizer, autoencoder, patch_size, block_size):
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    token_ids = pad_to_multiple(token_ids, patch_size, tokenizer.pad_token_id)
    block_size = max(patch_size, (block_size // patch_size) * patch_size)

    latent_chunks = []
    for start in range(0, token_ids.shape[1], block_size):
        chunk = token_ids[:, start : start + block_size]
        chunk = pad_to_multiple(chunk, patch_size, tokenizer.pad_token_id)
        encoded = autoencoder.encoder(input_ids=chunk)
        mean, _ = torch.chunk(encoded, 2, dim=-1)
        latent_chunks.append(mean.squeeze(0).cpu())

    return torch.cat(latent_chunks, dim=0)


@torch.no_grad()
def decode_latent(autoencoder, tokenizer, latent):
    logits = autoencoder.decoder(latent_states=latent.reshape(1, 1, -1))
    token_ids = torch.argmax(logits, dim=-1)[0]
    return token_ids.tolist(), tokenizer.decode(token_ids, skip_special_tokens=True)


@torch.no_grad()
def rollout(predictor, start_latents, steps, normalize_inputs=False):
    """
    Args:
        predictor: MLP or Transformer predictor
        start_latents: tensor of shape (latent_size,) or (context_length, latent_size)
        steps: number of steps to rollout
        normalize_inputs: whether to normalize latents before prediction
    """
    is_transformer = isinstance(predictor, LatentPredictorTransformer)
    
    if is_transformer:
        # start_latents should be (context_length, latent_size)
        current_sequence = start_latents.float().clone()
        if current_sequence.dim() == 1:
            # Fallback if only one latent provided: repeat it
            current_sequence = current_sequence.unsqueeze(0).repeat(predictor.context_length, 1)
        elif current_sequence.shape[0] > predictor.context_length:
            current_sequence = current_sequence[-predictor.context_length:]
        elif current_sequence.shape[0] < predictor.context_length:
            # Pad by repeating first element if too short
            padding = current_sequence[0:1].repeat(predictor.context_length - current_sequence.shape[0], 1)
            current_sequence = torch.cat([padding, current_sequence], dim=0)
    else:
        # MLP expects (latent_size,)
        current_latent = start_latents.float().clone()
        if current_latent.dim() == 2:
            current_latent = current_latent[-1]

    predicted_latents = []
    for _ in range(steps):
        if is_transformer:
            predictor_input = normalize_latents(current_sequence) if normalize_inputs else current_sequence
            # Add batch dimension
            next_latent = predictor(predictor_input.unsqueeze(0)).squeeze(0)
            # Add noise during rollout to break cyclic attractors
            next_latent = next_latent + torch.randn_like(next_latent) * 0.02
            predicted_latents.append(next_latent.detach().clone())
            
            # Update sequence window
            current_sequence = torch.cat([current_sequence[1:], next_latent.unsqueeze(0)], dim=0)
        else:
            predictor_input = normalize_latents(current_latent) if normalize_inputs else current_latent
            next_latent = predictor(predictor_input)
            predicted_latents.append(next_latent.detach().clone())
            current_latent = next_latent
            
    return predicted_latents


def load_predictor(path, fallback_latent_size):
    checkpoint = torch.load(path, map_location="cpu")
    model_type = checkpoint.get("model_type", "mlp")
    latent_size = int(checkpoint.get("latent_size", fallback_latent_size))
    
    if model_type == "transformer":
        context_length = int(checkpoint.get("context_length", 3))
        predictor = LatentPredictorTransformer(latent_size=latent_size, context_length=context_length)
    else:
        hidden_size = int(checkpoint.get("hidden_size", 128))
        predictor = LatentPredictor(latent_size=latent_size, hidden_size=hidden_size)
        
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()
    return predictor, checkpoint


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.ae_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoencoderConfig.from_pretrained(args.ae_path)
    config.pad_token_id = tokenizer.pad_token_id
    autoencoder = Autoencoder.from_pretrained(args.ae_path, config=config)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    predictor, predictor_meta = load_predictor(args.predictor_path, config.latent_size)
    if predictor_meta.get("latent_size", config.latent_size) != config.latent_size:
        raise ValueError(
            f"Predictor latent_size={predictor_meta.get('latent_size')} does not match "
            f"autoencoder latent_size={config.latent_size}."
        )

    latents = encode_text_to_latents(
        args.sentence,
        tokenizer,
        autoencoder,
        config.patch_size,
        args.block_size,
    )
    normalize_inputs = bool(predictor_meta.get("normalize_latents", False))
    
    is_transformer = isinstance(predictor, LatentPredictorTransformer)
    if is_transformer:
        context_length = predictor.context_length
        start_latents = latents[-context_length:]
        # We use the norm of the last latent to scale for decoding
        decode_scale = latents[-1].norm(dim=-1, keepdim=True) if normalize_inputs else 1.0
    else:
        start_latents = latents[-1]
        decode_scale = latents[-1].norm(dim=-1, keepdim=True) if normalize_inputs else 1.0
    
    predicted_latents = rollout(predictor, start_latents, args.steps, normalize_inputs)

    print(f"input={args.sentence!r}")
    print(f"encoded_patches={latents.shape[0]} latent_size={latents.shape[1]} steps={args.steps}")
    print()
    print("Rollout:")

    previous = latents[-1]
    decoded_token_sequences = []
    for step, latent in enumerate(predicted_latents, start=1):
        decode_latent_value = latent * decode_scale if normalize_inputs else latent
        token_ids, decoded = decode_latent(autoencoder, tokenizer, decode_latent_value)
        l2 = torch.norm(latent - previous).item()
        cosine = F.cosine_similarity(latent, previous, dim=0).item()
        decoded_token_sequences.append(tuple(token_ids))
        print(
            f"Step {step}: text={decoded!r} tokens={token_ids} "
            f"l2_from_previous={l2:.6f} cosine_from_previous={cosine:.6f}"
        )
        previous = latent

    unique_outputs = len(set(decoded_token_sequences))
    print()
    print(f"unique_decoded_token_sequences={unique_outputs}/{len(decoded_token_sequences)}")
    if unique_outputs == 1 and decoded_token_sequences:
        print("warning=all rollout steps decoded to identical tokens; possible predictor collapse")


if __name__ == "__main__":
    main()
