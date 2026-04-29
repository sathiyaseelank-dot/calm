#!/usr/bin/env python
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small MSE predictor for latent_t -> latent_t+1 transitions."
    )
    parser.add_argument("--ae_path", default="out_untied_lr1e3")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--train_file", default="data/train_20.txt")
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--output_dir", default="out_latent_predictor")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--rollout_steps", type=int, default=3)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--test_sentence",
        default="The evolution of pasta is a strange but interesting concept.",
    )
    return parser.parse_args()


def read_text(path: str) -> str:
    file_path = Path(path)
    if file_path.suffix in {".json", ".jsonl"}:
        texts = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    return file_path.read_text(encoding="utf-8")


def pad_to_multiple(input_ids: torch.Tensor, multiple: int, pad_token_id: int) -> torch.Tensor:
    remainder = input_ids.shape[1] % multiple
    if remainder == 0:
        return input_ids
    pad_len = multiple - remainder
    pad = torch.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype)
    return torch.cat([input_ids, pad], dim=1)


@torch.no_grad()
def encode_text_to_latents(
    text: str,
    tokenizer,
    autoencoder: Autoencoder,
    patch_size: int,
    block_size: int,
) -> torch.Tensor:
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


def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents / (latents.norm(dim=-1, keepdim=True) + 1e-6)


def build_rollout_sequences(latents: torch.Tensor, rollout_steps: int, max_sequences: int = None):
    latents = normalize_latents(latents.float())
    if latents.shape[0] <= rollout_steps:
        raise ValueError(
            f"Need at least {rollout_steps + 1} latent patches to build rollout sequences."
        )

    inputs = []
    targets = []
    for start in range(latents.shape[0] - rollout_steps):
        inputs.append(latents[start])
        targets.append(latents[start + 1 : start + 1 + rollout_steps])

    x = torch.stack(inputs, dim=0)
    y = torch.stack(targets, dim=0)
    if max_sequences is not None:
        x = x[:max_sequences]
        y = y[:max_sequences]
    return x, y


def split_sequences(x, y, validation_fraction, seed):
    sequence_count = x.shape[0]
    indices = list(range(sequence_count))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(sequence_count * validation_fraction)) if sequence_count > 1 else 0
    val_indices = indices[:val_count]
    train_indices = indices[val_count:] or indices
    return (
        x[train_indices],
        y[train_indices],
        x[val_indices] if val_indices else x[train_indices],
        y[val_indices] if val_indices else y[train_indices],
    )


def make_sequences_from_file(args, path, tokenizer, autoencoder, patch_size):
    text = read_text(path)
    latents = encode_text_to_latents(
        text,
        tokenizer,
        autoencoder,
        patch_size,
        args.block_size,
    )
    return build_rollout_sequences(latents, args.rollout_steps, args.max_sequences)


def rollout_loss(model, start_latents, target_latents, loss_fn):
    current = start_latents
    loss = start_latents.new_tensor(0.0)
    predictions = []
    for step in range(target_latents.shape[1]):
        current = model(current)
        predictions.append(current)
        loss = loss + loss_fn(current, target_latents[:, step, :])
    return loss, torch.stack(predictions, dim=1)


def evaluate(model, x, y, loss_fn):
    model.eval()
    with torch.no_grad():
        loss, pred = rollout_loss(model, x, y, loss_fn)
        loss = loss.item()
        cosine = nn.functional.cosine_similarity(pred.reshape(-1, pred.shape[-1]), y.reshape(-1, y.shape[-1]), dim=-1).mean().item()
        distance = torch.norm(pred - y, dim=-1).mean().item()
    return loss, cosine, distance


@torch.no_grad()
def decode_latent(autoencoder, tokenizer, latent):
    logits = autoencoder.decoder(latent_states=latent.reshape(1, 1, -1))
    token_ids = torch.argmax(logits, dim=-1)[0]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    x, y = make_sequences_from_file(args, args.train_file, tokenizer, autoencoder, config.patch_size)
    if args.validation_file:
        train_x, train_y = x, y
        val_x, val_y = make_sequences_from_file(
            args,
            args.validation_file,
            tokenizer,
            autoencoder,
            config.patch_size,
        )
    else:
        train_x, train_y, val_x, val_y = split_sequences(x, y, args.validation_fraction, args.seed)

    predictor = LatentPredictor(config.latent_size, args.hidden_size)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    initial_val_loss, initial_cosine, initial_distance = evaluate(predictor, val_x, val_y, loss_fn)
    print(
        f"sequences={x.shape[0]} train_sequences={train_x.shape[0]} val_sequences={val_x.shape[0]} "
        f"latent_size={config.latent_size} rollout_steps={args.rollout_steps}"
    )
    print(
        f"epoch=0 train_loss=nan val_loss={initial_val_loss:.6f} "
        f"val_cosine={initial_cosine:.4f} val_l2={initial_distance:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            noisy_batch_x = batch_x + torch.randn_like(batch_x) * args.noise_std
            loss, _ = rollout_loss(predictor, noisy_batch_x, batch_y, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_cosine, val_distance = evaluate(predictor, val_x, val_y, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cosine": val_cosine,
                "val_l2": val_distance,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_cosine={val_cosine:.4f} val_l2={val_distance:.4f}"
        )

    test_latents = encode_text_to_latents(
        args.test_sentence,
        tokenizer,
        autoencoder,
        config.patch_size,
        args.block_size,
    )
    predicted_next = predictor(normalize_latents(test_latents[-1].float())).detach()
    decoded_prediction = decode_latent(autoencoder, tokenizer, predicted_next)

    torch.save(
        {
            "model_state_dict": predictor.state_dict(),
            "latent_size": config.latent_size,
            "hidden_size": args.hidden_size,
            "ae_path": args.ae_path,
            "tokenizer_path": tokenizer_path,
            "patch_size": config.patch_size,
            "residual": True,
            "noise_std": args.noise_std,
            "rollout_steps": args.rollout_steps,
            "normalize_latents": True,
        },
        output_dir / "predictor.pt",
    )

    metrics = {
        "initial_val_loss": initial_val_loss,
        "initial_val_cosine": initial_cosine,
        "initial_val_l2": initial_distance,
        "final": history[-1] if history else None,
        "history": history,
        "test_sentence": args.test_sentence,
        "decoded_predicted_next_patch": decoded_prediction,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved={output_dir / 'predictor.pt'}")
    print(f"decoded_predicted_next_patch={decoded_prediction!r}")


if __name__ == "__main__":
    main()
