# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CALM (Continuous Autoregressive Language Models) is a research repository that trains LLMs to predict continuous vectors representing K-token chunks instead of discrete tokens. The system has two training stages: (1) train an autoencoder to compress K tokens into a single vector and reconstruct them, then (2) train a generative head (energy-based, diffusion, or flow matching) on the latent space.

## Architecture

### Models (`models/`)

| File | Purpose |
|------|---------|
| `configuration_autoencoder.py` | `AutoencoderConfig`: encoder-decoder config with `latent_size`, `patch_size`, `num_encoder_layers`, `num_decoder_layers` |
| `configuration_calm.py` | `CALMConfig`: Transformer decoder config with Llama-style args |
| `modeling_autoencoder.py` | `Autoencoder` with `Encoder` + `Decoder` classes; `Encoder` groups tokens into patches via `squeeze_layer`, predicts latent mean/log_std; `Decoder` reconstructs |
| `modeling_calm.py` | `CALM`: base Transformer for latent autoregression; includes `CustomCausalLMOutput` |
| `modeling_energy.py` | `EnergyTransformer`: energy-based CALM head; scores (context, latent) pairs with an `MLPBlock`; primary training method |
| `modeling_diffusion.py` | `DiffusionTransformer`: diffusion-based head |
| `modeling_flow.py` | `FlowTransformer`: flow matching head |
| `diffusion/` | Gaussian diffusion utilities (`gaussian_diffusion.py`, `respace.py`) |

### Training Scripts (`train/`)

All are `torchrun -m` entrypoints using HuggingFace Trainer, running in the repo root:
- `train_autoencoder.py` / `train_autoencoder.sh` — Stage 1: trains the autoencoder
- `train_calm.py` / `train_energy.sh` — Stage 2: trains energy-based CALM head; requires `--ae_name_or_path`
- `train_diffusion.sh` / `train_flow.sh` — Stage 2: alternate head types
- `train_ar.py` / `train_ar.sh` — Baseline autoregressive Transformer (token-level, same BrierLM eval)
- `eval_energy.sh` — Evaluates a pretrained checkpoint; runs `--do_eval` with `train_calm.py`

### Data Preparation (`data/`)

Run `bash data/get_data.sh` to download and process the pile-uncopyrighted dataset (~2.5TB disk required). Output is a set of `.text.jsonl` files at `pile-uncopyrighted/train/XX.text.jsonl`.

### Other Files

- `make_data.py` — Dataset preprocessing utilities
- `train_ae_semantic_direct.py`, `train_latent_predictor.py`, `train_latent_predictor_transformer.py` — Exploratory scripts for latent predictor experiments

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare tokenizer (downloads LLaMA 3 tokenizer files into llama3_tokenizer/)
# Must be done before training

# Stage 1: Train autoencoder
bash train/train_autoencoder.sh

# Stage 2: Train CALM (energy-based)
bash train/train_energy.sh

# Evaluate pretrained models
bash train/eval_energy.sh

# Train baseline autoregressive model
bash train/train_ar.sh
```

All training scripts require environment variables to be set: `WORK_PATH` (repo root), `CHECKPOINT_PATH`, `TOKENIZER_PATH`, `AE_PATH`, `DATASET_TRAIN`, `DATASET_VALID`. The repo root also needs `llama3_tokenizer/` (LLaMA 3 tokenizer files) and `pile-uncopyrighted/` (processed training data).

## Key Concepts

- **patch_size (K)**: number of tokens grouped into one latent vector; default K=4
- **latent_size**: dimension of the continuous representation per patch; default 128
- **BrierLM**: likelihood-free calibration metric used for evaluation (computed during eval steps)
- **Energy loss**: primary training objective; the model learns to assign low energy to correct (context, latent) pairs and high energy to negatives
- **Streaming mode**: `transformers` Trainer is used with `--streaming` for large dataset handling without full loading

## Dependencies

Key pinned versions: `transformers==4.43.0`, `accelerate==0.30.1`, `deepspeed==0.10.0`, `torch>=1.13.0`. Flash attention is optionally supported.