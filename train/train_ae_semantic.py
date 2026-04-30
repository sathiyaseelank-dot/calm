#!/usr/bin/env python
# coding=utf-8
"""
Train autoencoder on monology/pile-uncopyrighted with quality filtering.
"""
import logging
import math
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from models.configuration_autoencoder import AutoencoderConfig
from models.modeling_autoencoder import Autoencoder

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    tokenizer_name: Optional[str] = field(default="gpt2")
    model_name_or_path: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(default="monology/pile-uncopyrighted")
    max_train_samples: Optional[int] = field(default=20000)
    max_eval_samples: Optional[int] = field(default=1000)
    streaming: bool = field(default=True)
    block_size: int = field(default=128)
    min_text_length: int = field(default=20)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="out_ae_semantic")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: float = field(default=4)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=5000)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=5000)
    learning_rate: float = field(default=3e-4)
    lr_scheduler_type: str = field(default="constant")
    logging_steps: int = field(default=100)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    overwrite_output_dir: bool = field(default=True)
    bf16: bool = field(default=True)
    seed: int = field(default=42)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging setup
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)

    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    # Set seed
    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

    # Model config
    config = AutoencoderConfig()
    if model_args.config_overrides:
        config.update_from_string(model_args.config_overrides)
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    # Explicit config
    config.hidden_size = 128
    config.latent_size = 8
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.patch_size = 4
    config.kl_weight = 0.0
    config.ae_dropout = 0.0

    model = Autoencoder._from_config(config)

    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    raw_datasets = load_dataset(data_args.dataset_name, streaming=data_args.streaming, split="train")

    # Filter short texts
    logger.info(f"Filtering texts shorter than {data_args.min_text_length} chars")
    raw_datasets = raw_datasets.filter(lambda x: len(x["text"]) >= data_args.min_text_length)

    # Prepare validation split from streaming dataset - take first 1000 samples for eval
    logger.info("Creating validation split")
    all_data = list(raw_datasets.take(data_args.max_train_samples + data_args.max_eval_samples))
    train_texts = [x["text"] for x in all_data[:data_args.max_train_samples]]
    val_texts = [x["text"] for x in all_data[data_args.max_train_samples:data_args.max_train_samples + data_args.max_eval_samples]]

    logger.info(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

    # Tokenize
    text_column_name = "text"

    def tokenize_function(examples):
        with CaptureLogger(logger) as cl:
            output = tokenizer(examples[text_column_name])
        return output

    def group_texts(examples):
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        processed_input_ids = []
        processed_attention_masks = []

        for i in range(len(examples['input_ids'])):
            input_ids_seq = examples['input_ids'][i]
            attention_mask_seq = examples['attention_mask'][i]
            input_ids_seq.append(eos_token_id)
            attention_mask_seq.append(1)
            current_length = len(input_ids_seq)
            remainder = current_length % config.patch_size
            if remainder != 0:
                padding_needed = config.patch_size - remainder
                input_ids_seq.extend([pad_token_id] * padding_needed)
                attention_mask_seq.extend([1] * padding_needed)
            processed_input_ids.append(input_ids_seq)
            processed_attention_masks.append(attention_mask_seq)

        concatenated_examples = {
            'input_ids': list(chain(*processed_input_ids)),
            'attention_mask': list(chain(*processed_attention_masks))
        }
        total_length = len(concatenated_examples['input_ids'])
        total_length = (total_length // data_args.block_size) * data_args.block_size
        result = {
            k: [t[i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    from datasets import Dataset
    train_dataset = Dataset.from_list([{"text": t} for t in train_texts])
    val_dataset = Dataset.from_list([{"text": t} for t in val_texts])

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing train")
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing val")

    lm_train = tokenized_train.map(group_texts, batched=True, desc="Grouping train")
    lm_val = tokenized_val.map(group_texts, batched=True, desc="Grouping val")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info(f"Training complete. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
