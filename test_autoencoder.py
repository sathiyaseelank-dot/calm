import torch
from transformers import AutoTokenizer
from models.modeling_autoencoder import Autoencoder
from models.configuration_autoencoder import AutoencoderConfig

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("out")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoencoderConfig.from_pretrained("out")
model = Autoencoder.from_pretrained("out", config=config)

model.eval()

# Test sentence
text = "The evolution of pasta is a strange but interesting concept."

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]

# pad to a full patch
patch_size = config.patch_size
length = input_ids.shape[1]

pad_len = (patch_size - (length % patch_size)) % patch_size

if pad_len > 0:
    pad = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, pad], dim=1)

inputs["input_ids"] = input_ids

with torch.no_grad():
    outputs = model(**inputs, labels=input_ids)

# Get logits
logits = outputs.logits

# Convert to token ids
pred_ids = torch.argmax(logits, dim=-1)

if pred_ids.dim() == 2:
    pred_ids = pred_ids[0]

pred_ids = pred_ids[:length]

# Decode
decoded = tokenizer.decode(pred_ids, skip_special_tokens=True)

print("\nINPUT:")
print(text)

print("\nOUTPUT:")
print(decoded)
