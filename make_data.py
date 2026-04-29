#!/usr/bin/env python
import os
os.makedirs('data', exist_ok=True)

from datasets import load_dataset
print('Loading dataset...')
ds = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)

samples = []
for i, item in enumerate(ds):
    if i >= 20:
        break
    text = item['text'][:500]  # Truncate long texts
    samples.append(text)
    print(f'Sample {i}')

print(f'Total: {len(samples)} samples')

with open('data/train_20.txt', 'w') as f:
    for s in samples:
        f.write(s + '\n')

print('Saved to data/train_20.txt')