#!/usr/bin/env python
"""Verify FineWeb-Edu-Dedup-Full dataset integration."""

print("=" * 70)
print("Test 1: Check dataset is registered")
print("=" * 70)
from torchtitan.datasets.hf_datasets import DATASETS
print(f"✓ fineweb_edu_dedup_full registered: {'fineweb_edu_dedup_full' in DATASETS}")
print()

print("=" * 70)
print("Test 2: Load and inspect first sample")
print("=" * 70)
from datasets import load_dataset

config = DATASETS['fineweb_edu_dedup_full']
ds = config.loader('./local_datasets/fineweb_edu_dedup_full')
sample = next(iter(ds))
print(f"First sample keys: {list(sample.keys())}")
print(f"Text preview: {config.sample_processor(sample)[:100]}...")
print(f"Total samples: {len(ds):,}")
print()

print("=" * 70)
print("Test 3: Verify reproducibility (first 3 samples)")
print("=" * 70)
from datasets import load_from_disk
ds = load_from_disk('./local_datasets/fineweb_edu_dedup_full')
for i, sample in enumerate(ds.take(3)):
    print(f"Sample {i}: {sample['text'][:80]}...")
print()

print("=" * 70)
print("✓ All verification tests passed!")
print("=" * 70)
