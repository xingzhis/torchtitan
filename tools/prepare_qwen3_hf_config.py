#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path


# Hard-coded sizes per flavor (kept in sync with qwen3_args in torchtitan/models/qwen3/__init__.py)
FLAVORS = {
    "0.6B": dict(
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
    ),
    "width_256": dict(
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=28,
        num_attention_heads=2,
        num_key_value_heads=2,
    ),
    "width_512": dict(
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=4,
    ),
    "width_768": dict(
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=28,
        num_attention_heads=6,
        num_key_value_heads=6,
    ),
    "1.7B": dict(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
    ),
    "4B": dict(
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "8B": dict(
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    # Add more flavors here if needed.
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare HF config/tokenizer files for a given Qwen3 flavor."
    )
    parser.add_argument(
        "--flavor",
        required=True,
        help="Model flavor key, e.g. width_256, width_512, 0.6B, ...",
    )
    parser.add_argument(
        "--assets-src",
        required=True,
        help="Directory with base HF assets (config.json, tokenizer, vocab, etc.).",
    )
    parser.add_argument(
        "--hf-dir",
        required=True,
        help="Output HF model directory (where convert_to_hf wrote weights).",
    )
    args = parser.parse_args()

    flavor = args.flavor
    if flavor not in FLAVORS:
        raise ValueError(
            f"Unknown flavor '{flavor}'. Known flavors: {', '.join(FLAVORS.keys())}"
        )

    assets_src = Path(args.assets_src)
    hf_dir = Path(args.hf_dir)
    hf_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy base HF assets into the target dir.
    to_copy = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    for fname in to_copy:
        src = assets_src / fname
        dst = hf_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} does not exist, skipping.")

    # 2) Edit config.json to match the flavor and untie embeddings.
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found after copying assets")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Apply size overrides from the selected flavor.
    for key, value in FLAVORS[flavor].items():
        cfg[key] = value

    # Always untie word embeddings for converted models.
    cfg["tie_word_embeddings"] = False

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Updated HF config in {hf_dir} for flavor '{flavor}'.")


if __name__ == "__main__":
    main()

