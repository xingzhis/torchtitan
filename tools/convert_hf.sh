#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to convert a Qwen3 checkpoint to HF format and
# prepare the HF config based on a chosen flavor.
#
# Usage:
#   tools/convert_hf.sh CKPT_DIR OUT_DIR FLAVOR [ASSETS_SRC] [REPO_ID]
#
# Examples:
#   tools/convert_hf.sh \
#     outputs/qwen_test/checkpoint_test/step-2480 \
#     outputs/qwen_test/hf_converted/step-2480 \
#     width_256
#
#   tools/convert_hf.sh \
#     /path/to/ckpt \
#     /path/to/hf_out \
#     0.6B \
#     ./assets/hf/Qwen3-0.6B \
#     Qwen/Qwen3-0.6B

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CKPT_DIR OUT_DIR FLAVOR [ASSETS_SRC] [REPO_ID]" >&2
  exit 1
fi

CKPT_DIR="$1"
OUT_DIR="$2"
FLAVOR="$3"        # e.g. width_256, width_512, 0.6B, ...
ASSETS_SRC="${4:-./assets/hf/Qwen3-0.6B}"
REPO_ID="${5:-Qwen/Qwen3-0.6B}"

python scripts/download_hf_assets.py --repo_id "$REPO_ID" --assets tokenizer config

python scripts/checkpoint_conversion/convert_to_hf.py \
  "$CKPT_DIR" \
  "$OUT_DIR" \
  --model_name qwen3 \
  --model_flavor "$FLAVOR" \
  --hf_assets_path "$ASSETS_SRC"

python tools/prepare_qwen3_hf_config.py \
  --flavor "$FLAVOR" \
  --assets-src "$ASSETS_SRC" \
  --hf-dir "$OUT_DIR"

