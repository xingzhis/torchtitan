#!/bin/bash
# Run lm-eval on local HF checkpoints (same settings as original: num_fewshot=5, max_eval_samples=200, max_gen_tokens=1024, seed=1).
#
# Usage: ./test_hf_eval.sh <path_to_checkpoints>
#
# Two modes:
#   (1) Single model:  path is one HF model dir (e.g. .../hf_converted/step-2480/)
#       → Eval runs once; results in <output_dir>/log.txt and lm_eval_begin_0.json
#   (2) Multi-step:    path is a dir containing step-* subdirs (e.g. .../hf_converted/)
#       → Eval runs for each step-*; results in <output_dir>/step-*/log.txt and lm_eval_begin_0.json
#
# Examples:
#   ./test_hf_eval.sh ../outputs_archive/qwen_test/hf_converted/step-2480/
#   ./test_hf_eval.sh ./qwen3_0.6B_dispersion_huggingface/
#
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_local_checkpoints>"
    echo "  Single model:  $0 /path/to/hf_model_dir/"
    echo "  Multi step-*:  $0 /path/to/dir/containing/step-*/"
    exit 1
fi

MODEL_PATH="$1"
# Sanitize model path for use in dir name: strip leading ./, ../, trailing /, replace / with _
MODEL_SLUG=$(echo "$MODEL_PATH" | sed 's|^\./||;s|^\.\./||;s|/*$||;s|/|_|g')
OUTPUT_DIR="./test_results_hf_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S)"

# Disable hf_transfer fast download path unless explicitly requested.
# This avoids:
#   ValueError: Fast download using 'hf_transfer' is enabled
#   (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available...
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Ensure we use the C++ runtime from this micromamba/conda env instead of the
# (older) system one, which avoids errors like:
#   ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found ...
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "==========================================="
echo "Running HF lm-eval test"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "==========================================="

# Reproducible eval: use original script and its defaults (num_fewshot=5, max_eval_samples=200, max_gen_tokens=1024, seed=1).
python eval_hf_original.py \
    --model_name "$MODEL_PATH" \
    --checkpoint_dir "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --parallelize

echo ""
echo "==========================================="
echo "HF Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "==========================================="
echo ""
echo "To view results:"
echo "  Single model:  cat $OUTPUT_DIR/log.txt  cat $OUTPUT_DIR/lm_eval_begin_0.json"
echo "  Multi step-*:  cat $OUTPUT_DIR/step-*/log.txt  cat $OUTPUT_DIR/step-*/lm_eval_begin_0.json"
