#!/bin/bash
# Test script for vLLM version of lm-eval
# Usage: ./test_vllm_eval.sh <path_to_local_model>

set -e

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_local_model>"
    echo "Example: $0 ./my_custom_model"
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="./test_results_vllm_$(date +%Y%m%d_%H%M%S)"

echo "==========================================="
echo "Running vLLM lm-eval test"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "==========================================="

python3 eval_vllm.py \
    --model_name "$MODEL_PATH" \
    --checkpoint_dir "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_fewshot 2 \
    --max_eval_samples 50 \
    --max_gen_tokens 256 \
    --batch_size auto \
    --seed 42 \
    --dtype auto \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096
    # Reduced from model's 40K to prevent OOM
    # Most eval tasks don't need >8K context
    # --zeroshot_tasks hellaswag piqa \
    # --fewshot_tasks arc_easy mmlu

echo ""
echo "==========================================="
echo "vLLM Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "==========================================="
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/step-*/log.txt"
echo "  cat $OUTPUT_DIR/step-*/lm_eval_begin_0.json"
