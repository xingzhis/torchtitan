#!/usr/bin/env python3
"""
Compare results from HF and vLLM evaluation runs.
This script loads the JSON results from both versions and compares metrics.
"""

import json
import sys
import os
from pathlib import Path


def load_results(result_dir):
    """Load all lm_eval results from a directory.

    Supports two structures:
    1. Multi-checkpoint: result_dir/step-*/lm_eval_begin_0.json
    2. Single model: result_dir/lm_eval_begin_0.json
    """
    results = {}
    result_dir = Path(result_dir)

    # First, check if there's a direct lm_eval_begin_0.json (single model case)
    single_model_file = result_dir / "lm_eval_begin_0.json"
    if single_model_file.exists():
        with open(single_model_file, "r") as f:
            data = json.load(f)
            results["model"] = data["results"]
        return results

    # Otherwise, look for step-* subdirectories (multi-checkpoint case)
    for step_dir in sorted(result_dir.glob("step-*")):
        json_file = step_dir / "lm_eval_begin_0.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                data = json.load(f)
                results[step_dir.name] = data["results"]

    return results


def compare_metrics(hf_results, vllm_results, tolerance=1e-4):
    """Compare metrics between HF and vLLM results."""
    all_match = True

    # Check if same checkpoints exist
    hf_steps = set(hf_results.keys())
    vllm_steps = set(vllm_results.keys())

    if hf_steps != vllm_steps:
        print("⚠️  WARNING: Different checkpoints found!")
        print(f"   HF only: {hf_steps - vllm_steps}")
        print(f"   vLLM only: {vllm_steps - hf_steps}")
        print()

    # Compare common checkpoints
    common_steps = hf_steps & vllm_steps

    for step in sorted(common_steps):
        print(f"\n{'='*60}")
        print(f"Checkpoint: {step}")
        print('='*60)

        hf_data = hf_results[step]
        vllm_data = vllm_results[step]

        # Check if same tasks
        hf_tasks = set(hf_data.keys())
        vllm_tasks = set(vllm_data.keys())

        if hf_tasks != vllm_tasks:
            print("⚠️  WARNING: Different tasks evaluated!")
            print(f"   HF only: {hf_tasks - vllm_tasks}")
            print(f"   vLLM only: {vllm_tasks - hf_tasks}")
            all_match = False

        # Compare common tasks
        common_tasks = hf_tasks & vllm_tasks

        for task in sorted(common_tasks):
            print(f"\n  Task: {task}")

            hf_metrics = hf_data[task]
            vllm_metrics = vllm_data[task]

            if not isinstance(hf_metrics, dict) or not isinstance(vllm_metrics, dict):
                continue

            # Check if same metrics exist
            hf_metric_names = set(hf_metrics.keys())
            vllm_metric_names = set(vllm_metrics.keys())

            if hf_metric_names != vllm_metric_names:
                print(f"    ⚠️  Different metrics!")
                print(f"       HF only: {hf_metric_names - vllm_metric_names}")
                print(f"       vLLM only: {vllm_metric_names - hf_metric_names}")
                all_match = False

            # Compare common metrics
            common_metrics = hf_metric_names & vllm_metric_names

            for metric in sorted(common_metrics):
                hf_val = hf_metrics[metric]
                vllm_val = vllm_metrics[metric]

                if not isinstance(hf_val, (int, float)) or not isinstance(vllm_val, (int, float)):
                    continue

                diff = abs(hf_val - vllm_val)
                match = diff <= tolerance

                status = "✓" if match else "✗"
                print(f"    {status} {metric:25s}: HF={hf_val:.6f}, vLLM={vllm_val:.6f}, diff={diff:.6e}")

                if not match:
                    all_match = False

    return all_match


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <hf_results_dir> <vllm_results_dir>")
        print("\nExample:")
        print("  python compare_results.py ./test_results_hf_20260206_120000 ./test_results_vllm_20260206_120100")
        sys.exit(1)

    hf_dir = sys.argv[1]
    vllm_dir = sys.argv[2]

    if not os.path.isdir(hf_dir):
        print(f"Error: HF results directory not found: {hf_dir}")
        sys.exit(1)

    if not os.path.isdir(vllm_dir):
        print(f"Error: vLLM results directory not found: {vllm_dir}")
        sys.exit(1)

    print("Loading results...")
    print(f"  HF:   {hf_dir}")
    print(f"  vLLM: {vllm_dir}")

    hf_results = load_results(hf_dir)
    vllm_results = load_results(vllm_dir)

    if not hf_results:
        print(f"\nError: No results found in HF directory: {hf_dir}")
        sys.exit(1)

    if not vllm_results:
        print(f"\nError: No results found in vLLM directory: {vllm_dir}")
        sys.exit(1)

    print(f"\nFound {len(hf_results)} HF checkpoints and {len(vllm_results)} vLLM checkpoints")

    all_match = compare_metrics(hf_results, vllm_results, tolerance=1e-4)

    print(f"\n{'='*60}")
    if all_match:
        print("✓ SUCCESS: All metrics match within tolerance!")
    else:
        print("✗ MISMATCH: Some metrics differ between HF and vLLM versions")
    print('='*60)

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
