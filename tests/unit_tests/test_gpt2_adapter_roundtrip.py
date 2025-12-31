#!/usr/bin/env python3
"""
Test round-trip conversion: TorchTitan → HuggingFace → TorchTitan
Validates that the adapter can convert in both directions without losing information.
"""

import torch
from torchtitan.models.gpt2 import GPT2, gpt2_args, GPT2StateDictAdapter


def test_roundtrip_conversion():
    """Test that we can convert TT → HF → TT without losing information."""
    print("=" * 70)
    print("Testing Round-Trip Conversion: TorchTitan → HF → TorchTitan")
    print("=" * 70)
    
    model_args = gpt2_args["124M"]
    
    print("\n1. Creating TorchTitan model...")
    model = GPT2(model_args)
    model.init_weights()
    original_state = model.state_dict()
    print(f"✓ Original TorchTitan state dict: {len(original_state)} keys")
    
    print("\n2. Converting to HuggingFace format...")
    adapter = GPT2StateDictAdapter(model_args, None)
    hf_state = adapter.to_hf(original_state)
    print(f"✓ HuggingFace state dict: {len(hf_state)} keys")
    
    # Verify we have the expected HF keys (GPT-2 format without transformer. prefix)
    expected_hf_keys = ["wte.weight", "wpe.weight", "lm_head.weight"]
    for key in expected_hf_keys:
        assert key in hf_state, f"Missing expected HF key: {key}"
    print(f"✓ All expected HF keys present")

    # Verify weight tying: wte.weight and lm_head.weight should be identical
    if torch.allclose(hf_state["wte.weight"], hf_state["lm_head.weight"]):
        print("✓ Weight tying preserved (wte.weight == lm_head.weight)")
    else:
        print("⚠️  Weight tying NOT preserved!")
        return False
    
    print("\n3. Converting back to TorchTitan format...")
    restored_state = adapter.from_hf(hf_state)
    print(f"✓ Restored TorchTitan state dict: {len(restored_state)} keys")
    
    print("\n4. Comparing original and restored state dicts...")
    # Check keys (lm_head.weight might be missing due to weight tying)
    original_keys = set(original_state.keys())
    restored_keys = set(restored_state.keys())
    
    missing_keys = original_keys - restored_keys
    expected_missing = {"lm_head.weight"}  # Weight tied, so this is expected
    
    if missing_keys == expected_missing:
        print(f"✓ Expected missing keys (weight-tied): {missing_keys}")
    elif missing_keys:
        print(f"⚠️  Unexpected missing keys: {missing_keys - expected_missing}")
        return False
    else:
        print("✓ All keys preserved")
    
    # Compare parameter values
    max_diff = 0.0
    for key in original_state.keys():
        if key in restored_state:
            diff = (original_state[key] - restored_state[key]).abs().max().item()
            max_diff = max(max_diff, diff)
            
            if diff > 1e-6:
                print(f"  ⚠️  {key}: diff={diff:.2e}")
    
    print(f"\n  Maximum difference across all parameters: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✓ Round-trip conversion is perfect!")
    elif max_diff < 1e-3:
        print("✓ Round-trip conversion successful (small numerical differences)")
    else:
        print(f"⚠️  Round-trip conversion has significant errors: {max_diff}")
        return False
    
    return True


def test_export_creates_valid_hf_structure():
    """Test that exported HF state dict has the correct structure."""
    print("\n" + "=" * 70)
    print("Testing HuggingFace Export Structure")
    print("=" * 70)
    
    model_args = gpt2_args["124M"]
    
    print("\n1. Creating TorchTitan model and exporting...")
    model = GPT2(model_args)
    model.init_weights()
    
    adapter = GPT2StateDictAdapter(model_args, None)
    hf_state = adapter.to_hf(model.state_dict())
    
    print("\n2. Checking HuggingFace state dict structure...")
    
    # Check essential keys
    required_keys = {
        "transformer.wte.weight",
        "transformer.wpe.weight", 
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "lm_head.weight",
    }
    
    for key in required_keys:
        if key not in hf_state:
            print(f"  ❌ Missing required key: {key}")
            return False
    print(f"✓ All required top-level keys present")
    
    # Check layer keys
    n_layers = model_args.n_layers
    for i in range(n_layers):
        layer_keys = [
            f"transformer.h.{i}.ln_1.weight",
            f"transformer.h.{i}.attn.c_attn.weight",
            f"transformer.h.{i}.attn.c_proj.weight",
            f"transformer.h.{i}.mlp.c_fc.weight",
            f"transformer.h.{i}.mlp.c_proj.weight",
        ]
        for key in layer_keys:
            if key not in hf_state:
                print(f"  ❌ Missing layer key: {key}")
                return False
    
    print(f"✓ All {n_layers} layers have required keys")
    
    # Check shapes
    print("\n3. Checking parameter shapes...")

    # Embeddings (GPT-2 format without transformer. prefix)
    assert hf_state["wte.weight"].shape == (model_args.vocab_size, model_args.dim)
    assert hf_state["wpe.weight"].shape == (model_args.max_seq_len, model_args.dim)
    print("✓ Embedding shapes correct")

    # Attention weights (Conv1D format: in_features, out_features)
    c_attn_shape = hf_state["h.0.attn.c_attn.weight"].shape
    expected_c_attn = (model_args.dim, model_args.dim * 3)  # Conv1D format (in, out)
    assert c_attn_shape == expected_c_attn, \
        f"c_attn shape mismatch: {c_attn_shape} != {expected_c_attn}"
    print("✓ Attention weight shapes correct (Conv1D format)")

    # MLP weights (Conv1D format)
    c_fc_shape = hf_state["h.0.mlp.c_fc.weight"].shape
    expected_c_fc = (model_args.dim, model_args.dim * 4)  # Conv1D format (in, out)
    assert c_fc_shape == expected_c_fc, \
        f"c_fc shape mismatch: {c_fc_shape} != {expected_c_fc}"
    print("✓ MLP weight shapes correct (Conv1D format)")
    
    print("\n✓ Export structure is valid!")
    return True


def main():
    print("=" * 70)
    print("GPT-2 Adapter Round-Trip Tests")
    print("=" * 70)
    print("\nValidating bidirectional conversion:")
    print("  - TorchTitan → HuggingFace")
    print("  - HuggingFace → TorchTitan")
    print("  - Round-trip preservation\n")
    
    try:
        result1 = test_roundtrip_conversion()
        result2 = test_export_creates_valid_hf_structure()
        
        if result1 and result2:
            print("\n" + "=" * 70)
            print("✅ ALL ROUND-TRIP TESTS PASSED!")
            print("=" * 70)
            print("\nThe adapter correctly converts in both directions:")
            print("  ✓ TorchTitan → HuggingFace (export)")
            print("  ✓ HuggingFace → TorchTitan (import)")
            print("  ✓ Round-trip preservation")
        else:
            print("\n⚠️  Some tests failed. Review the output above.")
            exit(1)
            
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

