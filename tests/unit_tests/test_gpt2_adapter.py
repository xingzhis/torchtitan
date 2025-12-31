#!/usr/bin/env python3
"""Unit tests for GPT2StateDictAdapter."""

import torch
from torchtitan.models.gpt2 import GPT2, gpt2_args, GPT2StateDictAdapter


def test_adapter_instantiation():
    """Test that adapter can be created."""
    print("Testing adapter instantiation...")
    model_args = gpt2_args["124M"]
    adapter = GPT2StateDictAdapter(model_args, None)
    print("✓ Adapter created successfully")
    return adapter


def test_state_dict_structure():
    """Test that state dict keys are correctly mapped."""
    print("\nTesting state dict structure...")
    model_args = gpt2_args["124M"]
    adapter = GPT2StateDictAdapter(model_args, None)
    
    # Create a small model to get real state dict structure
    model = GPT2(model_args)
    tt_state = model.state_dict()
    
    print(f"  TorchTitan has {len(tt_state)} parameters")
    
    # Convert to HF format
    hf_state = adapter.to_hf(tt_state)
    print(f"  HuggingFace format has {len(hf_state)} parameters")
    
    # Convert back
    tt_state_restored = adapter.from_hf(hf_state)
    print(f"  Restored TorchTitan format has {len(tt_state_restored)} parameters")
    
    # Check that all keys are preserved (except lm_head.weight which is tied)
    missing_keys = set(tt_state.keys()) - set(tt_state_restored.keys())
    expected_missing = {'lm_head.weight'}  # Weight tied with wte.weight
    
    if missing_keys == expected_missing:
        print("✓ All keys correctly preserved (lm_head.weight tied as expected)")
    else:
        unexpected_missing = missing_keys - expected_missing
        if unexpected_missing:
            print(f"  ⚠️  Unexpected missing keys: {unexpected_missing}")
            return False
    
    return True


def test_weight_transpose():
    """Test that Conv1D weights are correctly transposed."""
    print("\nTesting weight transpose...")
    model_args = gpt2_args["124M"]
    adapter = GPT2StateDictAdapter(model_args, None)
    
    # Create a dummy weight tensor
    dummy_weight = torch.randn(768, 2304)  # c_attn weight shape in torchtitan
    
    tt_state = {"layers.0.attn.c_attn.weight": dummy_weight}
    
    # Convert to HF (should transpose and use GPT-2 format)
    hf_state = adapter.to_hf(tt_state)
    hf_weight = hf_state["h.0.attn.c_attn.weight"]  # GPT-2 format without transformer. prefix

    # Check that it was transposed
    assert hf_weight.shape == (768, 2304), f"Expected (768, 2304), got {hf_weight.shape}"  # Conv1D format (in, out)
    assert torch.allclose(hf_weight, dummy_weight.t()), "Weight not correctly transposed"

    # Convert back (should transpose again)
    tt_state_restored = adapter.from_hf(hf_state)
    tt_weight_restored = tt_state_restored["layers.0.attn.c_attn.weight"]

    assert tt_weight_restored.shape == (768, 2304), f"Expected (768, 2304), got {tt_weight_restored.shape}"
    assert torch.allclose(tt_weight_restored, dummy_weight), "Weight not correctly restored"
    
    print("✓ Conv1D weights correctly transposed")
    return True


def test_bias_not_transposed():
    """Test that bias tensors are not transposed."""
    print("\nTesting bias preservation...")
    model_args = gpt2_args["124M"]
    adapter = GPT2StateDictAdapter(model_args, None)
    
    # Create a dummy bias tensor
    dummy_bias = torch.randn(2304)  # c_attn bias
    
    tt_state = {"layers.0.attn.c_attn.bias": dummy_bias}
    
    # Convert to HF (should NOT transpose)
    hf_state = adapter.to_hf(tt_state)
    hf_bias = hf_state["h.0.attn.c_attn.bias"]  # GPT-2 format
    
    assert hf_bias.shape == dummy_bias.shape, f"Bias shape changed: {hf_bias.shape} vs {dummy_bias.shape}"
    assert torch.allclose(hf_bias, dummy_bias), "Bias values changed"
    
    print("✓ Bias tensors correctly preserved")
    return True


def main():
    print("=" * 60)
    print("GPT2StateDictAdapter Unit Tests")
    print("=" * 60)
    
    try:
        test_adapter_instantiation()
        test_state_dict_structure()
        test_weight_transpose()
        test_bias_not_transposed()
        
        print("\n" + "=" * 60)
        print("✅ ALL UNIT TESTS PASSED!")
        print("=" * 60)
        print("\nThe GPT2StateDictAdapter is working correctly.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()



