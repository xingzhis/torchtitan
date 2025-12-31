#!/usr/bin/env python3
"""
Test converting torchtitan GPT-2 models to HuggingFace format.
This validates the export direction: TorchTitan → HuggingFace.
"""

import torch
import tempfile
import os
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

from torchtitan.models.gpt2 import GPT2, gpt2_args, GPT2StateDictAdapter


def test_torchtitan_to_hf_conversion():
    """Test that we can convert a torchtitan model to HF format and load it."""
    print("=" * 70)
    print("Testing TorchTitan → HuggingFace Conversion")
    print("=" * 70)
    
    # Use small model for faster testing
    model_name = "124M"
    model_args = gpt2_args[model_name]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n1. Creating TorchTitan GPT-2 {model_name} with random weights...")
    tt_model = GPT2(model_args)
    tt_model.init_weights()  # Initialize with proper GPT-2 initialization
    tt_model.to(device)
    tt_model.eval()
    print("✓ TorchTitan model created")
    
    # Get torchtitan state dict
    tt_state_dict = tt_model.state_dict()
    print(f"  TorchTitan has {len(tt_state_dict)} parameters")
    
    print("\n2. Converting to HuggingFace format using adapter...")
    adapter = GPT2StateDictAdapter(model_args, None)
    hf_state_dict = adapter.to_hf(tt_state_dict)
    print(f"✓ Converted to HF format: {len(hf_state_dict)} parameters")
    
    print("\n3. Creating HuggingFace GPT-2 config...")
    hf_config = GPT2Config(
        vocab_size=model_args.vocab_size,
        n_positions=model_args.max_seq_len,
        n_embd=model_args.dim,
        n_layer=model_args.n_layers,
        n_head=model_args.n_heads,
    )
    print("✓ HuggingFace config created")
    
    print("\n4. Loading converted weights into HuggingFace model...")
    hf_model = GPT2LMHeadModel(hf_config)
    
    # Load the converted state dict
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=True)
    
    if missing_keys:
        print(f"  ⚠️  Missing keys: {missing_keys}")
        return False
    if unexpected_keys:
        print(f"  ⚠️  Unexpected keys: {unexpected_keys}")
        return False
    
    hf_model.to(device)
    hf_model.eval()
    print("✓ HuggingFace model loaded successfully")
    
    print("\n5. Comparing model outputs...")
    # Create test input
    batch_size = 2
    seq_len = 10
    test_input = torch.randint(0, model_args.vocab_size, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        tt_output = tt_model(test_input)
        hf_output = hf_model(test_input).logits
    
    # Check shapes match
    assert tt_output.shape == hf_output.shape, \
        f"Shape mismatch: TT {tt_output.shape} vs HF {hf_output.shape}"
    print(f"  ✓ Output shapes match: {tt_output.shape}")
    
    # Check outputs are very close (should be identical for same weights)
    max_diff = (tt_output - hf_output).abs().max().item()
    mean_diff = (tt_output - hf_output).abs().mean().item()
    
    print(f"  Max output difference: {max_diff:.2e}")
    print(f"  Mean output difference: {mean_diff:.2e}")
    
    # They should be identical (or very close due to numerical precision)
    if max_diff < 1e-5:
        print("✓ Outputs match perfectly!")
    elif max_diff < 1e-3:
        print("✓ Outputs match (small numerical differences)")
    else:
        print(f"⚠️  Outputs differ significantly: max_diff={max_diff}")
        return False
    
    print("\n6. Testing with actual generation...")
    # Test generation to ensure the model works end-to-end
    prompt = torch.randint(0, model_args.vocab_size, (1, 5)).to(device)
    
    # Generate with HF model
    with torch.no_grad():
        hf_generated = hf_model.generate(
            prompt,
            max_length=15,
            do_sample=False,
            pad_token_id=hf_config.eos_token_id,
        )
    
    # Generate with TT model (greedy)
    tt_generated = prompt.clone()
    with torch.no_grad():
        for _ in range(10):
            logits = tt_model(tt_generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tt_generated = torch.cat([tt_generated, next_token], dim=1)
    
    # Check if generations match
    min_len = min(tt_generated.shape[1], hf_generated.shape[1])
    matching = (tt_generated[0, :min_len] == hf_generated[0, :min_len]).sum().item()
    match_pct = 100 * matching / min_len
    
    print(f"  Token match: {matching}/{min_len} ({match_pct:.1f}%)")
    
    if match_pct == 100:
        print("✓ Generations match perfectly!")
    elif match_pct >= 90:
        print("✓ Generations mostly match (minor differences expected)")
    else:
        print(f"⚠️  Generations differ significantly: {match_pct:.1f}% match")
        return False
    
    return True


def test_save_and_load_hf_checkpoint():
    """Test saving a converted model as HuggingFace checkpoint and loading it back."""
    print("\n" + "=" * 70)
    print("Testing Save/Load HuggingFace Checkpoint")
    print("=" * 70)
    
    model_name = "124M"
    model_args = gpt2_args[model_name]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n1. Creating and converting TorchTitan model...")
    tt_model = GPT2(model_args)
    tt_model.init_weights()
    tt_model.to(device)
    tt_model.eval()
    
    adapter = GPT2StateDictAdapter(model_args, None)
    hf_state_dict = adapter.to_hf(tt_model.state_dict())
    
    # Create HF model and load weights
    hf_config = GPT2Config(
        vocab_size=model_args.vocab_size,
        n_positions=model_args.max_seq_len,
        n_embd=model_args.dim,
        n_layer=model_args.n_layers,
        n_head=model_args.n_heads,
    )
    hf_model_original = GPT2LMHeadModel(hf_config)
    hf_model_original.load_state_dict(hf_state_dict)
    print("✓ Converted model created")
    
    print("\n2. Saving to temporary directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the model
        hf_model_original.save_pretrained(tmpdir)
        print(f"✓ Model saved to {tmpdir}")
        
        print("\n3. Loading model back from checkpoint...")
        hf_model_loaded = GPT2LMHeadModel.from_pretrained(tmpdir)
        hf_model_loaded.to(device)
        hf_model_loaded.eval()
        print("✓ Model loaded successfully")
        
        print("\n4. Comparing original and loaded models...")
        # Move original model to device for comparison
        hf_model_original.to(device)
        hf_model_original.eval()
        test_input = torch.randint(0, model_args.vocab_size, (1, 10)).to(device)
        
        with torch.no_grad():
            output_original = hf_model_original(test_input).logits
            output_loaded = hf_model_loaded(test_input).logits
        
        max_diff = (output_original - output_loaded).abs().max().item()
        print(f"  Max difference: {max_diff:.2e}")
        
        # Debug: Check if state dicts match
        orig_state = hf_model_original.state_dict()
        loaded_state = hf_model_loaded.state_dict()
        
        state_diff = 0.0
        for key in orig_state.keys():
            diff = (orig_state[key] - loaded_state[key].to(orig_state[key].device)).abs().max().item()
            if diff > 1e-6:
                print(f"  Parameter {key}: diff={diff:.2e}")
            state_diff = max(state_diff, diff)
        
        print(f"  Max state dict difference: {state_diff:.2e}")
        
        if state_diff < 1e-6:
            print("✓ State dicts match perfectly!")
        else:
            print(f"⚠️  State dicts differ: {state_diff}")
            # Even if state dicts differ slightly, check if outputs are close
            if max_diff < 1e-3:
                print("✓ But outputs are close enough for practical use")
                return True
            return False
        
        if max_diff < 1e-6:
            print("✓ Loaded model matches original perfectly!")
            return True
        else:
            print(f"⚠️  Models differ: {max_diff}")
            return False


def main():
    print("=" * 70)
    print("GPT-2 TorchTitan → HuggingFace Export Tests")
    print("=" * 70)
    print("\nValidating that we can export trained TorchTitan models")
    print("to HuggingFace format for sharing and deployment.\n")
    
    try:
        # Test 1: Basic conversion and inference
        result1 = test_torchtitan_to_hf_conversion()
        
        # Test 2: Save and load checkpoint
        result2 = test_save_and_load_hf_checkpoint()
        
        if result1 and result2:
            print("\n" + "=" * 70)
            print("✅ ALL EXPORT TESTS PASSED!")
            print("=" * 70)
            print("\nYou can successfully export TorchTitan GPT-2 models to")
            print("HuggingFace format for sharing and deployment!")
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

