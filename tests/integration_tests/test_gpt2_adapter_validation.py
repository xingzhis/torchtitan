#!/usr/bin/env python3
"""
Validate GPT-2 implementation using torchtitan's built-in HuggingFace checkpoint loading.
This validates that the state_dict_adapter correctly converts HF weights to torchtitan format.
"""

import torch
import torch.distributed.checkpoint as dcp
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.gpt2 import gpt2_args, GPT2, GPT2StateDictAdapter


def load_hf_weights_via_adapter(tt_model, hf_model_name, model_args):
    """Load HuggingFace weights using torchtitan's state dict adapter."""
    print(f"  Loading HF weights via torchtitan adapter...")
    
    # Create adapter
    adapter = GPT2StateDictAdapter(model_args, None)
    
    # Get model state dict (empty, just for structure)
    state_dict = tt_model.state_dict()
    
    # Convert to HF format (this creates the structure that HF reader expects)
    hf_state_dict = adapter.to_hf(state_dict)
    
    # Load HF model to get its state dict
    hf_model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_actual_state = hf_model.state_dict()
    
    # Copy HF weights into the hf_state_dict structure
    for key in hf_state_dict.keys():
        if key in hf_actual_state:
            hf_state_dict[key] = hf_actual_state[key].clone()
    
    # Convert back to torchtitan format
    tt_state_dict = adapter.from_hf(hf_state_dict)
    
    # Load into model
    missing_keys, unexpected_keys = tt_model.load_state_dict(tt_state_dict, strict=False)
    
    # lm_head.weight is expected to be missing due to weight tying
    expected_missing = {'lm_head.weight'}
    if set(missing_keys) != expected_missing:
        print(f"  Warning: unexpected missing keys: {set(missing_keys) - expected_missing}")
    if unexpected_keys:
        print(f"  Warning: unexpected keys: {unexpected_keys}")
    
    print(f"✓ Loaded weights via torchtitan adapter")
    return hf_model


def test_weights_match(tt_model, hf_model, n_layers):
    """Test that weights match between models after adapter loading."""
    print("\n=== Testing Weight Matching ===")
    
    tt_state = tt_model.state_dict()
    hf_state = hf_model.state_dict()
    
    # Check embedding weights
    assert torch.allclose(tt_state['wte.weight'], hf_state['transformer.wte.weight'], atol=1e-6)
    assert torch.allclose(tt_state['wpe.weight'], hf_state['transformer.wpe.weight'], atol=1e-6)
    print("✓ Embeddings match")
    
    # Check layer weights (need to transpose HF Conv1D weights for comparison)
    for i in range(n_layers):
        # Attention weights (Conv1D in HF, Linear in TT)
        hf_attn_weight = hf_state[f'transformer.h.{i}.attn.c_attn.weight'].t()
        tt_attn_weight = tt_state[f'layers.{i}.attn.c_attn.weight']
        assert torch.allclose(tt_attn_weight, hf_attn_weight, atol=1e-6), \
            f"Layer {i} c_attn.weight mismatch"
        
        hf_proj_weight = hf_state[f'transformer.h.{i}.attn.c_proj.weight'].t()
        tt_proj_weight = tt_state[f'layers.{i}.attn.c_proj.weight']
        assert torch.allclose(tt_proj_weight, hf_proj_weight, atol=1e-6), \
            f"Layer {i} c_proj.weight mismatch"
        
        # MLP weights (Conv1D in HF, Linear in TT)
        hf_fc_weight = hf_state[f'transformer.h.{i}.mlp.c_fc.weight'].t()
        tt_fc_weight = tt_state[f'layers.{i}.mlp.c_fc.weight']
        assert torch.allclose(tt_fc_weight, hf_fc_weight, atol=1e-6), \
            f"Layer {i} mlp.c_fc.weight mismatch"
        
        hf_mlp_proj_weight = hf_state[f'transformer.h.{i}.mlp.c_proj.weight'].t()
        tt_mlp_proj_weight = tt_state[f'layers.{i}.mlp.c_proj.weight']
        assert torch.allclose(tt_mlp_proj_weight, hf_mlp_proj_weight, atol=1e-6), \
            f"Layer {i} mlp.c_proj.weight mismatch"
    
    print("✓ All layer weights match")
    print("✓ Weight matching test PASSED")


def test_logits_match(tt_model, hf_model, tokenizer, device):
    """Test that output logits match."""
    print("\n=== Testing Logits Matching ===")
    
    # Test input
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Forward pass
    with torch.no_grad():
        hf_logits = hf_model(tokens).logits
        tt_logits = tt_model(tokens)
    
    # Check shapes
    assert hf_logits.shape == tt_logits.shape, f"Shape mismatch: {hf_logits.shape} vs {tt_logits.shape}"
    print(f"✓ Output shapes match: {tt_logits.shape}")
    
    # Check logits match
    max_diff = (hf_logits - tt_logits).abs().max().item()
    mean_diff = (hf_logits - tt_logits).abs().mean().item()
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    assert torch.allclose(hf_logits, tt_logits, atol=1e-4, rtol=1e-4), f"Logits don't match! Max diff: {max_diff}"
    
    print("✓ Logits matching test PASSED")


def test_generation_match(tt_model, hf_model, tokenizer, device):
    """Test that generated sequences match."""
    print("\n=== Testing Generation Matching ===")
    
    prompt = "Once upon a time"
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate with HuggingFace
    hf_generated = hf_model.generate(
        tokens,
        max_length=30,
        do_sample=False,  # Deterministic
        temperature=1.0,
        top_k=50,
    )
    
    # Generate with torchtitan (greedy decoding)
    tt_generated = tokens.clone()
    with torch.no_grad():
        for _ in range(30 - tokens.shape[1]):
            logits = tt_model(tt_generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tt_generated = torch.cat([tt_generated, next_token], dim=1)
    
    # Decode
    hf_text = tokenizer.decode(hf_generated[0], skip_special_tokens=True)
    tt_text = tokenizer.decode(tt_generated[0], skip_special_tokens=True)
    
    print(f"  Prompt: '{prompt}'")
    print(f"  HF output: '{hf_text}'")
    print(f"  TT output: '{tt_text}'")
    
    # Check if sequences match
    if hf_text == tt_text:
        print("✓ Generated sequences match exactly")
    else:
        # Allow small differences due to numerical precision
        hf_tokens = hf_generated[0].tolist()
        tt_tokens = tt_generated[0].tolist()
        matching = sum(1 for a, b in zip(hf_tokens, tt_tokens) if a == b)
        total = len(hf_tokens)
        match_pct = 100 * matching / total
        
        print(f"  Token match: {matching}/{total} ({match_pct:.1f}%)")
        
        assert match_pct >= 95.0, f"Too many token mismatches: {match_pct:.1f}%"
        print(f"✓ Generated sequences mostly match ({match_pct:.1f}%)")


def validate_model_size(model_name, hf_name, device):
    """Validate a specific GPT-2 model size using torchtitan's adapter."""
    print("\n" + "=" * 70)
    print(f"Testing GPT-2 {model_name} (with torchtitan's state_dict_adapter)")
    print("=" * 70)
    
    # Create torchtitan model
    print(f"\nCreating torchtitan GPT-2 {model_name}...")
    model_args = gpt2_args[model_name]
    tt_model = GPT2(model_args)
    tt_model.to(device)
    tt_model.eval()
    print("✓ Torchtitan model created")
    print(f"  Parameters: {sum(p.numel() for p in tt_model.parameters()):,}")
    
    # Load HF weights via adapter
    hf_model = load_hf_weights_via_adapter(tt_model, hf_name, model_args)
    tokenizer = GPT2Tokenizer.from_pretrained(hf_name)
    hf_model.to(device)
    hf_model.eval()
    
    # Run validation tests
    test_weights_match(tt_model, hf_model, model_args.n_layers)
    test_logits_match(tt_model, hf_model, tokenizer, device)
    test_generation_match(tt_model, hf_model, tokenizer, device)
    
    print(f"\n✅ {model_name} VALIDATION PASSED (using torchtitan adapter)!")
    return True


def main():
    print("=" * 70)
    print("GPT-2 Validation - Using TorchTitan's State Dict Adapter")
    print("=" * 70)
    print("\nThis validates that the GPT2StateDictAdapter correctly converts")
    print("HuggingFace checkpoints to torchtitan format.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Map torchtitan model names to HuggingFace model names
    model_mapping = {
        "124M": "gpt2",           # GPT-2 small (124M parameters)
        "355M": "gpt2-medium",    # GPT-2 medium (355M parameters)
        "774M": "gpt2-large",     # GPT-2 large (774M parameters)
        "1558M": "gpt2-xl",       # GPT-2 XL (1558M parameters)
    }
    
    results = {}
    for model_name, hf_name in model_mapping.items():
        try:
            results[model_name] = validate_model_size(model_name, hf_name, device)
        except Exception as e:
            print(f"\n❌ {model_name} VALIDATION FAILED!")
            print(f"Error: {e}")
            results[model_name] = False
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {model_name:10s} : {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL MODEL SIZES VALIDATED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print("\nThe GPT2StateDictAdapter correctly converts HuggingFace weights!")
    else:
        print("\n⚠️  Some validation tests failed. Please review the errors above.")
        exit(1)


if __name__ == "__main__":
    main()




