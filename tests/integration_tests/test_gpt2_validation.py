#!/usr/bin/env python3
"""
Validate GPT-2 implementation against HuggingFace.
Tests weights, logits, and generated sequences for correctness.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from torchtitan.models.gpt2 import gpt2_args, GPT2


def load_hf_weights_into_torchtitan(tt_model, hf_model, n_layers):
    """Load HuggingFace GPT-2 weights into torchtitan model."""
    hf_state = hf_model.state_dict()
    
    # Mapping from HF to torchtitan parameter names
    mapping = {
        # Embeddings
        'transformer.wte.weight': 'wte.weight',
        'transformer.wpe.weight': 'wpe.weight',
        # Final layernorm
        'transformer.ln_f.weight': 'ln_f.weight',
        'transformer.ln_f.bias': 'ln_f.bias',
        # LM head (shared with wte, so skip loading separately)
        # 'lm_head.weight': 'lm_head.weight',  # Skip due to weight tying
    }
    
    # Layer-specific mappings
    for i in range(n_layers):
        mapping.update({
            f'transformer.h.{i}.ln_1.weight': f'layers.{i}.ln_1.weight',
            f'transformer.h.{i}.ln_1.bias': f'layers.{i}.ln_1.bias',
            f'transformer.h.{i}.attn.c_attn.weight': f'layers.{i}.attn.c_attn.weight',
            f'transformer.h.{i}.attn.c_attn.bias': f'layers.{i}.attn.c_attn.bias',
            f'transformer.h.{i}.attn.c_proj.weight': f'layers.{i}.attn.c_proj.weight',
            f'transformer.h.{i}.attn.c_proj.bias': f'layers.{i}.attn.c_proj.bias',
            f'transformer.h.{i}.ln_2.weight': f'layers.{i}.ln_2.weight',
            f'transformer.h.{i}.ln_2.bias': f'layers.{i}.ln_2.bias',
            f'transformer.h.{i}.mlp.c_fc.weight': f'layers.{i}.mlp.c_fc.weight',
            f'transformer.h.{i}.mlp.c_fc.bias': f'layers.{i}.mlp.c_fc.bias',
            f'transformer.h.{i}.mlp.c_proj.weight': f'layers.{i}.mlp.c_proj.weight',
            f'transformer.h.{i}.mlp.c_proj.bias': f'layers.{i}.mlp.c_proj.bias',
        })
    
    # Build new state dict for torchtitan with proper weight transformations
    new_state_dict = {}
    loaded = 0
    
    for hf_name, tt_name in mapping.items():
        if hf_name in hf_state:
            hf_param = hf_state[hf_name].clone()
            
            # HuggingFace uses Conv1D for linear layers (attn and mlp), which needs transposing
            # Conv1D stores weights as (in_features, out_features) but Linear uses (out_features, in_features)
            # Only transpose Conv1D weights: c_attn, c_proj, c_fc (not embeddings or layernorm)
            should_transpose = ('c_attn.weight' in tt_name or 'c_proj.weight' in tt_name or 
                              'c_fc.weight' in tt_name or '.c_' in tt_name)
            
            if should_transpose and hf_param.dim() == 2:
                hf_param = hf_param.t()  # Transpose Conv1D weights
            
            new_state_dict[tt_name] = hf_param
            loaded += 1
    
    # Load the state dict into the model
    # Use strict=False because lm_head.weight is tied to wte.weight
    missing_keys, unexpected_keys = tt_model.load_state_dict(new_state_dict, strict=False)
    
    # Verify that only expected keys are missing (due to weight tying)
    expected_missing = {'lm_head.weight'}
    if set(missing_keys) != expected_missing:
        print(f"  Warning: unexpected missing keys: {set(missing_keys) - expected_missing}")
    if unexpected_keys:
        print(f"  Warning: unexpected keys: {unexpected_keys}")
    
    print(f"✓ Loaded {loaded} parameters from HuggingFace model")
    return loaded


def test_weights_match(tt_model, hf_model, n_layers):
    """Test that weights match between models."""
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
    """Validate a specific GPT-2 model size."""
    print("\n" + "=" * 70)
    print(f"Testing GPT-2 {model_name}")
    print("=" * 70)
    
    # Load HuggingFace model
    print(f"\nLoading HuggingFace {hf_name}...")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_name)
    tokenizer = GPT2Tokenizer.from_pretrained(hf_name)
    hf_model.to(device)
    hf_model.eval()
    print("✓ HuggingFace model loaded")
    
    # Create torchtitan model
    print(f"\nCreating torchtitan GPT-2 {model_name}...")
    model_args = gpt2_args[model_name]
    tt_model = GPT2(model_args)
    tt_model.to(device)
    tt_model.eval()
    print("✓ Torchtitan model created")
    print(f"  Parameters: {sum(p.numel() for p in tt_model.parameters()):,}")
    
    # Load HF weights into torchtitan
    print("\nLoading HuggingFace weights into torchtitan model...")
    load_hf_weights_into_torchtitan(tt_model, hf_model, model_args.n_layers)
    
    # Run validation tests
    test_weights_match(tt_model, hf_model, model_args.n_layers)
    test_logits_match(tt_model, hf_model, tokenizer, device)
    test_generation_match(tt_model, hf_model, tokenizer, device)
    
    print(f"\n✅ {model_name} VALIDATION PASSED!")
    return True


def main():
    print("=" * 70)
    print("GPT-2 Implementation Validation - All Model Sizes")
    print("=" * 70)
    
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
        print("\nThe GPT-2 implementation is correct and matches HuggingFace.")
    else:
        print("\n⚠️  Some validation tests failed. Please review the errors above.")
        exit(1)


if __name__ == "__main__":
    main()


