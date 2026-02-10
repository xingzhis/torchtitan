"""
Test: verify dispersion loss subsampling behavior.

1. max_tokens=0 (no subsampling, i.e. old behavior) vs max_tokens=L  → should be identical
2. max_tokens=0 vs max_tokens=256 with L=4096  → show the difference
3. Real model test: GPT-2 hidden states on wikitext, full vs subsampled
"""
import numpy as np
import torch
from torchtitan.components.dispersion.dispersion import DispersionLoss


def get_random_long_text(dataset_name: str,
                         min_word_count: int = 500,
                         max_word_count: int = 700,
                         split: str = 'train',
                         random_seed: int = 0) -> str:
    from datasets import load_dataset
    if dataset_name == 'wikipedia':
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        key = 'text'
    elif dataset_name == 'pubmed':
        dataset = load_dataset("pubmed_qa", "pqa_labeled")
        key = 'long_answer'
    elif dataset_name == 'imdb':
        dataset = load_dataset("imdb")
        key = 'text'
    elif dataset_name == 'squad':
        dataset = load_dataset("squad")
        key = 'context'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    rng = np.random.default_rng(seed=random_seed)
    idx = rng.integers(0, int(len(dataset[split]) * 0.95)).item()
    text = ''
    while len(text.split()) < min_word_count:
        text += dataset[split][idx][key]
        idx += 1
        if len(text.split()) > max_word_count:
            break
    return text


def test_no_subsampling_matches(device="cuda"):
    """max_tokens=0 and max_tokens>=L should give identical results."""
    B, L, F = 2, 512, 64
    torch.manual_seed(42)
    z = torch.randn(B, L, F, device=device)

    loss_fn_disabled = DispersionLoss(variant="angular_spread", max_tokens=0)
    loss_fn_large = DispersionLoss(variant="angular_spread", max_tokens=L)

    # Both should skip subsampling → identical output
    loss_disabled = loss_fn_disabled(z)
    loss_large = loss_fn_large(z)

    diff = (loss_disabled - loss_large).abs().item()
    print(f"[Test 1] max_tokens=0 vs max_tokens={L}  (L={L})")
    print(f"  loss (disabled) = {loss_disabled.item():.8f}")
    print(f"  loss (max={L})  = {loss_large.item():.8f}")
    print(f"  abs diff        = {diff:.2e}")
    assert diff < 1e-6, f"Expected identical, got diff={diff}"
    print("  PASSED\n")


def test_subsampling_vs_full(device="cuda"):
    """Compare full pairwise (max_tokens=0) vs subsampled (max_tokens=256) at L=4096."""
    B, L, F = 2, 4096, 64
    max_tok = 256
    n_trials = 5

    torch.manual_seed(0)
    z = torch.randn(B, L, F, device=device)

    loss_fn_full = DispersionLoss(variant="angular_spread", max_tokens=0)
    loss_fn_sub = DispersionLoss(variant="angular_spread", max_tokens=max_tok)

    loss_full = loss_fn_full(z).item()

    # Run subsampled version multiple times (different random subsets)
    losses_sub = []
    for i in range(n_trials):
        torch.manual_seed(i + 100)
        losses_sub.append(loss_fn_sub(z).item())

    mean_sub = sum(losses_sub) / len(losses_sub)
    std_sub = (sum((x - mean_sub) ** 2 for x in losses_sub) / len(losses_sub)) ** 0.5

    print(f"[Test 2] Full (L={L}) vs subsampled (max_tokens={max_tok}), {n_trials} trials")
    print(f"  full loss          = {loss_full:.8f}")
    print(f"  subsampled mean    = {mean_sub:.8f}")
    print(f"  subsampled std     = {std_sub:.8f}")
    print(f"  relative error     = {abs(mean_sub - loss_full) / abs(loss_full) * 100:.2f}%")
    print(f"  individual trials  = {['%.8f' % x for x in losses_sub]}")
    print()


def test_gradient_flows(device="cuda"):
    """Verify gradients flow correctly with subsampling."""
    B, L, F = 2, 1024, 64
    torch.manual_seed(42)
    z = torch.randn(B, L, F, device=device, requires_grad=True)

    loss_fn = DispersionLoss(variant="angular_spread", max_tokens=128)
    loss = loss_fn(z)
    loss.backward()

    assert z.grad is not None, "No gradient computed"
    grad_norm = z.grad.norm().item()
    print(f"[Test 3] Gradient check (max_tokens=128, L={L})")
    print(f"  loss      = {loss.item():.8f}")
    print(f"  grad norm = {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient is zero"
    print("  PASSED\n")


def test_real_model_subsampling(device="cuda"):
    """Test subsampling on real GPT-2 hidden states from wikitext."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("[Test 4] Real model: GPT-2 hidden states on wikitext")
    print("  Loading text...")
    text = get_random_long_text('wikipedia', min_word_count=1200, max_word_count=1500)
    print(f"  Text length: {len(text.split())} words")

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024,
              add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    L_actual = input_ids.shape[1]
    print(f"  Tokenized: L={L_actual}")

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = [h.detach() for h in out.hidden_states]  # list of [1, L, 768]
    print(f"  Layers: {len(hidden_states)}, shape per layer: {hidden_states[0].shape}")

    loss_fn_full = DispersionLoss(variant="angular_spread", max_tokens=0)
    loss_fn_sub = DispersionLoss(variant="angular_spread", max_tokens=256)

    n_trials = 10
    for layer_idx in [0, len(hidden_states) // 2, -1]:
        z = hidden_states[layer_idx]
        layer_name = {0: "first", len(hidden_states) // 2: "middle", -1: "last"}[layer_idx]

        loss_full = loss_fn_full(z).item()

        losses_sub = []
        for i in range(n_trials):
            torch.manual_seed(i)
            losses_sub.append(loss_fn_sub(z).item())

        mean_sub = sum(losses_sub) / len(losses_sub)
        std_sub = (sum((x - mean_sub) ** 2 for x in losses_sub) / len(losses_sub)) ** 0.5
        rel_err = abs(mean_sub - loss_full) / abs(loss_full) * 100

        print(f"\n  Layer {layer_idx} ({layer_name}), L={z.shape[1]}, F={z.shape[2]}:")
        print(f"    full (max_tokens=0)   = {loss_full:.10f}")
        print(f"    subsampled mean       = {mean_sub:.10f}  ({n_trials} trials)")
        print(f"    subsampled std        = {std_sub:.10f}")
        print(f"    relative error        = {rel_err:.4f}%")

    # Also test max_tokens=L gives exact match on real data
    z = hidden_states[-1]
    loss_full = loss_fn_full(z).item()
    loss_fn_L = DispersionLoss(variant="angular_spread", max_tokens=z.shape[1])
    loss_L = loss_fn_L(z).item()
    diff = abs(loss_full - loss_L)
    print(f"\n  Exact match check (max_tokens=L={z.shape[1]} vs disabled):")
    print(f"    diff = {diff:.2e}  [{'PASS' if diff < 1e-6 else 'FAIL'}]")
    assert diff < 1e-6
    print()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    test_no_subsampling_matches(device)
    test_gradient_flows(device)
    test_subsampling_vs_full(device)
    test_real_model_subsampling(device)
