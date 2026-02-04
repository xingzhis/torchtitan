
import torch

def test_acos_gradient_stability():
    print("Testing bfloat16 precision with 1e-4 epsilon...")
    
    # Simulate the value being exactly 1.0 (perfect correlation)
    x = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
    epsilon = 1e-2
    
    # In bfloat16, 1.0 - 1e-4 rounds back to 1.0 because machine epsilon is ~0.0078
    # 1e-4 is much smaller than machine epsilon
    upper_bound = 1.0 - epsilon
    print(f"1.0 - {epsilon} in float32: {1.0 - epsilon:.8f}")
    print(f"1.0 - {epsilon} cast to bfloat16: {torch.tensor(1.0 - epsilon, dtype=torch.bfloat16).item():.8f}")
    
    # Forward pass
    x_clamped = torch.clamp(x, -1 + epsilon, 1 - epsilon)
    print(f"Clamped value: {x_clamped.item()}")
    
    y = torch.arccos(x_clamped)
    print(f"arccos output: {y.item()}")
    
    # Backward pass
    try:
        y.backward()
        print(f"Gradient: {x.grad.item()}")
    except RuntimeError as e:
        print(f"Backward failed: {e}")
        
    if x.grad is not None and (torch.isnan(x.grad) or torch.isinf(x.grad)):
        print("\nCONCLUSION: Gradient is NaN/Inf! The epsilon was too small for bfloat16, so checking against 1.0 failed.")

def test_normalization_stability():
    print("\nTesting normalization stability (z / (norm + eps)) with bfloat16...")
    epsilon = 1e-4
    
    # Case 1: Zero vector. norm is 0. Denominator should be epsilon.
    z_zero = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    denom_zero = torch.linalg.norm(z_zero, dim=2, keepdim=True) + epsilon
    print(f"Zero vector: norm + {epsilon} = {denom_zero.item()}")
    
    if denom_zero.item() == 0.0:
        print("CRITICAL: Denominator became 0! Division by zero will happen.")
    else:
        print("SAFE: Denominator is non-zero.")
        
    # Case 2: Vector with norm 1.0. 
    z_one = torch.ones(1, 1, 4, dtype=torch.bfloat16) * 0.5 # norm is 1.0
    denom_one = torch.linalg.norm(z_one, dim=2, keepdim=True) + epsilon
    print(f"Unit vector: norm(1.0) + {epsilon} = {denom_one.item()}")
    if denom_one.item() == 1.0:
        print("INFO: Epsilon was absorbed into 1.0 (precision loss), but this is usually safe for normalization.")
    else:
        print("INFO: Epsilon is preserved.")

if __name__ == "__main__":
    test_acos_gradient_stability()
    test_normalization_stability()
