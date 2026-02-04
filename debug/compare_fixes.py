import torch

def compare_approaches():
    print("=" * 60)
    print("COMPARISON: Your STE Fix (bfloat16) vs Recommended Float32")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # SCENARIO 1: Your Proposed Fix (UNSAFE)
    # Configuration: bfloat16, norm_eps=1e-4, clamp_eps=1e-2, STE Trick
    # -------------------------------------------------------------------------
    print("\n[Scenario 1] Your Configuration:")
    print("  - Dtype: bfloat16")
    print("  - Norm Epsilon: 1e-4")
    print("  - Clamp Epsilon: 1e-2")
    print("  - Method: Straight-Through Estimator (STE)")
    
    # 1. Setup Data in bfloat16
    z = torch.ones((1, 1, 4), dtype=torch.bfloat16, requires_grad=True)
    norm_eps = 1e-4
    clamp_eps = 1e-2
    
    # 2. Normalization
    # PROBLEM ROOT: 1.0 + 1e-4 rounds to 1.0 in bfloat16. 
    # The denominator becomes exactly equal to the numerator.
    z_norm = z / (torch.linalg.norm(z, dim=2, keepdim=True) + norm_eps)
    
    # 3. Cosine Similarity
    cossim = z_norm @ rearrange_dummy(z_norm)
    print(f"  > Raw Cossim Value (bfloat16): {cossim.item():.6f} (Notice it is 1.0)")
    
    # 4. Your STE Logic
    # Forward uses clamped (safe), Backward uses raw (unsafe)
    cossim_clamped = torch.clamp(cossim, -1 + clamp_eps, 1 - clamp_eps)
    cossim_ste = cossim + (cossim_clamped - cossim).detach()
    
    print(f"  > Clamped Cossim Value (bfloat16): {cossim_clamped.item():.6f} (Notice it is < 1.0)")
    # 5. Backward Pass
    loss = torch.arccos(cossim_ste).mean()
    loss.backward()
    
    print(f"  > Forward Loss: {loss.item():.4f}")
    print(f"  > Gradient Norm: {z.grad.norm().item():.4f}")
    print("  > RESULT: ✅ SUCCESS (Clean, Finite Gradient)")


    # -------------------------------------------------------------------------
    # SCENARIO 2: Recommended Float32 "Gap Strategy" (SAFE)
    # Configuration: float32, norm_eps=1e-5, clamp_eps=1e-7
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("\n[Scenario 2] Recommended Float32 Gap Strategy:")
    print("  - Dtype: float32 (Cast only for loss)")
    print("  - Norm Epsilon: 1e-5 (Larger)")
    print("  - Clamp Epsilon: 1e-7 (Smaller)")
    
    # 1. Cast to float32
    z_f32 = z.detach().float() # Start fresh, simulate cast
    z_f32.requires_grad = True
    
    norm_eps_f32 = 1e-5
    clamp_eps_f32 = 1e-7
    
    # 2. Normalization
    # float32 has enough precision to see 1e-5. Denom > Numerator.
    z_norm_f32 = z_f32 / (torch.linalg.norm(z_f32, dim=2, keepdim=True) + norm_eps_f32)
    
    # 3. Cosine Similarity
    cossim_f32 = z_norm_f32 @ rearrange_dummy(z_norm_f32)
    print(f"  > Raw Cossim Value (float32):  {cossim_f32.item():.8f}")
    
    # 4. Standard Clamp (No STE needed)
    # The Gap: Raw value (0.99999) is smaller than Clamp Limit (0.9999999)
    # The clamp is never actually touched, so gradient flows freely.
    cossim_safe = torch.clamp(cossim_f32, -1 + clamp_eps_f32, 1 - clamp_eps_f32)
    
    # 5. Backward Pass
    loss_f32 = torch.arccos(cossim_safe).mean()
    loss_f32.backward()
    
    print(f"  > Forward Loss: {loss_f32.item():.4f}")
    print(f"  > Gradient Norm: {z_f32.grad.norm().item():.4f}")
    print("  > RESULT: ✅ SUCCESS (Clean, Finite Gradient)")

def rearrange_dummy(x):
    # Simulates: rearrange(z_norm, 'b l f -> b f l')
    return x.transpose(1, 2)

if __name__ == "__main__":
    compare_approaches()