import torch

def compare_approaches(norm_eps, clamp_eps, z_dim):
    # 1. Setup Data in bfloat16
    z_orig1 = torch.ones((1, 1, z_dim), dtype=torch.bfloat16, requires_grad=True)
    z_orig2 = torch.ones((1, 1, z_dim), dtype=torch.bfloat16) + 0.1 * torch.randn((1, 1, z_dim), dtype=torch.bfloat16)
    z_orig2.requires_grad = True
    z1 = z_orig1.float()
    z2 = z_orig2.float()
    # 2. Normalization
    # z_norm = z / (torch.linalg.norm(z, dim=2, keepdim=True) + norm_eps)
    z_norm1 = torch.nn.functional.normalize(z1, p=2, dim=2, eps=norm_eps) 
    z_norm2 = torch.nn.functional.normalize(z2, p=2, dim=2, eps=norm_eps) 
    
    # 3. Cosine Similarity
    cossim = z_norm1 @ rearrange_dummy(z_norm2)
    print(f"  > Raw Cossim Value (bfloat16): {cossim.item():.6f}")
    
    # 4. Your STE Logic
    cossim_clamped = torch.clamp(cossim, -1 + clamp_eps, 1 - clamp_eps)
    
    print(f"  > Clamped Cossim Value (bfloat16): {cossim_clamped.item():.6f}")
    # 5. Backward Pass
    loss = torch.arccos(cossim_clamped).mean()
    print(f"  > Loss: {loss.item():.4f}")
    loss.backward()
    
    print(f"  > Forward Loss: {loss.item():.4f}")
    print(f"  > Gradient Norm: {z_orig1.grad.norm().item():.4f}")
    print(f"  > Gradient Norm: {z_orig2.grad.norm().item():.4f}")

def rearrange_dummy(x):
    return x.transpose(1, 2)

if __name__ == "__main__":
    print("norm_eps: 1e-2, clamp_eps: 1e-2, z_dim: 2")
    compare_approaches(1e-2, 1e-2, 2)
    print("norm_eps: 1e-4, clamp_eps: 1e-2, z_dim: 2")
    compare_approaches(1e-4, 1e-2, 2)
    print("norm_eps: 1e-4, clamp_eps: 1e-4, z_dim: 2")
    compare_approaches(1e-4, 1e-4, 2)
    print("norm_eps: 1e-5, clamp_eps: 1e-7, z_dim: 10")
    compare_approaches(1e-5, 1e-7, 10)

    