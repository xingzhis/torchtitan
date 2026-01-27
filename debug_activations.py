import torch
import torch.distributed.checkpoint as dcp
from torchtitan.models.qwen3 import qwen3_args, Qwen3Model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtitan.components.checkpoint import ModelWrapper
import numpy as np

# 1. Setup w/ Hooks
titan_activations = {}
hf_activations = {}

def get_titan_hook(name):
    def hook(module, input, output):
        titan_activations[name] = output.detach().float().cpu()
    return hook

def get_hf_hook(name):
    def hook(module, input, output):
         # HF often returns tuples (output, cache, ...)
        if isinstance(output, tuple):
            val = output[0]
        else:
            val = output
        hf_activations[name] = val.detach().float().cpu()
    return hook

# 2. Titan Load
print("Loading Titan...")
ckpt_path = "/gpfs/gibbs/pi/krishnaswamy_smita/cl2482/Transformer-Dispersion/transformer_dispersion/pretrain_qwen3_huggingface/checkpoints/step-29610/"
flavor = "0.6B"
model_args = qwen3_args[flavor]
with torch.device("cpu"):
    titan_model = Qwen3Model(model_args)
    wrapper = ModelWrapper(titan_model)
    dcp.load(wrapper.state_dict(), checkpoint_id=ckpt_path)
    if isinstance(wrapper.model, list):
        titan_model = wrapper.model[0]
    else:
        titan_model = wrapper.model

# Force float32 for comparison to avoid bf16 noise
titan_model = titan_model.float()

titan_model.eval()

# Register Titan Hooks
# Embeddings
titan_model.tok_embeddings.register_forward_hook(get_titan_hook("embeddings"))
# Layer 0 Attention
# Titan structure: layers['0'].attention.wq
titan_model.layers['0'].attention.wq.register_forward_hook(get_titan_hook("l0_wq"))
titan_model.layers['0'].attention.register_forward_hook(get_titan_hook("l0_attn_out"))

# 3. HF Load
print("Loading HF...")
hf_ckpt_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/pretrain_qwen3_0.6B_ckpts/step-29610/"
hf_model = AutoModelForCausalLM.from_pretrained(hf_ckpt_path, trust_remote_code=True)
hf_model = hf_model.float()
hf_model.eval()

# Register HF Hooks
# Embeddings: model.embed_tokens
hf_model.model.embed_tokens.register_forward_hook(get_hf_hook("embeddings"))
# Layer 0 Attention: model.layers[0].self_attn.q_proj
hf_model.model.layers[0].self_attn.q_proj.register_forward_hook(get_hf_hook("l0_wq"))
hf_model.model.layers[0].self_attn.register_forward_hook(get_hf_hook("l0_attn_out"))

print(f"Titan dtype: {next(titan_model.parameters()).dtype}")
print(f"HF dtype:    {next(hf_model.parameters()).dtype}")

# Register More Hooks
# Titan L0 MLP
# layers['0'].feed_forward.w2 (Down proj output)
titan_model.layers['0'].feed_forward.w2.register_forward_hook(get_titan_hook("l0_mlp_out"))
# Titan L27 (Last Layer)
titan_model.layers['27'].feed_forward.w2.register_forward_hook(get_titan_hook("l27_mlp_out"))
# Titan Final Norm
titan_model.norm.register_forward_hook(get_titan_hook("final_norm"))

# HF L0 MLP
# model.layers[0].mlp.down_proj
hf_model.model.layers[0].mlp.down_proj.register_forward_hook(get_hf_hook("l0_mlp_out"))
# HF L27
hf_model.model.layers[27].mlp.down_proj.register_forward_hook(get_hf_hook("l27_mlp_out"))
# HF Final Norm
hf_model.model.norm.register_forward_hook(get_hf_hook("final_norm"))

# 4. Run (Existing code)
tokenizer = AutoTokenizer.from_pretrained(hf_ckpt_path, trust_remote_code=True)
text = "The future" 
inputs = tokenizer(text, return_tensors="pt")
print(f"Input: {inputs.input_ids}")

with torch.no_grad():
    print("Running Titan...")
    titan_model(inputs.input_ids)
    print("Running HF...")
    hf_model(**inputs)

# 5. Compare
print("\n--- Activation Analysis ---")
check_keys = ["embeddings", "l0_wq", "l0_attn_out", "l0_mlp_out", "l27_mlp_out", "final_norm"]

for key in check_keys:
    if key not in hf_activations:
        print(f"Skipping {key} (missing in HF capture)")
        continue
        
    t = titan_activations[key]
    h = hf_activations[key]
    
    print(f"Checking {key}...")
    print(f"Titan shape: {t.shape}")
    print(f"HF shape:    {h.shape}")
    
    diff = (t - h).abs()
    print(f"Max Diff: {diff.max()}")
    print(f"Mean Diff: {diff.mean()}")
    
    if diff.max() > 1e-3:
        print("MISMATCH DETECTED")
        print("Titan slice:", t.view(-1)[:10])
        print("HF slice:   ", h.view(-1)[:10])
    else:
        print("MATCH")

print("\n--- Weight Tying Check ---")
emb_w = titan_model.tok_embeddings.weight
out_w = titan_model.output.weight
print(f"Titan Embed Weight: {emb_w.shape}")
print(f"Titan Output Weight: {out_w.shape}")
diff = (emb_w - out_w).abs().max()
print(f"Max Diff between Embed and Output in Titan: {diff}")
if diff > 1e-3:
    print("CRITICAL: Titan weights are NOT tied! HF is forcing tying, leading to mismatch.")
else:
    print("Titan weights are tied (numerically).")
