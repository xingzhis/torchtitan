import torch
import torch.distributed.checkpoint as dcp
from torchtitan.models.qwen3 import qwen3_args, Qwen3Model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
import numpy as np

# 1. Setup Paths
titan_ckpt_path = "/gpfs/gibbs/pi/krishnaswamy_smita/cl2482/Transformer-Dispersion/transformer_dispersion/pretrain_qwen3_huggingface/checkpoints/step-29610/"
hf_ckpt_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/pretrain_qwen3_0.6B_ckpts/step-29610/"
flavor = "0.6B"

print("--- Logit Parity Test ---")

# 2. Load Titan Model
print(f"Loading Titan Model ({flavor})...")
model_args = qwen3_args[flavor]
# Ensure args match config (dim=1024, etc)
print(f"Titan Config: dim={model_args.dim}, heads={model_args.n_heads}, kv={model_args.n_kv_heads}")

with torch.device("cpu"):
    titan_model = Qwen3Model(model_args)

# Load weights
state_dict = {}
dcp.load(state_dict, checkpoint_id=titan_ckpt_path)
# Titan model expects keys without 'model.' prefix sometimes depending on implementation, 
# but DCP usually loads what was saved. Let's check if we need to adjust.
# The model.load_state_dict might range based on how it was saved (FSDP vs standard).
# For simple testing, we can try loading. Titan often saves sharded.
# Using the adapter's logic normally handles conversion, but here we want raw load.
# Actually, the easiest way to load Titan model 'correctly' for inference 
# without FSDP is to use the state_dict we implicitly trust from the file.

# However, Titan's model.py doesn't have a simple 'load_dcp' helper, 
# and the keys in DCP might differ from local model if it was saved with FSDP.
# BUT, we know `convert_to_hf.py` successfully loads it into a state_dict.
# Let's use the exact loading mechanism from convert_to_hf.py
from torchtitan.components.checkpoint import ModelWrapper
titan_model = ModelWrapper(titan_model)
dcp.load(titan_model.state_dict(), checkpoint_id=titan_ckpt_path)
# dcp.load loads in-place into the state_dict of the wrapper
# titan_model was a ModelWrapper. 
# Let's inspect what titan_model.model is.
# If torchtitan's ModelWrapper.model is a list (e.g. of partitions), that would explain it.
# But Qwen3Model itself should be a module.
# Let's assume the first element if it is a list, or print type.
if isinstance(titan_model.model, list):
     titan_model = titan_model.model[0]
else:
     titan_model = titan_model.model
titan_model.eval()

# 3. Load HF Model
print(f"Loading HF Model from {hf_ckpt_path}...")
hf_model = AutoModelForCausalLM.from_pretrained(hf_ckpt_path, trust_remote_code=True)
hf_model.eval()

# 4. Prepare Input
tokenizer = AutoTokenizer.from_pretrained(hf_ckpt_path, trust_remote_code=True)
text = "The future of artificial intelligence is"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tokens: {inputs.input_ids}")

# 5. Run Inference
print("Running Forward Passes...")
with torch.no_grad():
    # Titan Forward
    # Titan forward() signature: tokens, attention_masks=None, input_batch=None
    # It constructs masks internally if None? No, let's look at model.py
    # Qwen3Model.forward(tokens, attention_masks, input_batch)
    # If we pass just tokens, it might fail if masks are needed.
    # Looking at Qwen3Model.forward: 
    #   h = self.tok_embeddings(tokens)
    #   for layer in self.layers: h = layer(h, rope, masks)
    # We need to construct the rope cache and mask manually if not provided?
    # Actually model.py handles it? 
    #   layer(h, self.rope_cache, attention_masks)
    # We should provide a causal mask.
    
    # Let's try minimal call first.
    titan_logits = titan_model(inputs.input_ids)

    # HF Forward
    import inspect
    print("Qwen3Attention Source:", inspect.getfile(hf_model.model.layers[0].self_attn.__class__))
    
    # Inspect model structure
    print("HF Model Structure (Layer 0):")

    print(hf_model.model.layers[0].self_attn)
    if hasattr(hf_model.model.layers[0].self_attn, "q_norm"):
        print("Has q_norm:", hf_model.model.layers[0].self_attn.q_norm)
        # Check if weight is not None
        if hf_model.model.layers[0].self_attn.q_norm is not None:
             print("q_norm weight slice:", hf_model.model.layers[0].self_attn.q_norm.weight[:10])
    
    hf_outputs = hf_model(**inputs)
    hf_logits = hf_outputs.logits

# 6. Compare
print("--- Comparison ---")
t_np = titan_logits.float().numpy()
h_np = hf_logits.float().numpy()

diff = np.abs(t_np - h_np)
print(f"Max Difference: {diff.max()}")
print(f"Mean Difference: {diff.mean()}")

if diff.max() < 1e-3: # loose tolerance for bfloat16/float32 mixed ops
    print("SUCCESS: Logits match!")
else:
    print("FAIL: Logits mismatch.")
    # Debug print some values
    print("Titan slice:", t_np[0, -1, :10])
    print("HF slice:   ", h_np[0, -1, :10])
