import torch
import torch.distributed.checkpoint as dcp
from torchtitan.models.qwen3 import qwen3_args, Qwen3Model
from transformers import AutoTokenizer
from torchtitan.components.checkpoint import ModelWrapper

# 1. Setup
ckpt_path = "/gpfs/gibbs/pi/krishnaswamy_smita/cl2482/Transformer-Dispersion/transformer_dispersion/pretrain_qwen3_huggingface/checkpoints/step-29610/"
hf_assets_path = "./assets/hf/Qwen3-0.6B" # For tokenizer
flavor = "0.6B"
params = qwen3_args[flavor]

print(f"Loading Titan Model ({flavor}) from {ckpt_path}...")

# 2. Init Model
with torch.device("cpu"):
    # TorchTitan models are often wrapped for FSDP saving, so we use ModelWrapper for loading if needed
    # But Qwen3Model is the underlying module. 
    # dcp.load expects a state_dict.
    model = Qwen3Model(params)
    
    # We need to match the structure expected by DCP. 
    # Usually convert_to_hf.py does:
    # model = ModelWrapper(model)
    # dcp.load(model.state_dict(), ...)
    wrapper = ModelWrapper(model)
    state_dict = wrapper.state_dict()
    
    dcp.load(state_dict, checkpoint_id=ckpt_path)
    
    # Unwrap
    if isinstance(wrapper.model, list):
        model = wrapper.model[0]
    else:
        model = wrapper.model

model.eval()
print("Model loaded.")

# 3. Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_assets_path, trust_remote_code=True)

# 4. Generation Loop
prompt_text = "The future of artificial intelligence is"
#prompt_text = "aaaaaaaa"
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

print(f"Prompt: {prompt_text}")
print(f"Input IDs: {input_ids}")

max_new_tokens = 10
generated = input_ids

print("Generating...")
with torch.no_grad():
    for i in range(max_new_tokens):
        # Forward pass
        # model.forward(tokens)
        logits = model(generated)
        
        # Get last token logits
        next_token_logits = logits[0, -1, :]
        
        # Greedy decode
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # Print progress
        print(f"{i}: {next_token.item()}")

# 5. Decode
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("\n--- Titan Generation Output ---")
print(output_text)
print("-------------------------------")
