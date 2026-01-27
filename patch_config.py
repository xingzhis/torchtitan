import json
import sys
from pathlib import Path

config_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/pretrain_qwen3_0.6B_ckpts/step-29610/config.json"

if not Path(config_path).exists():
    print(f"Error: {config_path} not found.")
    sys.exit(1)

with open(config_path, "r") as f:
    config = json.load(f)

print(f"Original tie_word_embeddings: {config.get('tie_word_embeddings', 'Not Set')}")

config["tie_word_embeddings"] = False

# Also ensure float32 loading if desired (optional, but sticking to bfloat16 is fine if conversion logic is right)
# config["torch_dtype"] = "bfloat16" 

print(f"New tie_word_embeddings: {config['tie_word_embeddings']}")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("Config updated successfully.")
