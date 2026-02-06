import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Load and test a causal language model.")
parser.add_argument("model_path", type=str, help="Path to the model directory or HuggingFace model ID")
args = parser.parse_args()

model_path = args.model_path
print(f"Loading model from {model_path}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# Test prompt
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n--- Generating Output ---")
output_tokens = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)
