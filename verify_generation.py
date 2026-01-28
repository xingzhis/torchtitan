import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the newly converted checkpoint
# model_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/pretrain_qwen3_0.6B_ckpts/step-29610"
# model_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/pretrain_qwen3_0.6B_ckpts/step-29460"
# model_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/test_distcp_merging_huggingface_no_merge/step-29460"
# model_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/test_distcp_merged_hf/step-29460/"
model_path = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/test_distcp_merged_hf/step-29460/"

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
