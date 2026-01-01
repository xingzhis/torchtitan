```
conda activate torchtitan
# midtrain on wikitext for ~200M tokens
python3 scripts/download_hf_assets.py --repo_id openai-community/gpt2 --assets tokenizer safetensors config
python3 scripts/checkpoint_conversion/convert_from_hf.py \
  ./assets/hf/gpt2 \
  ./assets/tt/gpt2 \
  --model_name gpt2 \
  --model_flavor 124M
NGPU=1 CONFIG_FILE="./test_midtrain_gpt2.toml" ./run_train.sh
```