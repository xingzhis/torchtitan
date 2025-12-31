```
conda activate torchtitan
python3 scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --assets tokenizer 
# pretrain on c4
NGPU=1 CONFIG_FILE="./test_dispersion.toml" ./run_train.sh
# midtrain on wikitext for ~200M tokens
python3 scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B-Base --assets tokenizer safetensors config
python3 scripts/checkpoint_conversion/convert_from_hf.py \
  ./assets/hf/Qwen3-0.6B-Base \
  ./assets/tt/Qwen3-0.6B-Base \
  --model_name qwen3 \
  --model_flavor 0.6B
NGPU=1 CONFIG_FILE="./test_dispersion_midtraining.toml" ./run_train.sh
```