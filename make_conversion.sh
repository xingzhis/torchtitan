## dispersion
# bash ./convert_ckpts.sh \
#   /gpfs/gibbs/pi/krishnaswamy_smita/cl2482/Transformer-Dispersion/transformer_dispersion/pretrain_qwen3_huggingface/checkpoints \
#   /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/pretrain_qwen3_0.6B_ckpts/ \
#   /gpfs/gibbs/project/krishnaswamy_smita/xs272/conda_envs/torchtitan/bin/python3 \
#   ./assets/hf/Qwen3-0.6B/ \
#   qwen3 \
#   0.6B

## baseline
bash ./convert_ckpts.sh \
  /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/qwen3_0.6B_baseline_torchtitan \
  /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/qwen3_0.6B_baseline_huggingface \
  /gpfs/gibbs/project/krishnaswamy_smita/xs272/conda_envs/torchtitan/bin/python3 \
  ./assets/hf/Qwen3-0.6B/ \
  qwen3 \
  0.6B

## test_distcp_merging

bash ./convert_ckpts.sh \
  /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/test_distcp_merging \
  /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/test_distcp_merging_huggingface_no_merge \
  /gpfs/gibbs/project/krishnaswamy_smita/xs272/conda_envs/torchtitan/bin/python3 \
  ./assets/hf/Qwen3-0.6B/ \
  qwen3 \
  0.6B

