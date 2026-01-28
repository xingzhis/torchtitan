## test_distcp_merging (merged pt workflow)

bash ./convert_pt_ckpts.sh \
  /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/test_distcp_merged \
  /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/test_distcp_merged_hf \
  /gpfs/gibbs/project/krishnaswamy_smita/xs272/conda_envs/torchtitan/bin/python3 \
  ./assets/hf/Qwen3-0.6B/ \
  qwen3 \
  0.6B
