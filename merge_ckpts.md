mkdir -p <output folder for step-XXXX>
python3 -c "import torch.distributed.checkpoint.format_utils as fmt; fmt.dcp_to_torch_save('<input folder for step-XXXX>', '<output folder for step-XXXX>/merged_checkpoint.pt')"
