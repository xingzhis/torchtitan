#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <input_dir_parent> <output_dir_parent> <python_path> <hf_assets_path> <model_name> <model_flavor>"
    exit 1
fi

input_dir_parent=$1
output_dir_parent=$2
python_path=$3
hf_assets_path=$4
model_name=$5
model_flavor=$6

# Create output directory if it doesn't exist
mkdir -p $output_dir_parent

# Generate jobfile for dSQ
jobfile="convert_pt_ckpts_jobs.txt"
rm -f $jobfile

# Find all step-* directories and create conversion commands
for step_dir in $input_dir_parent/step-*/; do
    if [ -d "$step_dir" ]; then
        step_name=$(basename $step_dir)
        # Check if merged_checkpoint.pt exists in the step directory
        if [ -f "$step_dir/merged_checkpoint.pt" ]; then
             echo "mkdir -p $output_dir_parent/$step_name && $python_path scripts/checkpoint_conversion/convert_pt_to_hf.py $step_dir/merged_checkpoint.pt $output_dir_parent/$step_name/ --model_name $model_name --model_flavor $model_flavor --hf_assets_path $hf_assets_path && cp $hf_assets_path/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt} $output_dir_parent/$step_name/ && sed -i 's/\"tie_word_embeddings\": true/\"tie_word_embeddings\": false/' $output_dir_parent/$step_name/config.json" >> $jobfile
        else
            echo "Warning: merged_checkpoint.pt not found in $step_dir, skipping."
        fi
    fi
done

echo "Generated $jobfile with $(wc -l < $jobfile) jobs"
