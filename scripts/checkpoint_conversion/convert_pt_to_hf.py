# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper


@torch.inference_mode()
def convert_pt_to_hf(input_pt_file, output_dir, model_name, model_flavor, hf_assets_path):
    if model_name == "flux":
        import torchtitan.experiments.flux  # noqa: F401
    # load model and model args so that we can get the state dict shape
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # Load the merged PT file explicitly
    print(f"Loading merged checkpoint from {input_pt_file}...")
    loaded_state_dict = torch.load(input_pt_file, map_location="cpu", weights_only=False)

    # Load into the model wrapper (handling any strictness issues if extra keys exist)
    # ModelWrapper.load_state_dict calls set_model_state_dict with strict=False
    model.load_state_dict(loaded_state_dict)

    # Get the clean state_dict from the model
    state_dict = model.state_dict()

    # convert state dict tt->hf
    print("Converting state dict to HF format...")
    hf_state_dict = sd_adapter.to_hf(state_dict)

    print(f"Saving HF checkpoint to {output_dir}...")
    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Merged PT weights to HF format.")
    parser.add_argument(
        "input_pt_file", type=Path, help="Input .pt file (merged from DCP)."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    args = parser.parse_args()

    convert_pt_to_hf(
        args.input_pt_file,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
    )
