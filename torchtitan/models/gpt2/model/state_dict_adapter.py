# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

import torch

logger = logging.getLogger()

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import GPT2ModelArgs


class GPT2StateDictAdapter(StateDictAdapter):
    """State dict adapter for converting between HuggingFace GPT-2 and torchtitan formats.
    
    Main differences:
    - HuggingFace uses Conv1D (transposed Linear) for attention and MLP projections
    - Parameter name differences (transformer.h vs layers, etc.)
    - Weight tying between wte and lm_head
    """
    
    def __init__(
        self,
        model_args: GPT2ModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

        # Ensure fqn_to_index_mapping is set (needed by convert_to_hf.py)
        # For GPT-2, we don't use sharded checkpoints, so this is None
        if not hasattr(self, 'fqn_to_index_mapping'):
            self.fqn_to_index_mapping = None
        
        # Mapping from HuggingFace keys to torchtitan keys
        # GPT-2 HF format doesn't use 'transformer.' prefix like other models
        self.from_hf_map = {
            # Embeddings
            "wte.weight": "wte.weight",
            "wpe.weight": "wpe.weight",
            # Final layernorm
            "ln_f.weight": "ln_f.weight",
            "ln_f.bias": "ln_f.bias",
            # LM head (weight tied with wte, so map to same location)
            "lm_head.weight": "wte.weight",  # Weight tying
            # Layer-specific mappings (using {} as placeholder for layer number)
            "h.{}.ln_1.weight": "layers.{}.ln_1.weight",
            "h.{}.ln_1.bias": "layers.{}.ln_1.bias",
            "h.{}.attn.c_attn.weight": "layers.{}.attn.c_attn.weight",
            "h.{}.attn.c_attn.bias": "layers.{}.attn.c_attn.bias",
            "h.{}.attn.c_proj.weight": "layers.{}.attn.c_proj.weight",
            "h.{}.attn.c_proj.bias": "layers.{}.attn.c_proj.bias",
            "h.{}.ln_2.weight": "layers.{}.ln_2.weight",
            "h.{}.ln_2.bias": "layers.{}.ln_2.bias",
            "h.{}.mlp.c_fc.weight": "layers.{}.mlp.c_fc.weight",
            "h.{}.mlp.c_fc.bias": "layers.{}.mlp.c_fc.bias",
            "h.{}.mlp.c_proj.weight": "layers.{}.mlp.c_proj.weight",
            "h.{}.mlp.c_proj.bias": "layers.{}.mlp.c_proj.bias",
        }
    
    def _should_transpose(self, key: str) -> bool:
        """Check if a weight tensor should be transposed.
        
        HuggingFace uses Conv1D for attention and MLP projections, which stores
        weights as (in_features, out_features). PyTorch Linear uses (out_features, in_features).
        So we need to transpose these specific weight tensors.
        """
        # Only transpose weight tensors (not biases) for Conv1D layers
        return "weight" in key and any(x in key for x in ["c_attn", "c_proj", "c_fc"])
    
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HuggingFace format."""
        # Reverse the mapping
        to_hf_map = {}
        for hf_key, tt_key in self.from_hf_map.items():
            if tt_key not in to_hf_map:
                to_hf_map[tt_key] = []
            to_hf_map[tt_key].append(hf_key)
        
        hf_state_dict = {}
        
        for key, value in state_dict.items():
            if "layers" in key:
                # Extract layer number and create abstract key
                abstract_key = re.sub(r"\.(\d+)\.", ".{}.", key, count=1)
                layer_num = re.search(r"layers\.(\d+)\.", key).group(1)
                
                if abstract_key in to_hf_map:
                    hf_keys = to_hf_map[abstract_key]
                    for hf_key_template in hf_keys:
                        hf_key = hf_key_template.format(layer_num)

                        # GPT-2 HF format doesn't use 'transformer.' prefix like other models
                        # Remove the transformer. prefix for GPT-2
                        if hf_key.startswith("transformer."):
                            hf_key = hf_key[len("transformer."):]

                        # Transpose if needed (Conv1D in HF)
                        if self._should_transpose(hf_key) and value.dim() == 2:
                            hf_state_dict[hf_key] = value.t().contiguous()
                        else:
                            hf_state_dict[hf_key] = value.contiguous() if value.dim() == 2 else value.clone()
            else:
                if key in to_hf_map:
                    hf_keys = to_hf_map[key]
                    for hf_key in hf_keys:
                        # Skip lm_head.weight - GPT-2 always has weight tying
                        # In HF checkpoints, lm_head.weight is tied to transformer.wte.weight
                        # and doesn't exist separately, so DCP will fail if we include it
                        if hf_key == "lm_head.weight":
                            continue

                        # GPT-2 HF format doesn't use 'transformer.' prefix like other models
                        # Remove the transformer. prefix for GPT-2
                        if hf_key.startswith("transformer."):
                            hf_key = hf_key[len("transformer."):]

                        hf_state_dict[hf_key] = value.contiguous() if value.dim() == 2 else value.clone()

        # GPT-2 has weight tying: lm_head.weight == wte.weight
        # In HF checkpoints, lm_head.weight doesn't exist separately (it's tied)
        # So we never include lm_head.weight in the structure returned by to_hf()
        # The from_hf() method handles this by mapping both to wte.weight
        
        return hf_state_dict
    
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace state dict to torchtitan format."""
        state_dict = {}
        
        for hf_key, value in hf_state_dict.items():
            if "h." in hf_key and re.search(r"h\.(\d+)\.", hf_key):
                # Extract layer number and create abstract key
                abstract_key = re.sub(r"\.(\d+)\.", ".{}.", hf_key, count=1)
                layer_num = re.search(r"h\.(\d+)\.", hf_key).group(1)
                
                if abstract_key in self.from_hf_map:
                    tt_key = self.from_hf_map[abstract_key].format(layer_num)
                    
                    # Transpose if needed (Conv1D in HF -> Linear in torchtitan)
                    if self._should_transpose(hf_key) and value.dim() == 2:
                        state_dict[tt_key] = value.t().clone()
                    else:
                        state_dict[tt_key] = value.clone()
            else:
                if hf_key in self.from_hf_map:
                    tt_key = self.from_hf_map[hf_key]
                    state_dict[tt_key] = value.clone()
        
        # Handle weight tying: GPT-2 has weight tying between wte.weight and lm_head.weight
        # In HF checkpoints, lm_head.weight doesn't exist separately (it's tied)
        # But TorchTitan model expects lm_head.weight in the state dict
        # So we create lm_head.weight as a reference to wte.weight
        if "wte.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["wte.weight"]

        return state_dict



