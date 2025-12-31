# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

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
        
        # Mapping from HuggingFace keys to torchtitan keys
        self.from_hf_map = {
            # Embeddings
            "transformer.wte.weight": "wte.weight",
            "transformer.wpe.weight": "wpe.weight",
            # Final layernorm
            "transformer.ln_f.weight": "ln_f.weight",
            "transformer.ln_f.bias": "ln_f.bias",
            # LM head (weight tied with wte, so map to same location)
            "lm_head.weight": "wte.weight",  # Weight tying
            # Layer-specific mappings (using {} as placeholder for layer number)
            "transformer.h.{}.ln_1.weight": "layers.{}.ln_1.weight",
            "transformer.h.{}.ln_1.bias": "layers.{}.ln_1.bias",
            "transformer.h.{}.attn.c_attn.weight": "layers.{}.attn.c_attn.weight",
            "transformer.h.{}.attn.c_attn.bias": "layers.{}.attn.c_attn.bias",
            "transformer.h.{}.attn.c_proj.weight": "layers.{}.attn.c_proj.weight",
            "transformer.h.{}.attn.c_proj.bias": "layers.{}.attn.c_proj.bias",
            "transformer.h.{}.ln_2.weight": "layers.{}.ln_2.weight",
            "transformer.h.{}.ln_2.bias": "layers.{}.ln_2.bias",
            "transformer.h.{}.mlp.c_fc.weight": "layers.{}.mlp.c_fc.weight",
            "transformer.h.{}.mlp.c_fc.bias": "layers.{}.mlp.c_fc.bias",
            "transformer.h.{}.mlp.c_proj.weight": "layers.{}.mlp.c_proj.weight",
            "transformer.h.{}.mlp.c_proj.bias": "layers.{}.mlp.c_proj.bias",
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
                        
                        # Transpose if needed (Conv1D in HF)
                        if self._should_transpose(hf_key) and value.dim() == 2:
                            hf_state_dict[hf_key] = value.t().clone()
                        else:
                            hf_state_dict[hf_key] = value.clone()
            else:
                if key in to_hf_map:
                    hf_keys = to_hf_map[key]
                    for hf_key in hf_keys:
                        hf_state_dict[hf_key] = value.clone()
        
        return hf_state_dict
    
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace state dict to torchtitan format."""
        state_dict = {}
        
        for hf_key, value in hf_state_dict.items():
            if "transformer.h." in hf_key:
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
        
        return state_dict


