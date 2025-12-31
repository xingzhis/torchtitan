from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs


@dataclass
class GPT2ModelArgs(BaseModelArgs):
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dropout: float = 0.0
    bias: bool = True  # GPT-2 uses bias

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        # Count total parameters
        n_params = sum(p.numel() for p in model.parameters())
        
        # Estimate FLOPs (simplified)
        # Forward pass: ~6 * n_params * seq_len (attention + MLP)
        flops_per_token = 6 * n_params
        flops = flops_per_token * seq_len
        
        return n_params, flops


