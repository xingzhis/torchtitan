"""Apply parallelizations to GPT-2 model."""

import torch
import torch.nn as nn

from torch.distributed import DeviceMesh
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.tools.logging import logger


def apply_tp(model: nn.Module, tp_mesh: DeviceMesh):
    """Apply tensor parallelism to GPT-2."""
    for layer_id, layer in model.layers.items():
        # Parallelize attention
        parallelize_module(
            layer.attn.c_attn,
            tp_mesh,
            ColwiseParallel(input_layouts=Replicate()),
        )
        parallelize_module(
            layer.attn.c_proj,
            tp_mesh,
            RowwiseParallel(output_layouts=Replicate()),
        )
        
        # Parallelize MLP
        parallelize_module(
            layer.mlp.c_fc,
            tp_mesh,
            ColwiseParallel(input_layouts=Replicate()),
        )
        parallelize_module(
            layer.mlp.c_proj,
            tp_mesh,
            RowwiseParallel(output_layouts=Replicate()),
        )


def parallelize_gpt2(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """Apply parallelisms to GPT-2 model."""
    world_mesh = parallel_dims.world_mesh
    
    # Tensor parallelism
    if parallel_dims.tp_enabled:
        apply_tp(model, world_mesh["tp"])
    
    # Activation checkpointing
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            layers=model.layers,
        )
    
    # torch.compile
    if job_config.compile.enable and "model" in job_config.compile.components:
        model = torch.compile(model)
    
    # FSDP / DDP
    if parallel_dims.dp_enabled:
        if parallel_dims.dp_shard_enabled:
            # FSDP
            mp_policy = MixedPrecisionPolicy(
                param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            )
            
            # Reshard layers
            for layer in model.layers.values():
                fully_shard(
                    layer,
                    mesh=world_mesh["dp"],
                    mp_policy=mp_policy,
                    reshard_after_forward=True,
                )
            
            # Shard full model
            fully_shard(
                model,
                mesh=world_mesh["dp"],
                mp_policy=mp_policy,
                reshard_after_forward=False,
            )
        else:
            # DDP
            replicate(model, device_mesh=world_mesh["dp"])
    
    logger.info("Applied parallelizations to GPT-2 model")
    return model

