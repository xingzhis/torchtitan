from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_gpt2
from .model.args import GPT2ModelArgs
from .model.model import GPT2
from .model.state_dict_adapter import GPT2StateDictAdapter


# GPT-2 model configurations
gpt2_args = {
    "debugmodel": GPT2ModelArgs(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=50257,
        max_seq_len=512,
    ),
    "124M": GPT2ModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        vocab_size=50257,
        max_seq_len=1024,
    ),
    "355M": GPT2ModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=16,
        vocab_size=50257,
        max_seq_len=1024,
    ),
    "774M": GPT2ModelArgs(
        dim=1280,
        n_layers=36,
        n_heads=20,
        vocab_size=50257,
        max_seq_len=1024,
    ),
    "1558M": GPT2ModelArgs(
        dim=1600,
        n_layers=48,
        n_heads=25,
        vocab_size=50257,
        max_seq_len=1024,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=GPT2,
        model_args=gpt2_args,
        parallelize_fn=parallelize_gpt2,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=GPT2StateDictAdapter,
    )


