"""
GPT-2 model implementation for torchtitan.
Adapted from nanoGPT by Andrej Karpathy.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.protocols.model import ModelProtocol

from .args import GPT2ModelArgs


class CausalSelfAttention(nn.Module):
    def __init__(self, args: GPT2ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        # Query, key, value projections
        self.c_attn = nn.Linear(args.dim, 3 * args.dim, bias=args.bias)
        # Output projection
        self.c_proj = nn.Linear(args.dim, args.dim, bias=args.bias)
        # Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        
    def init_weights(self):
        # GPT-2 style initialization
        nn.init.normal_(self.c_attn.weight, std=0.02)
        nn.init.normal_(self.c_proj.weight, std=0.02)
        if self.c_attn.bias is not None:
            nn.init.zeros_(self.c_attn.bias)
            nn.init.zeros_(self.c_proj.bias)
    
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=2)
        
        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, args: GPT2ModelArgs):
        super().__init__()
        self.c_fc = nn.Linear(args.dim, 4 * args.dim, bias=args.bias)
        self.c_proj = nn.Linear(4 * args.dim, args.dim, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
        
    def init_weights(self):
        nn.init.normal_(self.c_fc.weight, std=0.02)
        nn.init.normal_(self.c_proj.weight, std=0.02)
        if self.c_fc.bias is not None:
            nn.init.zeros_(self.c_fc.bias)
            nn.init.zeros_(self.c_proj.bias)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')  # GPT-2 uses approximate GELU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, args: GPT2ModelArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(args.dim, eps=1e-5)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = nn.LayerNorm(args.dim, eps=1e-5)
        self.mlp = MLP(args)
        
    def init_weights(self):
        # LayerNorm already initialized correctly by default
        self.attn.init_weights()
        self.mlp.init_weights()
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module, ModelProtocol):
    def __init__(self, model_args: GPT2ModelArgs):
        super().__init__()
        self.model_args = model_args
        
        # Token and position embeddings
        self.wte = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.wpe = nn.Embedding(model_args.max_seq_len, model_args.dim)
        self.drop = nn.Dropout(model_args.dropout)
        
        # Transformer blocks (as ModuleDict for pipeline parallelism)
        self.layers = nn.ModuleDict({
            str(i): Block(model_args) for i in range(model_args.n_layers)
        })
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(model_args.dim, eps=1e-5)
        
        # Language model head
        self.lm_head = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        
        # Weight tying (share embeddings with output layer)
        self.wte.weight = self.lm_head.weight
    
    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize weights following GPT-2 scheme."""
        # Embeddings
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.01)
        
        # Transformer blocks
        for layer in self.layers.values():
            layer.init_weights()
        
        # LayerNorm (already initialized correctly by default)
        # Note: lm_head shares weights with wte, so no separate init needed
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.model_args.n_layers))
    
    def forward(self, tokens, targets=None):
        B, T = tokens.size()
        assert T <= self.model_args.max_seq_len, f"Sequence length {T} exceeds maximum {self.model_args.max_seq_len}"
        
        # Token embeddings + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
        tok_emb = self.wte(tokens)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for layer in self.layers.values():
            x = layer(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        return logits
    
    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ):
        # GPT-2 uses standard causal masking, handled in attention
        raise NotImplementedError("GPT-2 uses built-in causal masking")


