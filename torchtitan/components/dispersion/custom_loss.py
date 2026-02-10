# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import torch
from torchtitan.tools.logging import logger

from .dispersion import DispersionLoss


class DispersionLossWrapper:
    """
    Wraps standard loss function to add dispersion loss on hidden states.
    Matches the CustomLossTrainer from transformer_dispersion/midtrain_gpt2.py.
    """
    
    def __init__(
        self,
        base_loss_fn,
        variant: Optional[str] = None,
        dispersion_coeff: float = 1.0,
        dispersion_loc: str = "all",
        tau_l2: float = 0.5,
        tau_cos: float = 0.5,
        max_tokens: int = 256,
        accumulation_steps: int = 1,
    ):
        self.base_loss_fn = base_loss_fn
        self.variant = variant
        self.dispersion_coeff = dispersion_coeff
        self.dispersion_loc = dispersion_loc
        self.accumulation_steps = accumulation_steps

        # Check if dispersion is enabled
        self.use_disp = variant is not None and dispersion_coeff > 0.0

        if self.use_disp:
            self.disp_loss_fn = DispersionLoss(
                variant=variant,
                tau_l2=tau_l2,
                tau_cos=tau_cos,
                max_tokens=max_tokens,
            )
            logger.info(
                f"[DispersionLoss] Enabled: variant={variant}, "
                f"coeff={dispersion_coeff}, loc={dispersion_loc}"
            )
        else:
            self.disp_loss_fn = None
            logger.info("[DispersionLoss] Disabled")
        
        # Storage for hidden states (captured by hooks)
        self.hidden_states: Optional[List[torch.Tensor]] = None
        
        # Tracking for logging
        self.last_dispersion_loss = 0.0
        self.last_ce_loss = 0.0
    
    def disperse_hidden_states(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Compute dispersion loss from hidden states."""
        if self.dispersion_loc == "last":
            return self.disp_loss_fn(hidden_states[-1])
        
        # Average across all captured layers
        # Note: hidden_states are transformer block outputs, NOT embeddings
        loss_values = []
        for idx, h in enumerate(hidden_states):
            loss_values.append(self.disp_loss_fn(h))
        return torch.stack(loss_values).mean()
    
    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """Compute loss with optional dispersion component."""
        ce_loss = self.base_loss_fn(pred, labels)

        # Add dispersion only if training and enabled
        if self.use_disp and is_training and self.hidden_states is not None:
            disp_loss = self.disperse_hidden_states(self.hidden_states)
            # Fix: divide disp_loss by accumulation_steps to match the
            # rescaled CE from RescaleAccumulatedLoss, so both terms
            # contribute symmetrically to the accumulated gradient.
            total_loss = ce_loss + self.dispersion_coeff * disp_loss / self.accumulation_steps

            # Fix: log unscaled CE (undo the 1/N rescaling) so the metric
            # is comparable to baseline's global_avg_loss.
            self.last_ce_loss = ce_loss.detach().item() * self.accumulation_steps
            self.last_dispersion_loss = disp_loss.detach().item()
        else:
            total_loss = ce_loss
            self.last_ce_loss = ce_loss.detach().item() * self.accumulation_steps
            self.last_dispersion_loss = 0.0
        
        # Clear hidden states for next iteration
        self.hidden_states = None
        
        return total_loss
    
    def get_metrics(self) -> dict:
        """Get current loss metrics for logging."""
        return {
            "loss_metrics/ce_loss": self.last_ce_loss,
            "loss_metrics/dispersion_loss": self.last_dispersion_loss,
            "loss_metrics/total_with_dispersion_loss": self.last_ce_loss + self.dispersion_coeff * self.last_dispersion_loss,
        }
    
    def no_rescale(self):
        """Passthrough to base loss function's no_rescale context manager."""
        if hasattr(self.base_loss_fn, 'no_rescale'):
            return self.base_loss_fn.no_rescale()
        else:
            from contextlib import nullcontext
            return nullcontext()

