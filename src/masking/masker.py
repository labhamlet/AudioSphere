from typing import Tuple, Optional
from torch import nn
import torch

from .utils import (
    generate_masks_batch
)

class SpatialMaskMaker(nn.Module):
    def __init__(
        self,
        mask_patch: int = 10,
        context_cluster: bool = False,
        mask_mode: Optional[str] = None,   # "random" | "cluster" | "time" | "freq"
        n_freq_patches: int = 8,           # = model.p_f_dim
        p_t_dim: Optional[int] = None,     # = model.p_t_dim (cluster mode only)
    ):
        super().__init__()  # type: ignore
        self.mask_patch = mask_patch
        self.context_cluster = context_cluster
        self.mask_mode = mask_mode
        self.n_freq_patches = n_freq_patches
        self.p_t_dim = p_t_dim

    def forward(
        self,
        local_features: Optional[torch.Tensor],
        batch_size: Optional[int],
        n_times: Optional[int],
    ) -> torch.Tensor:
        """Returns (batch_size, n_times) bool mask, True = masked."""
        if local_features is not None:
            batch_size, n_times, _ = local_features.size()
        return generate_masks_batch(
            B=batch_size,
            sequence_len=n_times,
            mask_patch=self.mask_patch,
            cluster_ctx=self.context_cluster,
            mask_mode=self.mask_mode,
            n_freq_patches=self.n_freq_patches,
            p_t_dim=self.p_t_dim,
        )