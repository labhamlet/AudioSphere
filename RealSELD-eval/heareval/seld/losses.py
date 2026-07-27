"""
Auxiliary Duplicating Permutation Invariant Training (ADPIT) loss.

A direct port of `seldnet_model.MSELoss_ADPIT`, with one addition: a frame mask,
so padded tail frames do not contribute. The 13 permutations and the per-class
per-frame minimum are unchanged.

The idea, briefly. Multi-ACCDOA predicts N tracks per class, but the reference
has no canonical track ordering: two simultaneous dogs could be assigned to
tracks in either order and both are equally correct. ADPIT enumerates every
consistent assignment of the six reference slots (A0; B0,B1; C0,C1,C2) onto the
three prediction tracks - 1 + 6 + 6 = 13 of them - and charges the model the
cheapest. "Auxiliary duplicating" is the padding trick: a frame with one source
is compared against that source duplicated across all three tracks, so tracks
that should be silent are pulled toward the active DOA rather than toward zero.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["MSELossADPIT"]


class MSELossADPIT:
    """
    output : (B, T, n_tracks*3, C) or flat (B, T, n_tracks*3*C)
    target : (B, T, 6, 4, C)   slot, (activity, x, y, z), class
    mask   : (B, T) or None
    """

    def __init__(self) -> None:
        self._each_loss = nn.MSELoss(reduction="none")

    def _each_calc(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # mean over the track*axis dimension -> (B, T, C), class-wise frame-level
        return self._each_loss(output, target).mean(dim=2)

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # activity gate * xyz, per reference slot -> (B, T, 3, C)
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # no same-class ov
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # ov with 2
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # ov with 3
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(
            output.shape[0], output.shape[1],
            target_A0A0A0.shape[2], target_A0A0A0.shape[3],
        )

        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1

        losses = torch.stack((
            self._each_calc(output, target_A0A0A0 + pad4A),
            self._each_calc(output, target_B0B0B1 + pad4B),
            self._each_calc(output, target_B0B1B0 + pad4B),
            self._each_calc(output, target_B0B1B1 + pad4B),
            self._each_calc(output, target_B1B0B0 + pad4B),
            self._each_calc(output, target_B1B0B1 + pad4B),
            self._each_calc(output, target_B1B1B0 + pad4B),
            self._each_calc(output, target_C0C1C2 + pad4C),
            self._each_calc(output, target_C0C2C1 + pad4C),
            self._each_calc(output, target_C1C0C2 + pad4C),
            self._each_calc(output, target_C1C2C0 + pad4C),
            self._each_calc(output, target_C2C0C1 + pad4C),
            self._each_calc(output, target_C2C1C0 + pad4C),
        ), dim=0)                                   # (13, B, T, C)

        # The baseline builds the same thing via argmin then a sum of 13 masked
        # terms; amin is equivalent and keeps the graph smaller.
        loss = losses.amin(dim=0)                   # (B, T, C)

        if mask is None:
            return loss.mean()
        m = mask.unsqueeze(-1)                      # (B, T, 1)
        return (loss * m).sum() / (m.sum() * loss.shape[-1]).clamp(min=1.0)