import random
from typing import List, Optional
 
import torch
 
def _line_token_indices(line_idx: int, line_type: str, T: int, F: int) -> List[int]:
    if line_type == "time":                 # one time column, all freq rows
        return [f * T + line_idx for f in range(F)]
    return [line_idx * T + t for t in range(T)]   # one freq row, all times
 
 
def gen_maskid_structured(T: int, F: int, mask_patch: int, line_type: str) -> torch.Tensor:
    """Full time-columns or freq-rows totalling EXACTLY mask_patch tokens
    (one final line partially masked when the budget is not divisible)."""
    n_lines = T if line_type == "time" else F
    line_size = F if line_type == "time" else T
    if not (0 < mask_patch <= T * F):
        raise ValueError(f"mask_patch={mask_patch} out of range for {T*F} tokens")
    k_full, rem = divmod(mask_patch, line_size)
    n_pick = k_full + (1 if rem > 0 else 0)
    if n_pick > n_lines:
        raise ValueError(f"mask_patch={mask_patch} needs {n_pick} {line_type} lines; grid has {n_lines}")
    lines = random.sample(range(n_lines), n_pick)
    ids: List[int] = []
    for ln in lines[:k_full]:
        ids.extend(_line_token_indices(ln, line_type, T, F))
    if rem > 0:
        ids.extend(random.sample(_line_token_indices(lines[-1], line_type, T, F), rem))
    return torch.tensor(ids, dtype=torch.long)
 
 
# --------------------------------------------------------------------------- #
# Batch generator — extends your generate_masks_batch
# --------------------------------------------------------------------------- #
 
def generate_masks_batch(
    B: int,
    sequence_len: int,
    mask_patch: int,
    cluster_ctx: bool = False,
    mask_mode: Optional[str] = None,     # None -> legacy behavior via cluster_ctx
    n_freq_patches: int = 8,
    p_t_dim: Optional[int] = None,       # required for "cluster"
) -> torch.Tensor:
    """(B, sequence_len) bool mask, True = masked, exactly mask_patch per row."""
    if mask_mode is None:
        mask_mode = "cluster" if cluster_ctx else "random"
 
    mask = torch.zeros((B, sequence_len), requires_grad=False, dtype=torch.bool)
 
    if mask_mode in ("time", "freq"):
        if sequence_len % n_freq_patches != 0:
            raise ValueError(
                f"sequence_len={sequence_len} not divisible by n_freq_patches={n_freq_patches}"
            )
        T = sequence_len // n_freq_patches
        for i in range(B):
            mask[i, gen_maskid_structured(T, n_freq_patches, mask_patch, mask_mode)] = True
        return mask
 
    for i in range(B):
        if mask_mode == "cluster":
            if p_t_dim is None:
                raise ValueError("mask_mode='cluster' requires p_t_dim "
                                 "(your gen_maskid_patch takes it as first arg)")
            from utils import gen_maskid_patch  # adjust to your package path
            mask_id = gen_maskid_patch(p_t_dim=p_t_dim, sequence_len=sequence_len,
                                       mask_patch=mask_patch)
        elif mask_mode == "random":
            mask_id = torch.tensor(random.sample(range(sequence_len), mask_patch))
        else:
            raise ValueError(f"unknown mask_mode '{mask_mode}'")
        mask[i, mask_id] = True
    return mask
 