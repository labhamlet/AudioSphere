import random
from typing import List, Optional
from random import randrange
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
            mask_id = gen_maskid_patch(p_t_dim=p_t_dim, sequence_len=sequence_len,
                                       mask_patch=mask_patch)
        elif mask_mode == "random":
            mask_id = torch.tensor(random.sample(range(sequence_len), mask_patch))
        else:
            raise ValueError(f"unknown mask_mode '{mask_mode}'")
        mask[i, mask_id] = True
    return mask
 
def generate_masks(patches, encoded_patches, cluster, mask_patch, device, p_t_dim):
    B = encoded_patches.shape[0]
    embedding_dim = encoded_patches.shape[2]

    num_patches = patches.shape[1]
    patch_dim = patches.shape[2]

    mask_index = torch.empty((B, mask_patch), requires_grad=False).long().to(device)
    encode_samples = torch.empty((B, mask_patch, patch_dim), requires_grad=False).to(
        device
    )
    mask_dense = torch.ones([B, num_patches, embedding_dim]).to(device)
    for i in range(B):
        if cluster:
            mask_index[i] = gen_maskid_patch(
                p_t_dim=p_t_dim, sequence_len=num_patches, mask_patch=mask_patch
            )
        else:
            mask_index[i] = gen_maskid_frame(
                sequence_len=num_patches, mask_size=mask_patch
            )
        # copy the masked embeddings, note gradients are stopped in this path
        # encode_samples gets which patches in the input are masked and clones them.
        encode_samples[i] = patches[i, mask_index[i], :].clone().detach()
        # mask the encode samples with 0, otherwise it is 1
        mask_dense[i, mask_index[i], :] = 0
    return mask_index, mask_dense, encode_samples


def gen_maskid_patch(p_t_dim, sequence_len=512, mask_patch=100, cluster=3):
    """
    :p_t_dim: The patch time dimension...
    :mask_patch: Number of patches to mask
    """
    mask_id = []

    # randomize clutering factor in [3,6)
    cur_clus = randrange(cluster) + 3
    while len(list(set(mask_id))) < mask_patch:
        start_id = randrange(sequence_len)
        cur_mask = []
        for i in range(0, cur_clus):
            for j in range(0, cur_clus):
                mask_cand = start_id + p_t_dim * i + j
                if mask_cand >= 0 and mask_cand < sequence_len:
                    cur_mask.append(mask_cand)
        mask_id = mask_id + cur_mask
    mask_id = list(set(mask_id))[:mask_patch]
    return torch.tensor(mask_id)


# using cluster for frame masking hurts the performance, so just use the naive random sampling
def gen_maskid_frame(sequence_len=512, mask_size=100):
    mask_id = random.sample(range(0, sequence_len), mask_size)
    return torch.tensor(mask_id)


def mask_input(x, mask_dense, mask_embed):
    mask_tokens = mask_embed.expand(x.shape[0], x.shape[1], -1)
    # Drop the masked tokens by making sure that in the x we have the masked tokens replaced with the embedding of masked tokens
    x = x * mask_dense + (1 - mask_dense) * mask_tokens
    return x