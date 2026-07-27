"""
Temporal ACCDOA heads for hear-eval-kit SELD evaluation.

The DCASE SELDnet head is, structurally: conv stack -> BiGRU -> multi-head
self-attention -> FC -> ACCDOA. A HEAR encoder has already done the conv stack's
job, so this head starts at the frequency-patch axis and works down.

Shape flow, matching AudioSphereSELD:

    (B, T, F*D)   raw AudioSphere output, e.g. 8 patches x 768 = 6144
      |  frequency pooling (attention over the F patches, or mean, or none)
    (B, T, D)
      |  projection block: LayerNorm -> Linear -> GELU -> Dropout
    (B, T, P)
      |  BiGRU, merged by concat or SELDnet's multiplicative gating
    (B, T, H)
      |  MHSA blocks, residual + LayerNorm
      |  optional temporal average pooling to the label grid
      |  Linear -> tanh
    (B, T', nlabels, 3)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ACCDOAHead", "FrequencyPool", "masked_accdoa_mse"]


class FrequencyPool(nn.Module):
    """
    Collapse the frequency-patch axis of a flattened (B, T, F*D) embedding.

    mode="attention": a learned query scores each of the F patches per frame and
        the patches are combined by the resulting softmax. Lets the head weight
        frequency bands differently per frame, which matters for SELD because
        the bands carrying reliable directional information move with the
        source.
    mode="mean": unweighted average. Cheaper, no parameters, a reasonable
        control to run against attention.
    mode="none": pass through untouched, for encoders that already pool.

    NOTE: the reshape assumes the flattened axis is patch-major, i.e.
    [f0_d0..f0_dD, f1_d0..], which is what AudioSphereSELD._freq_pool assumes.
    If a future encoder flattens dim-major instead, this silently scrambles -
    the divisibility assert cannot catch that.
    """

    def __init__(self, in_dim: int, embed_dim: int, mode: str = "attention"):
        super().__init__()
        if mode not in ("attention", "mean", "none"):
            raise ValueError(f"unknown freq_pool mode {mode!r}")

        self.mode = mode
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        if mode == "none" or in_dim == embed_dim:
            # Nothing to pool: the encoder already collapsed frequency.
            self.mode = "none"
            self.f_patches = 1
            self.out_dim = in_dim
            return

        if in_dim % embed_dim:
            raise ValueError(
                f"embedding dim {in_dim} is not a multiple of embed_dim "
                f"{embed_dim}; set embed_dim to the encoder's token width "
                f"(768 for AudioSphere base) or use freq_pool='none'."
            )
        self.f_patches = in_dim // embed_dim
        self.out_dim = embed_dim

        if mode == "attention":
            self.query = nn.Parameter(torch.randn(embed_dim) * 0.02)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        b, t, _ = x.shape
        z = x.view(b, t, self.f_patches, self.embed_dim)
        if self.mode == "mean":
            return z.mean(dim=2)
        attn = (self.k_proj(z) @ self.query) / (self.embed_dim ** 0.5)  # (B,T,F)
        return (z * attn.softmax(dim=-1).unsqueeze(-1)).sum(dim=2)

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, in_dim={self.in_dim}, "
                f"f_patches={self.f_patches}, out_dim={self.out_dim}")


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, nheads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, nheads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]):
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(x + h)


class ACCDOAHead(nn.Module):
    """
    (B, T, D_in) embeddings -> (B, T // pool_factor, nlabels, 3) ACCDOA vectors.

    Parameters
    ----------
    freq_pool : "attention" | "mean" | "none". With AudioSphere's raw strategy
        the embedding is F*D (8*768 = 6144); pooling recovers the (B, T, 768)
        token stream instead of flattening 6144 features into one Linear.
        Falls back to "none" automatically when in_dim == embed_dim, so the same
        config works for pooled encoders.
    embed_dim : encoder token width, needed only when pooling. 768 for base.
    proj_dim : width of the projection block feeding the GRU. Defaults to
        hidden_dim. AudioSphereSELD's `proj_gru` is
        LayerNorm(768) -> Linear(768, 256) -> GELU -> Dropout, i.e.
        norm_position="both", use_projection=True, proj_dim=256.
    norm_position : where the LayerNorm goes. "pre_pool" is input_norm over
        F*D; "post_pool" is the LayerNorm at the head of proj_gru; "both" keeps
        input_norm AND proj_gru's norm; "none" drops both.
    gru_merge : "concat" runs a bidirectional GRU at hidden_dim//2 per direction
        and concatenates. "gate" reproduces SELDnet exactly: hidden_dim per
        direction, tanh, then elementwise multiply of the two halves.
    pool_factor : average-pool the time axis before the output projection, to
        map the encoder frame rate onto the label grid. Leave at 1 when the hop
        is already 100 ms.
    bounded : tanh the output. Targets are unit vectors or zeros, so bounding to
        [-1, 1] is well matched and is what the DCASE baseline does.
    """

    def __init__(
        self,
        in_dim: int,
        nlabels: int,
        hidden_dim: int = 256,
        proj_dim: Optional[int] = None,
        gru_layers: int = 2,
        attn_layers: int = 2,
        attn_heads: int = 8,
        dropout: float = 0.05,
        pool_factor: int = 1,
        freq_pool: str = "none",
        embed_dim: Optional[int] = None,
        gru_merge: str = "concat",
        use_projection: bool = True,
        norm_position: str = "post_pool",
        fnn_layers: int = 0,
        fnn_size: Optional[int] = None,
        out_dropout: Optional[float] = None,
        n_tracks: int = 1,
        bounded: bool = True,
    ):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even (bidirectional GRU)")
        if hidden_dim % attn_heads:
            raise ValueError("hidden_dim must be divisible by attn_heads")
        if gru_merge not in ("concat", "gate"):
            raise ValueError(f"unknown gru_merge {gru_merge!r}")
        if freq_pool != "none" and embed_dim is None:
            raise ValueError("freq_pool needs embed_dim (the encoder token width)")
        if norm_position not in ("pre_pool", "post_pool", "both", "none"):
            raise ValueError(
                f"unknown norm_position {norm_position!r}; expected one of "
                f"pre_pool, post_pool, both, none"
            )

        self.nlabels = nlabels
        self.n_tracks = n_tracks
        self.pool_factor = pool_factor
        self.gru_merge = gru_merge
        self.bounded = bounded

        # ---- frequency pooling -------------------------------------------- #
        # "pre_pool"  normalises the flattened F*D vector before pooling, which
        #             is AudioSphereSELD.input_norm.
        # "post_pool" normalises the pooled token instead, which is the
        #             LayerNorm at the head of proj_gru.
        # "both"      does both - input_norm AND proj_gru's LayerNorm, which is
        #             AudioSphereSELD with proj_gru uncommented.
        # "none"      neither.
        self.pre_norm = (
            nn.LayerNorm(in_dim) if norm_position in ("pre_pool", "both")
            else nn.Identity()
        )
        self.freq_pool = FrequencyPool(in_dim, embed_dim or in_dim, freq_pool)
        token_dim = self.freq_pool.out_dim
        post_norm = (
            nn.LayerNorm(token_dim) if norm_position in ("post_pool", "both")
            else nn.Identity()
        )

        # ---- projection block --------------------------------------------- #
        # use_projection=False feeds the pooled token straight to the GRU, as
        # AudioSphereSELD does. True adds LayerNorm -> Linear -> GELU -> Dropout,
        # which is usually better but is an extra block the baseline lacks.
        if use_projection:
            proj_dim = proj_dim or hidden_dim
            self.proj = nn.Sequential(
                post_norm,
                nn.Linear(token_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            gru_in = proj_dim
        else:
            self.proj = post_norm
            gru_in = token_dim

        # ---- recurrence ---------------------------------------------------- #
        gru_hidden = hidden_dim if gru_merge == "gate" else hidden_dim // 2
        self.gru = nn.GRU(
            gru_in,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ---- attention and output ------------------------------------------ #
        self.attn_blocks = nn.ModuleList(
            _AttentionBlock(hidden_dim, attn_heads, dropout) for _ in range(attn_layers)
        )
        # AudioSphereSELD has no dropout between the attention stack and the
        # FNN head. out_dropout=0.0 removes it for parity; None keeps `dropout`.
        self.dropout = nn.Dropout(dropout if out_dropout is None else out_dropout)

        # ---- FNN stack ------------------------------------------------------ #
        # SeldModel / AudioSphereSELD apply these with NO activation between
        # them - only the final tanh. Reproduced as-is rather than "fixed".
        fnn_size = fnn_size or hidden_dim
        width = hidden_dim
        fnn = []
        for _ in range(fnn_layers):
            fnn.append(nn.Linear(width, fnn_size))
            width = fnn_size
        self.fnn = nn.ModuleList(fnn)
        self.out = nn.Linear(width, n_tracks * 3 * nlabels)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x    : (B, T, D_in)
        mask : (B, T) at the input frame rate, 1 for real frames. Optional but
               recommended: it keeps padded tail frames out of the attention.
        """
        h = self.proj(self.freq_pool(self.pre_norm(x)))
        h, _ = self.gru(h)

        if self.gru_merge == "gate":
            # SELDnet's multiplicative gating: tanh, then forward * backward.
            h = torch.tanh(h)
            half = h.shape[-1] // 2
            h = h[..., half:] * h[..., :half]

        key_padding = None
        if mask is not None and self.attn_blocks:
            key_padding = mask < 0.5  # True == ignore
            # A fully-padded row makes softmax produce NaNs; keep one frame alive.
            all_pad = key_padding.all(dim=1)
            if all_pad.any():
                key_padding = key_padding.clone()
                key_padding[all_pad, 0] = False
        for block in self.attn_blocks:
            h = block(h, key_padding)

        if self.pool_factor > 1:
            t = (h.shape[1] // self.pool_factor) * self.pool_factor
            h = F.avg_pool1d(
                h[:, :t].transpose(1, 2), kernel_size=self.pool_factor
            ).transpose(1, 2)

        h = self.dropout(h)
        for layer in self.fnn:
            h = layer(h)
        h = self.out(h)
        if self.bounded:
            h = torch.tanh(h)
        # Two layouts, because the two decoders want different things.
        # Single-ACCDOA: (B, T, C, 3), which get_accdoa_labels indexes per class.
        # Multi-ACCDOA:  (B, T, n_tracks*3, C), matching the flat
        #   [track0_x_allclasses, track0_y..., track1_x...] ordering that
        #   MSELoss_ADPIT and get_multi_accdoa_labels assume.
        if self.n_tracks == 1:
            return h.reshape(h.shape[0], h.shape[1], self.nlabels, 3)
        return h.reshape(h.shape[0], h.shape[1], self.n_tracks * 3, self.nlabels)


def masked_accdoa_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    pred, target : (B, T, C, 3)
    mask         : (B, T), 1 for real frames.

    Padded frames must be excluded or the model is rewarded for predicting
    silence on frames that do not exist, which quietly biases recall down.
    """
    se = ((pred - target) ** 2).mean(dim=(-1, -2))  # (B, T)
    return (se * mask).sum() / mask.sum().clamp(min=1.0)