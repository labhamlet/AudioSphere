import random
from typing import List, Optional, Tuple

import torch
from torch import nn

CHANNEL_NAMES = ["W", "Y", "Z", "X", "I_y", "I_z", "I_x"]
MEL_CHANNELS = [0, 1, 2, 3]        # W, Y, Z, X log-mels
W_CHANNEL = 0
XYZ_MEL_CHANNELS = [1, 2, 3]       # directional mels (Y, Z, X)
IV_CHANNELS = [4, 5, 6]            # intensity-vector planes


class ChannelMaskMaker(nn.Module):
    def __init__(
        self,
        mask_patch: int = 160,          # locations per channel (tube/independent)
        mode: str = "tube",             # tube | independent | mel2iv | w_iv2xyz
        n_channels: int = 7,
        iv_channels: List[int] = IV_CHANNELS,
        xyz_mel_channels: List[int] = XYZ_MEL_CHANNELS,
    ):
        super().__init__()  # type: ignore
        assert mode in ("tube", "independent", "mel2iv", "w_iv2xyz")
        self.mask_patch = mask_patch
        self.mode = mode
        self.C = n_channels
        self.iv_channels = iv_channels
        self.xyz_mel_channels = xyz_mel_channels

    def forward(self, batch_size: int, n_tokens: int) -> torch.Tensor:
        B, C, N = batch_size, self.C, n_tokens
        mask = torch.zeros((B, C, N), dtype=torch.bool, requires_grad=False)
        if self.mode == "tube":
            for i in range(B):
                ids = torch.tensor(random.sample(range(N), self.mask_patch))
                mask[i, :, ids] = True                      # same locations, all channels
        elif self.mode == "independent":
            for i in range(B):
                for c in range(C):
                    ids = torch.tensor(random.sample(range(N), self.mask_patch))
                    mask[i, c, ids] = True                  # per-channel locations
        elif self.mode == "mel2iv":
            mask[:, self.iv_channels, :] = True             # predict IVs from mels
        elif self.mode == "w_iv2xyz":
            mask[:, self.xyz_mel_channels, :] = True        # predict XYZ mels from W+IV
        return mask


# --------------------------------------------------------------------------- #
# Input-level infilling
# --------------------------------------------------------------------------- #

def channel_mask_to_pixel_mask(
    chan_mask: torch.Tensor,   # (B, C, N) bool, N = p_f * p_t, flat = f * p_t + t
    p_f: int, p_t: int, fshape: int, tshape: int,
) -> torch.Tensor:
    """Upsample the token-grid mask to spectrogram pixels: (B, C, F, T) bool."""
    B, C, N = chan_mask.shape
    assert N == p_f * p_t
    grid = chan_mask.reshape(B, C, p_f, p_t)
    return grid.repeat_interleave(fshape, dim=2).repeat_interleave(tshape, dim=3)


def apply_infill(
    x: torch.Tensor,           # (B, C, F, T) normalized input (post-transpose)
    pixel_mask: torch.Tensor,  # (B, C, F, T) bool
    channel_fill: torch.Tensor,        # (C,) learnable
    add_indicator: bool = True,
) -> torch.Tensor:
    """Replace masked pixels with per-channel learnable fills; optionally
    concatenate the per-channel mask planes so 'masked' is unambiguous.
    Output channels: C (add_indicator=False) or 2C (True)."""
    fill = channel_fill.view(1, -1, 1, 1).to(x.dtype)
    x_masked = torch.where(pixel_mask, fill.expand_as(x), x)
    if add_indicator:
        x_masked = torch.cat([x_masked, pixel_mask.to(x.dtype)], dim=1)
    return x_masked


# --------------------------------------------------------------------------- #
# Channel-wise masked loss
# --------------------------------------------------------------------------- #

def channelwise_masked_loss(
    pred: torch.Tensor,        # (B, N, C * fshape * tshape) from spec_pred
    target_patches: torch.Tensor,  # (B, N, C, fshape, tshape) from generate_patches
    chan_mask: torch.Tensor,   # (B, C, N) bool
) -> torch.Tensor:
    B, N, C, fs, ts = target_patches.shape
    pred = pred.view(B, N, C, fs * ts)
    target = target_patches.flatten(3)                     # (B, N, C, fs*ts)
    loss = ((pred - target) ** 2).mean(dim=-1)             # (B, N, C)
    m = chan_mask.permute(0, 2, 1).float()                 # (B, N, C)
    return (loss * m).sum() / m.sum().clamp_min(1.0)



AUDIOSPHERE_AVAILABLE = True
try:
    from .audiosphere import AudioSphere 
except Exception:
    AUDIOSPHERE_AVAILABLE = False
    AudioSphere = object 

class AudioSphereChannelMasked(AudioSphere):
    """
    AudioSphere with channel-axis infill masking.

    Differences vs. parent:
      * masking happens at the INPUT (learnable fills + indicator planes),
        so the encoder processes ALL p_f*p_t tokens (~2x encoder FLOPs);
      * the conv patch-embed takes 2*C input channels when indicators are on
        (C masked content + C indicator planes) — set at init;
      * loss is per (location, channel) via channelwise_masked_loss;
      * the decoder's mask_token path is unused (all tokens visible).

    For the AC comparison, run mode="tube" AND mode="independent" with this
    same class so the pair differs ONLY in mask policy.
    """

    def __init__(self, *args, channel_mask_mode: str = "tube",
                 add_mask_indicator: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_mask_mode = channel_mask_mode
        self.add_mask_indicator = add_mask_indicator
        self.channel_fill = nn.Parameter(torch.zeros(self.in_channels))
        self.channel_mask_maker = ChannelMaskMaker(
            mask_patch=self.mask_patch, mode=channel_mask_mode,
            n_channels=self.in_channels,
        )
        if add_mask_indicator:
            # Re-init the conv to accept [content ; indicator] planes.
            self.patch_embed.proj = nn.Conv2d(
                2 * self.in_channels, self.encoder_embedding_dim,
                kernel_size=(self.patch_strategy.fshape, self.patch_strategy.tshape),
                stride=(self.patch_strategy.fstride, self.patch_strategy.tstride),
            )

    def forward(self, x, chan_mask):
        """x: (B, C, T, F); chan_mask: (B, C, N) bool (from ChannelMaskMaker)."""
        assert x.ndim == 4, f"Have to be B,C,T,F got {x.shape}"
        B = x.shape[0]
        x = x.transpose(2, 3)                                  # (B, C, F, T)
        x = self.spectrogram_normalize(x)

        # Targets: the clean patches, same as parent.
        patches = self.patch_strategy.patch(x)                 # (B, N, C, fs, ts)
        self.patches_shape = patches.shape

        # Infill the input.
        pm = channel_mask_to_pixel_mask(
            chan_mask.to(x.device), self.p_f_dim, self.p_t_dim,
            self.patch_strategy.fshape, self.patch_strategy.tshape,
        )
        x_in = apply_infill(x, pm, self.channel_fill, self.add_mask_indicator)

        # Encode ALL tokens (no dropping): all-False location mask.
        encoded = self.patch_strategy.embed(x_in, self.patch_embed)
        no_drop = torch.zeros((B, self.num_patches), dtype=torch.bool, device=x.device)
        h = self.pass_through_encoder(encoded, ~no_drop, B)
        pred = self.pass_through_decoder(h, ~no_drop, B)       # (B, N, C*fs*ts)

        loss = channelwise_masked_loss(pred, patches, chan_mask.to(x.device))
        return pred, patches.flatten(2), loss

    def training_step(self, batch, batch_idx):
        audio_input, _ = self._prepare_batch(batch)            # ignore location mask
        chan_mask = self.channel_mask_maker(audio_input.shape[0], self.num_patches)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            pred, patches, loss = self.forward(audio_input, chan_mask)
        if self.global_step % self.log_every_n_steps == 0:
            self.log_channel_masking(patches, pred, chan_mask)
        self.log_dict({"MSE_Loss": loss})
        return loss

    # ------------------------- visualization ------------------------------ #
    def log_channel_masking(self, patches_flat, pred_flat, chan_mask):
        """Per-channel figure: original | masked input | reconstruction."""
        fig = plot_channel_masking(
            patches_flat[:1].float(), pred_flat[:1].float(), chan_mask[:1],
            patches_shape=self.patches_shape, patch_strategy=self.patch_strategy,
            input_shape=self.input_shape,
        )
        self.logger.experiment.add_figure(
            f"channel_masking/{self.channel_mask_mode}", fig, global_step=self.global_step
        )


def plot_channel_masking(
    patches_flat: torch.Tensor,     # (1, N, C*fs*ts)  clean targets
    pred_flat: torch.Tensor,        # (1, N, C*fs*ts)  reconstruction
    chan_mask: torch.Tensor,        # (1, C, N) bool
    patches_shape: Tuple[int, ...], # (B, N, C, fs, ts)
    patch_strategy,                 # your PatchStrategy (for combine_patches)
    input_shape: List[int],         # [F, T]
    channel_names: Optional[List[str]] = None,
):
    """C rows x 3 cols: [original | masked input (fills shown as blank) |
    reconstruction, masked regions only]. Returns a matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _, N, C, fs, ts = patches_shape
    names = channel_names or (CHANNEL_NAMES[:C] if C <= len(CHANNEL_NAMES)
                              else [f"ch{i}" for i in range(C)])

    def to_img(flat, zero_mask=None, keep_only_mask=None):
        p = flat.reshape(1, N, C, fs, ts).clone()
        if zero_mask is not None:                      # blank masked patches
            p[0][zero_mask.permute(1, 0)] = float("nan")
        if keep_only_mask is not None:                 # blank VISIBLE patches
            p[0][(~keep_only_mask).permute(1, 0)] = float("nan")
        img = patch_strategy.combine_patches(p, input_shape)   # (1, C, F, T)
        return img[0].detach().cpu().numpy()

    orig = to_img(patches_flat)
    masked = to_img(patches_flat, zero_mask=chan_mask[0])
    recon_full = pred_flat.reshape(1, N, C, fs, ts)
    recon = to_img(recon_full.flatten(2), keep_only_mask=chan_mask[0])

    vmin, vmax = np.nanpercentile(orig, 1), np.nanpercentile(orig, 99)
    fig, axes = plt.subplots(C, 3, figsize=(11, 1.6 * C), constrained_layout=True)
    col_titles = ["original", "masked input", "reconstruction (masked only)"]
    for c in range(C):
        for j, img in enumerate((orig, masked, recon)):
            ax = axes[c, j] if C > 1 else axes[j]
            ax.imshow(img[c], origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                      cmap="magma", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(col_titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(names[c], fontsize=9, rotation=0, ha="right", va="center")
    return fig