"""
Single-variant IV-loss ablation for AudioSphere:
    loss = MSE(mel channels)  +  (1 - cos) on intensity vectors.

Drop regime, original pipeline — ONLY the loss changes. The MSE control is the
plain AudioSphere mask_random run; do not retrain a control with this class.

Notes that make this defensible:
  * Direction is computed on DE-NORMALIZED IVs. spectrogram_normalize is
    LayerNorm over [C, F, T] with one per-sample (mu, sigma) shared across all
    channels; subtracting a shared scalar changes a 3-vector's angle, so
    cosine in the normalized domain would not measure direction.
  * Near-silent pixels (target IV norm < dir_eps) have undefined direction and
    are excluded; the excluded fraction is logged every step.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

MEL_CHANNELS = [0, 1, 2, 3]   # W, Y, Z, X log-mels
IV_CHANNELS = [4, 5, 6]       # I_y, I_z, I_x


def masked_mel_mse(pred, target, mask, mel_channels=MEL_CHANNELS):
    """pred/target: (B, N, C, fs, ts) normalized domain; mask: (B, N) True=masked."""
    p = pred[:, :, mel_channels].flatten(2)
    t = target[:, :, mel_channels].flatten(2)
    per_patch = ((p - t) ** 2).mean(dim=-1)               # (B, N)
    m = mask.float()
    return (per_patch * m).sum() / m.sum().clamp_min(1.0)


def masked_iv_cosine(pred_iv, target_iv, mask, dir_eps: float = 1e-6
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """pred_iv/target_iv: (B, N, 3, fs, ts) RAW (de-normalized) domain.
    Returns (mean(1 - cos) over masked directional pixels, excluded fraction)."""
    B, N, three, fs, ts = pred_iv.shape
    assert three == 3
    p = pred_iv.permute(0, 1, 3, 4, 2).reshape(B, N, fs * ts, 3)
    t = target_iv.permute(0, 1, 3, 4, 2).reshape(B, N, fs * ts, 3)

    t_norm = t.norm(dim=-1)
    valid = (t_norm > dir_eps) & mask.unsqueeze(-1)
    n_valid = valid.float().sum().clamp_min(1.0)

    cos = F.cosine_similarity(p, t, dim=-1, eps=dir_eps)
    loss = ((1.0 - cos) * valid.float()).sum() / n_valid

    masked_px = mask.unsqueeze(-1).expand_as(t_norm).float().sum().clamp_min(1.0)
    excluded_frac = 1.0 - valid.float().sum() / masked_px
    return loss, excluded_frac


def layernorm_stats(x: torch.Tensor, eps: float = 1e-5):
    """Per-sample (mu, sigma) of LayerNorm([C,F,T], affine=False): dims (1,2,3)."""
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
    return mu, torch.sqrt(var + eps)


try:
    from .audiosphere import AudioSphere          # adjust to your package path
except Exception:
    AudioSphere = object                          # standalone import/testing

class AudioSphereIVCosine(AudioSphere):
    """AudioSphere with loss = mel MSE + IV cosine. Nothing else differs."""

    def __init__(self, *args, dir_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir_eps = dir_eps

    def forward(self, x, mask):
        assert x.ndim == 4, f"Have to be B,C,T,F got {x.shape}"
        B = x.shape[0]
        x = x.transpose(2, 3)                     # (B, C, F, T)
        mu, sigma = layernorm_stats(x)
        x_n = self.spectrogram_normalize(x)

        patches = self.patch_strategy.patch(x_n)  # (B, N, C, fs, ts)
        self.patches_shape = patches.shape

        encoded = self.patch_strategy.embed(x_n, self.patch_embed)
        h = self.pass_through_encoder(encoded, ~mask, B)
        pred_flat = self.pass_through_decoder(h, ~mask, B)

        Bp, N, C, fs, ts = patches.shape
        pred = pred_flat.view(Bp, N, C, fs, ts)

        loss_mel = masked_mel_mse(pred, patches, mask)

        s = sigma.view(B, 1, 1, 1, 1)
        m_ = mu.view(B, 1, 1, 1, 1)
        loss_iv, excluded = masked_iv_cosine(
            pred[:, :, IV_CHANNELS] * s + m_,
            patches[:, :, IV_CHANNELS] * s + m_,
            mask, dir_eps=self.dir_eps,
        )

        loss = loss_mel + loss_iv
        self._loss_components = {
            "loss_mel_mse": loss_mel.detach(),
            "loss_iv_cosine": loss_iv.detach(),
            "iv_pixels_excluded_frac": excluded.detach(),
        }
        return pred_flat, patches.flatten(2), loss

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        if getattr(self, "_loss_components", None):
            self.log_dict(self._loss_components)
        return out