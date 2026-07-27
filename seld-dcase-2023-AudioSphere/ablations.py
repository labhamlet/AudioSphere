# The SELDnet architecture
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")

from hear_api.runtime import RuntimeAudioSphere


def _to_bool(v) -> bool:
    """Options often arrive as strings ('true'/'false'); bool('false') is True."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*4*13]
            target: [batch_size, frames, num_track_dummy=6, num_axis=5, num_class=13]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6,
                         loss_7, loss_8, loss_9, loss_10, loss_11, loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    """Baseline SELDnet: CNN -> BiGRU -> MHSA -> FNN head."""

    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params = params
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(
                    in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                    out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d(
                    (params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(
            np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(
                embed_dim=self.params['rnn_size'], num_heads=params['nb_heads'],
                dropout=params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(
                    params['fnn_size'] if fc_cnt else self.params['rnn_size'],
                    params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(
            params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'],
            out_shape[-1], bias=True))

    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa


class AudioSphereSELD(nn.Module):
    """SELD model built on an AudioSphere-Ambisonics encoder loaded
    from a local checkpoint (.ckpt) via RuntimeAudioSphere.

    Ablation support
    ----------------
    `model_class` selects the pretraining class the checkpoint came from:
        "AudioSphere"              mask_random / mask_time / mask_freq /
                                   ratio sweep (drop regime)
        "AudioSphereChannelMasked" chan_tube / chan_independent / chan_mel2iv /
                                   chan_w_iv2xyz (infill regime; 14-ch conv +
                                   zero-indicator shim, handled inside
                                   RuntimeAudioSphere)
        "AudioSphereIVCosine"      ivloss_cosine (architecture == parent)

    Example (chan_independent probe):
        model = AudioSphereSELD(out_shape, params,
                                ckpt_path=".../ChannelMask=independent/Indicator=True/step=20000.ckpt",
                                model_class="AudioSphereChannelMasked",
                                use_mwmae_decoder=True)

    Pipeline
    --------
        log_mel (B, 7, T_in, F_mel)   ambisonic log-mel + IVs (from the extractor)
          |
          v  AudioSphere raw output          self._encode(log_mel)
        (B, T, F*D)
          |
          v  attention frequency pooling     learned query over F patches
        (B, T, D)
          |
          v  time adaptation to SELD len     adaptive_avg_pool -> T_seld
        (B, T_seld, D)
          |
          v  BiGRU + tanh multiplicative gating
        (B, T_seld, rnn_size)
          |
          v  MHSA blocks (residual + LayerNorm)
          |
          v  FNN head -> tanh
        doa (B, T_seld, out_shape[-1])
    """

    def __init__(self, out_shape, params,
                 ckpt_path: Optional[str] = None,
                 audio_sphere: Optional[nn.Module] = None,
                 in_channels: int = 7,
                 strategy: str = "raw",
                 layer: Optional[int] = None,
                 embed_dim: int = 768, f_patches: int = 8,
                 freeze_encoder: bool = True,
                 model_class: str = "AudioSphere",
                 channel_mask_mode: str = "tube",
                 add_mask_indicator: bool = True,
                 dir_eps: float = 1e-6,
                 **runtime_kwargs):
        super().__init__()
        self.params = params
        self.nb_classes = params['unique_classes']
        self.T_seld = out_shape[-2]
        self.freeze_encoder = freeze_encoder
        self.strategy = strategy
        self.layer = layer

        # ---- AudioSphere encoder from a local checkpoint ----
        if audio_sphere is None:
            if ckpt_path is None:
                raise ValueError("Provide either `ckpt_path` or a pre-built `audio_sphere` module.")
            weights = torch.load(ckpt_path, map_location="cpu")
            audio_sphere = RuntimeAudioSphere(
                model_size=runtime_kwargs.pop("model_size", "base"),
                decoder_embedding_dim=runtime_kwargs.pop("decoder_embedding_dim", 512),
                in_channels=in_channels,
                weights=weights,
                fshape=runtime_kwargs.pop("fshape", 16),
                tshape=runtime_kwargs.pop("tshape", 8),
                fstride=runtime_kwargs.pop("fstride", 16),
                tstride=runtime_kwargs.pop("tstride", 8),
                input_tdim=runtime_kwargs.pop("input_tdim", 200),
                starategy=strategy,           # (sic) RuntimeAudioSphere's arg name
                layer=layer,
                use_mwmae_decoder=_to_bool(runtime_kwargs.pop("use_mwmae_decoder", False)),
                decoder_window_sizes=runtime_kwargs.pop(
                    "decoder_window_sizes", [2, 5, 10, 25, 50, 100, 0, 0]),
                # ---- ablation plumbing (consumed by the fixed wrapper) ----
                model_class=model_class,
                channel_mask_mode=channel_mask_mode,
                add_mask_indicator=_to_bool(add_mask_indicator),
                dir_eps=dir_eps,
                **runtime_kwargs,
            )
        self.audio_sphere = audio_sphere
        if freeze_encoder:
            for p in self.audio_sphere.parameters():
                p.requires_grad = False
            self.audio_sphere.eval()

        self.D = embed_dim              # token dim
        self.F = f_patches              # nr. frequency patches (F*D == raw last dim)

        # ---- Attention frequency pooling (learned query over F patches) ----
        self.input_norm = nn.LayerNorm(self.F * self.D)
        self.q_freq = nn.Parameter(torch.randn(self.D) * 0.02)
        self.k_proj_freq = nn.Linear(self.D, self.D, bias=False)

        # ---- BiGRU ----
        self.gru = nn.GRU(
            input_size=self.D, hidden_size=params['rnn_size'],
            num_layers=params['nb_rnn_layers'], batch_first=True,
            dropout=params['dropout_rate'], bidirectional=True,
        )

        # ---- Multi-head self-attention ----
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=params['rnn_size'], num_heads=params['nb_heads'],
                    dropout=params['dropout_rate'], batch_first=True,
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(params['rnn_size']))

        # ---- FNN head (mirrors SeldModel) ----
        self.fnn_list = nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'],
                              params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'],
                      out_shape[-1], bias=True)
        )

    # keep the frozen encoder in eval mode regardless of .train()/.eval()
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.audio_sphere.eval()
        return self

    def _encode(self, x):
        if self.freeze_encoder:
            return self.audio_sphere.encode(x)          # no_grad inside: correct here

        unit_frames = self.audio_sphere.input_size[0]
        cur_frames = x.shape[2]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            x = F.pad(x, (0, 0, 0, pad_frames), mode="constant")
        embeddings = []
        for i in range(x.shape[2] // unit_frames):
            x_inp = x[:, :, i * unit_frames: (i + 1) * unit_frames, :]
            if self.layer is not None:
                emb = self.audio_sphere.model.get_audio_representation_from_layer(
                    x_inp, strategy=self.strategy, block_num=self.layer)
            else:
                emb = self.audio_sphere.model.get_audio_representation(
                    x_inp, strategy=self.strategy)
            embeddings.append(emb)
        if self.strategy == "raw":
            z = torch.hstack(embeddings)
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                z = z[:, :-pad_emb_frames]
            return z
        return torch.stack(embeddings, dim=1)

    @staticmethod
    def _freq_pool(z_flat, query, k_proj, F_dim, D_dim):
        """Attention pooling over the frequency-patch axis. (B,T,F*D)->(B,T,D)."""
        B, T, _ = z_flat.shape
        z = z_flat.view(B, T, F_dim, D_dim)
        attn = (k_proj(z) @ query) / (D_dim ** 0.5)   # (B, T, F)
        attn = attn.softmax(dim=-1).unsqueeze(-1)      # (B, T, F, 1)
        return (z * attn).sum(dim=2)                   # (B, T, D)

    def _adapt_time(self, z):                          # (B,T,D)->(B,T_seld,D)
        if z.shape[1] == self.T_seld:
            return z
        return F.adaptive_avg_pool1d(z.transpose(1, 2), self.T_seld).transpose(1, 2)

    def forward(self, log_mel):
        """log_mel: extractor features, (B, 7, T_in, F_mel)."""
        z = self._encode(log_mel)                      # (B, T, F*D)
        assert z.shape[-1] == self.F * self.D, (
            f"raw dim {z.shape[-1]} != F*D ({self.F}*{self.D}); "
            f"set embed_dim/f_patches to match the checkpoint")

        # ---- Attention frequency pooling ----
        z = self.input_norm(z)
        z = self._freq_pool(z, self.q_freq, self.k_proj_freq, self.F, self.D)  # (B, T, D)

        # ---- Time adaptation to SELD resolution ----
        z = self._adapt_time(z)                        # (B, T_seld, D)

        # ---- BiGRU + tanh multiplicative gating (as in SeldModel) ----
        z, _ = self.gru(z)
        z = torch.tanh(z)
        z = z[:, :, z.shape[-1] // 2:] * z[:, :, :z.shape[-1] // 2]

        # ---- MHSA (residual + LayerNorm) ----
        for mhsa, ln in zip(self.mhsa_block_list, self.layer_norm_list):
            z_in = z
            z, _ = mhsa(z_in, z_in, z_in)
            z = z + z_in
            z = ln(z)

        # ---- FNN head ----
        for fnn in self.fnn_list[:-1]:
            z = fnn(z)
        doa = torch.tanh(self.fnn_list[-1](z))         # (B, T_seld, out_shape[-1])
        return doa