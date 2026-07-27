import sys

sys.path.append("..")
import torch

from src.model import AudioSphere
from src.model import AudioSphereChannelMasked
from src.model import AudioSphereIVCosine

from src.patching import PatchStrategy

from .feature_helper_audio_sphere import FeatureExtractor, get_timestamps
import torch.nn.functional as F

import os
import warnings
 
import torch
import torch.nn.functional as F
 
ENV_TARGET = "HEAR_TIMESTAMP_HOP_MS"
ENV_NATIVE = "HEAR_NATIVE_HOP_MS"
DEFAULT_NATIVE_HOP_MS = 80.0
 
 
def _env_float(name, default=None):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "" or raw.strip().lower() in ("null", "none"):
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None
    if value <= 0:
        raise ValueError(f"{name}={raw!r} must be positive")
    return value
 
 
def native_hop_ms():
    return _env_float(ENV_NATIVE, DEFAULT_NATIVE_HOP_MS)
 
 
def target_hop_ms():
    """None when no resampling was requested."""
    return _env_float(ENV_TARGET, None)
 
 
def target_frames(n_native, native_hop, target_hop):
    """How many frames the same span covers at the target hop."""
    return max(1, int(round(n_native * native_hop / target_hop)))
 
 
def resample_timestamps(embeddings, timestamps, *,
                        native_hop=None, target_hop=None):
    """
    embeddings: (B, T, D) at the native hop
    timestamps: (B, T) or (T,) frame times in MILLISECONDS
    Returns (embeddings, timestamps) on the target grid. The timestamps are
    always regenerated at the hop the embeddings are on, so pooled output is
    never paired with the native time axis.
 
    Explicit native_hop/target_hop override the environment; both in ms.
    """
    if native_hop is None:
        native_hop = native_hop_ms()
    if target_hop is None:
        target_hop = target_hop_ms()
 
    if embeddings.dim() != 3:
        raise ValueError(f"expected (B, T, D) embeddings, got {tuple(embeddings.shape)}")
 
    n_native = embeddings.shape[1]
    hop = native_hop if target_hop is None else target_hop
    n_frames = (n_native if target_hop is None
                else target_frames(n_native, native_hop, target_hop))
 
    if target_hop is not None and n_frames > n_native:
        warnings.warn(
            f"{ENV_TARGET}={target_hop} is finer than the native hop {native_hop} "
            f"ms ({n_native} -> {n_frames} frames); adaptive_avg_pool1d duplicates "
            f"frames rather than adding detail.",
            stacklevel=2,
        )
 
    if n_frames != n_native:
        # (B,T,D) -> (B,D,T) -> pool -> (B,T',D)
        embeddings = F.adaptive_avg_pool1d(
            embeddings.transpose(1, 2), n_frames
        ).transpose(1, 2)
 
    new_ts = torch.arange(n_frames, dtype=torch.float32,
                          device=embeddings.device) * hop
    if timestamps is not None and timestamps.dim() == 2:
        new_ts = new_ts.unsqueeze(0).expand(embeddings.shape[0], -1).contiguous()

    return embeddings, new_ts

def _to_bool(v) -> bool:
    """model_options arrive as JSON strings: 'false' is truthy in Python."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


class _ZeroIndicatorProj(torch.nn.Module):
    def __init__(self, proj: torch.nn.Conv2d):
        super().__init__()
        self.proj = proj

    def forward(self, x):
        return self.proj(torch.cat([x, torch.zeros_like(x)], dim=1))


class RuntimeAudioSphere(torch.nn.Module):
    def __init__(
        self,
        model_size,
        decoder_embedding_dim,
        in_channels,
        weights,
        fshape,
        tshape,
        fstride,
        tstride,
        input_tdim,
        starategy: str = "raw",
        layer: int = None,
        skip_weights=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.decoder_depth = kwargs.get("decoder_depth", 8)
        self.use_mwmae_decoder = _to_bool(kwargs.get("use_mwmae_decoder", False))
        self.decoder_num_heads = kwargs.get("decoder_num_heads", 8)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        self.decoder_window_sizes = kwargs.get(
            "decoder_window_sizes", [2, 5, 10, 25, 50, 100, 0, 0]
        )
        self.num_mel_bins = kwargs.get("num_mel_bins", 128)

        # ------------------------------------------------ class selection --- #
        model_class = kwargs.get("model_class", "AudioSphere")
        networks = {"AudioSphere": AudioSphere}
        if AudioSphereChannelMasked is not None:
            networks["AudioSphereChannelMasked"] = AudioSphereChannelMasked
        if AudioSphereIVCosine is not None:
            networks["AudioSphereIVCosine"] = AudioSphereIVCosine
        if model_class not in networks:
            raise ValueError(
                f"model_class={model_class!r} not importable/known; "
                f"have {sorted(networks)}"
            )

        extra = {}
        if model_class == "AudioSphereChannelMasked":
            extra = dict(
                channel_mask_mode=kwargs.get("channel_mask_mode", "tube"),
                add_mask_indicator=_to_bool(kwargs.get("add_mask_indicator", True)),
            )
        elif model_class == "AudioSphereIVCosine":
            extra = dict(dir_eps=float(kwargs.get("dir_eps", 1e-6)))

        self.model = networks[model_class](
            model_size=model_size,
            patch_strategy=PatchStrategy(
                tstride=tstride,
                tshape=tshape,
                fstride=fstride,
                fshape=fshape,
                input_fdim=self.num_mel_bins,
                input_tdim=input_tdim,
            ),
            in_channels=in_channels,
            decoder_window_sizes=self.decoder_window_sizes,
            use_mwmae_decoder=self.use_mwmae_decoder,
            **extra,
        )

        # ------------------------------------------------ checkpoint load --- #
        if not skip_weights:
            result = self.model.load_state_dict(weights["state_dict"], strict=False)
            # strict=False still hard-fails on shape mismatches, so if we get
            # here shapes agree — but a checkpoint/class mix-up also shows up
            # as *missing* encoder weights silently left at random init, which
            # produces garbage embeddings, not a crash. Refuse that.
            critical = [
                k for k in result.missing_keys
                if k.startswith(("encoder.", "patch_embed.", "cls_token"))
            ]
            if critical:
                raise RuntimeError(
                    f"Checkpoint is missing {len(critical)} encoder-side keys "
                    f"(e.g. {critical[:3]}); wrong model_class or wrong ckpt? "
                    f"model_class={model_class}"
                )
            if result.missing_keys or result.unexpected_keys:
                print(
                    f"[RuntimeAudioSphere] non-critical load report: "
                    f"missing={result.missing_keys} "
                    f"unexpected={result.unexpected_keys}"
                )

        # ---------------------------------------------- indicator shim ------ #
        # AFTER loading: keys must match the checkpoint before wrapping.
        if getattr(self.model, "add_mask_indicator", False):
            self.model.patch_embed.proj = _ZeroIndicatorProj(
                self.model.patch_embed.proj
            )

        # The input size to the model is the input_t_dim and the number of mel bins.
        self.grid_size = self.model.grid_size
        self.input_size = (input_tdim, self.num_mel_bins)
        self.embedding_size = self.model.encoder_embedding_dim
        self.scene_embedding_size = self.model.encoder_embedding_dim
        self.timestamp_embedding_size = self.model.encoder_embedding_dim

        # That's where we set the sample rate!
        self.sample_rate = 32000
        self.strategy = starategy
        self.mel_spec = FeatureExtractor(
            in_channels=self.in_channels,
            sr=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
        )
        self.until_layer = layer

    def to_feature(self, batch_audio):
        return self.mel_spec(batch_audio)

    def encode(self, x):
        unit_frames = self.input_size[0]
        cur_frames = x.shape[2]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            # Padding with constant 0s
            pad_arg = (
                0,
                0,
                0,
                pad_frames,
            )  # (channel, channel, height, height, width, width)
            x = torch.nn.functional.pad(x, pad_arg, mode="constant")
        embeddings = []
        # Now get the embeddings of the model.
        for i in range(x.shape[2] // unit_frames):
            x_inp = x[:, :, i * unit_frames : (i + 1) * unit_frames, :]
            with torch.no_grad():
                if self.until_layer is not None:
                    embedding = self.model.get_audio_representation_from_layer(
                        x_inp, strategy=self.strategy, block_num=self.until_layer
                    )
                else:
                    embedding = self.model.get_audio_representation(
                        x_inp, strategy=self.strategy
                    )
            embeddings.append(embedding)
        # Stack the embeddings here if it is raw
        if self.strategy == "raw":
            x = torch.hstack(embeddings)
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[:, :-pad_emb_frames]  # remove padded tail
            return x
        else:
            x = torch.stack(embeddings, dim=1)
            return x

    def audio2feats(self, audio):
        x = self.to_feature(audio)
        x = self.encode(x)
        return x

    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        x, ts = resample_timestamps(x, ts)
        return x, ts