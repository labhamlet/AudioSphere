import sys

sys.path.append("..")
import torch

from src.model import AudioSphere
from src.model import AudioSphereChannelMasked
from src.model import AudioSphereIVCosine

from src.patching import PatchStrategy

from .feature_helper_audio_sphere import FeatureExtractor, get_timestamps
import torch.nn.functional as F




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
        # This takes the mean embedding across the scene!
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts