import torch

from hear_api.runtime import RuntimeAudioSphere


def _to_bool(v) -> bool:
    """model_options values arrive as JSON strings. The old check
    `str(kwargs.get(...)) == "true"` failed for a real bool (str(True) ==
    "True" != "true"), so passing use_mwmae_decoder as an actual JSON bool
    silently built the wrong decoder."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def load_model(*args, **kwargs):
    if len(args) == 0:
        raise ValueError(
            "load_model needs the checkpoint path as its first positional "
            "argument (heareval --model). The old code fell through to an "
            "unbound `model_path` NameError."
        )
    model_path = args[0]

    strategy = kwargs.get("strategy", "raw")
    use_mwmae_decoder = _to_bool(kwargs.get("use_mwmae_decoder", False))
    in_channels = int(kwargs.get("in_channels", 2))
    layer = int(kwargs["layer"]) if "layer" in kwargs else None

    model_class = kwargs.get("model_class", "AudioSphere")
    channel_mask_mode = kwargs.get("channel_mask_mode", "tube")
    add_mask_indicator = _to_bool(kwargs.get("add_mask_indicator", True))
    dir_eps = float(kwargs.get("dir_eps", 1e-6))

    fshape = int(kwargs.get("fshape", 16))
    fstride = int(kwargs.get("fstride", 16))
    tshape = int(kwargs.get("tshape", 8))
    tstride = int(kwargs.get("tstride", 8))

    weights = torch.load(
        model_path,
        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    model = RuntimeAudioSphere(
        model_size="base",
        decoder_embedding_dim=512,
        weights=weights,
        fshape=fshape,
        fstride=fstride,
        tshape=tshape,
        tstride=tstride,
        input_tdim=200,
        starategy=strategy,
        use_mwmae_decoder=use_mwmae_decoder,
        decoder_window_sizes=[2, 5, 10, 25, 50, 100, 0, 0],
        in_channels=in_channels,
        layer=layer,
        model_class=model_class,
        channel_mask_mode=channel_mask_mode,
        add_mask_indicator=add_mask_indicator,
        dir_eps=dir_eps,
    )
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
    