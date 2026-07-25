import glob

def _san(value):
    """Path-safe token value: no '_' (train.py's separator), no '/'."""
    return str(value).replace("_", "-").replace("/", "-")


def _resolved_mask_patch(cfg):
    """Same resolution rule as train.py: prefer the masking group, fall back
    to data.mask_patch. Legacy configs have no masking group -> unchanged."""
    mask_cfg = cfg.get("masking", None)
    if mask_cfg is not None:
        return mask_cfg.get("mask_patch", cfg.data.mask_patch)
    return cfg.data.mask_patch


def get_identity_from_cfg(cfg):
    # ---- legacy core: DO NOT REORDER (existing checkpoints depend on it) ---
    identity = "InChannels={}_Fraction={}_CleanDataFraction={}_".format(
        cfg.data.get("in_channels"),
        cfg.data.get("data_ratio"),
        cfg.data.get("clean_data_ratio"),
    )
    identity += "Model={}_ModelSize={}_".format(
        cfg.model, cfg.model_size,
    )
    identity += "LR={}_BatchSize={}_NrSamples={}_".format(
        cfg.optimizer.get("lr"),
        cfg.trainer.get("batch_size"),
        cfg.data.get("samples_per_audio"),
    )
    identity += "Patching={}_MaskPatch={}_InputL={}_Cluster={}".format(
        _san(cfg.patching.get("name")),
        _resolved_mask_patch(cfg),
        cfg.data.target_length,
        cfg.masking.cluster,
    )

    # ---- ablation tokens: appended ONLY when the keys exist ---------------
    mask_cfg = cfg.get("masking", None)
    mask_mode = mask_cfg.get("mask_mode", None) if mask_cfg is not None else None
    if mask_mode not in (None, "random"):          # random == legacy behavior
        identity += "_MaskMode={}".format(_san(mask_mode))

    if cfg.model == "AudioSphereChannelMasked":
        identity += "_ChannelMask={}_Indicator={}".format(
            _san(cfg.get("channel_mask_mode", "tube")),
            cfg.get("add_mask_indicator", True),
        )
    elif cfg.model == "AudioSphereIVCosine":
        identity += "_IVLoss=cosine"

    return identity


def get_save_paths(cfg):
    """Single source of truth for where a run logs and checkpoints.
    Returns dict(identity, run_name, log_dir, ckpt_dir)."""
    identity = get_identity_from_cfg(cfg)
    nested = identity.replace("_", "/")
    if cfg.data.in_channels == 7:
        log_dir = f"{cfg.save_dir}/audio_sphere"
        ckpt_dir = f"{cfg.save_dir}/audio_sphere/{nested}"
    else:
        log_dir = f"{cfg.save_dir}/tb_logs_naturalistic_mixing"
        ckpt_dir = f"{cfg.save_dir}/saved_models_naturalistic_mixing/{nested}"
    return {
        "identity": identity,
        "run_name": nested,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
    }

