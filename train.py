import gc

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_modules import WebAudioDataModule
from src.model import AudioSphere
from src.masking import SpatialMaskMaker
from src.patching import PatchStrategy
from utils import get_identity_from_cfg

from src.audiosphere_channel_masking import AudioSphereChannelMasked
from src.audiosphere_iv_cosine import AudioSphereIVCosine


networks = {
    "AudioSphere": AudioSphere,
    "AudioSphereChannelMasked": AudioSphereChannelMasked,
    "AudioSphereIVCosine": AudioSphereIVCosine,
}

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="./configs", config_name="base")
def main(cfg):
    identity = get_identity_from_cfg(cfg)
    log_dir = f"{cfg.save_dir}/audio_sphere" if cfg.data.in_channels == 7 else f"{cfg.save_dir}/tb_logs_naturalistic_mixing"
    save_dir = f"{cfg.save_dir}/audio_sphere/{identity.replace('_', '/')}" if cfg.data.in_channels == 7 else f"{cfg.save_dir}/saved_models_naturalistic_mixing/{identity.replace('_', '/')}"
    logger = TensorBoardLogger(
        log_dir,
        name=identity.replace("_", "/"),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{step}",
        verbose=True,
        every_n_train_steps=5000,
        save_last=True,
        enable_version_counter=True,
        save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.epochs,
        max_steps=cfg.trainer.steps // cfg.trainer.num_gpus,
        precision=cfg.trainer.precision,
        deterministic=False,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=5,
        gradient_clip_algorithm="norm",
        log_every_n_steps=1,
        check_val_every_n_epoch=100,
        num_nodes=1,
        use_distributed_sampler=False,
        devices=int(cfg.trainer.num_gpus),
        strategy="ddp_find_unused_parameters_true"
        if int(cfg.trainer.num_gpus) > 1
        else "auto",
    )

    mask_cfg = cfg.get("masking", None)
    mask_mode = mask_cfg.get("mask_mode", None) if mask_cfg is not None else None
    mask_patch = int(mask_cfg.get("mask_patch", cfg.data.mask_patch)) if mask_cfg is not None else int(cfg.data.mask_patch)

    # ---------------- model ------------------------------------------------- #
    extra_model_kwargs = {}
    if cfg.model == "AudioSphereChannelMasked":
        extra_model_kwargs = dict(
            channel_mask_mode=cfg.get("channel_mask_mode", "tube"),
            add_mask_indicator=bool(cfg.get("add_mask_indicator", True)),
        )
    elif cfg.model == "AudioSphereIVCosine":
        extra_model_kwargs = dict(dir_eps=float(cfg.get("dir_eps", 1e-6)))

    Network: pl.LightningModule = networks[cfg.model]
    network_instance = Network(
        model_size=cfg.model_size,
        lr=cfg.optimizer.lr,
        trainer=cfg.optimizer.name,
        b1=cfg.optimizer.b1,
        b2=cfg.optimizer.b2,
        weight_decay=cfg.optimizer.weight_decay,
        patch_strategy=PatchStrategy(
            input_tdim=cfg.data.target_length,
            input_fdim=cfg.data.num_mel_bins,
            tstride=cfg.patching.tstride,
            tshape=cfg.patching.tshape,
            fstride=cfg.patching.fstride,
            fshape=cfg.patching.fshape,
        ),
        mask_patch=mask_patch,
        cluster=cfg.data.cluster,
        decoder_window_sizes=cfg.patching.decoder_window_sizes,
        use_mwmae_decoder=cfg.use_mwmae_decoder,
        in_channels=cfg.data.in_channels,
        num_mel_bins=cfg.data.num_mel_bins,
        target_length=cfg.data.target_length,
        input_length=cfg.data.input_length,
        nr_samples_per_audio=cfg.data.samples_per_audio,
        sr=cfg.data.sr,
        compile_modules=cfg.trainer.compile_modules,
        clean_data_ratio=cfg.data.clean_data_ratio,
        **extra_model_kwargs,
    )

    # ---------------- launch-time sanity guards ----------------------------- #
    p_f_dim, p_t_dim = network_instance.p_f_dim, network_instance.p_t_dim
    n_tokens = p_f_dim * p_t_dim
    assert 0 < mask_patch < n_tokens, (
        f"mask_patch={mask_patch} vs. grid {p_f_dim}x{p_t_dim}={n_tokens} tokens: "
        f"mask_patch must be < n_tokens (== masks 100% and the visible set is "
        f"empty). For 80%: {int(round(0.8 * n_tokens))}."
    )
    print(
        f"[ablation] model={cfg.model}  mask_mode={mask_mode or ('cluster' if cfg.data.cluster else 'random')}  "
        f"grid={p_f_dim}x{p_t_dim} ({n_tokens} tokens)  mask_patch={mask_patch} "
        f"({mask_patch / n_tokens:.0%})  extra={extra_model_kwargs}  save={save_dir}",
        flush=True,
    )

    # ---------------- data-side location masker ----------------------------- #
    # For AudioSphereChannelMasked the location mask in the batch is ignored
    # (the model draws its own channel mask per step); the masker stays wired
    # so the data pipeline is identical across all runs.
    masker = SpatialMaskMaker(
        mask_patch=mask_patch,
        context_cluster=cfg.data.cluster,
        mask_mode=mask_mode,          # None -> legacy random/cluster behavior
        n_freq_patches=p_f_dim,       # from the actual grid, not from config
        p_t_dim=p_t_dim,              # needed by cluster mode
    )

    data = WebAudioDataModule(
        base_data_dir=cfg.data.base_data_dir,
        rir_data_dir=cfg.data.rir_data_dir,
        val_data_dir=cfg.data.val_data_dir,
        base_noise_dir=cfg.data.base_noise_dir,
        batch_size=cfg.trainer.batch_size,
        masker=masker,
        nr_patches=network_instance.num_patches,
        nr_samples_per_audio=cfg.data.samples_per_audio,
        sr=cfg.data.sr,
        with_noise=cfg.data.with_noise,
        with_rir=cfg.data.with_rir,
    )
    seed_everything(cfg.seed, workers=True)
    trainer.fit(network_instance, data, ckpt_path=cfg.get("ckpt_path", None))


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    gc.collect()
    torch.cuda.empty_cache()
