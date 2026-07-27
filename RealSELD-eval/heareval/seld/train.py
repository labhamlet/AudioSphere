"""
SELD entry points, mirroring `heareval.predictions.task_predictions` but
sequence-aware.

    from heareval.seld.train import seld_predictions, seld_finetune

Both reuse the kit's `available_scores["SELD"]`, its task metadata, and its
split logic, so results sit next to the existing probe results.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader

from ._compat import (
    available_scores,
    get_splits_from_metadata,
    label_vocab_as_dict,
    label_vocab_nlabels,
    load_timestamps,
    map_to_frames,
)
from .data import (
    SELDAudioChunkDataset,
    build_embedding_dataset,
    probe_frame_grid,
    references_and_timestamps,
    seld_collate,
)
from .lightning import SELDFinetuneModule, SELDSequenceModule

__all__ = ["SELD_PARAM_GRID", "AUDIOSPHERE_PARITY_GRID",
           "AUDIOSPHERE_PROJ_GRID", "MULTI_ACCDOA_GRID",
           "seld_predictions", "seld_finetune"]


# seq_len is in encoder frames. At HEAR_TIMESTAMP_HOP_MS=100 that is 100 ms per
# frame, so 20 == SELDNet's 2 s window and 60 == 6 s. Longer helps here: SELDNet
# could afford 2 s because its conv stack had already pooled time 10x, whereas
# the attention layers see one token per frame.

SELD_PARAM_GRID: Dict[str, List[Any]] = {
    "hidden_dim": [256],
    "proj_dim": [None],
    "freq_pool": ["attention"],
    "embed_dim": [768],
    "gru_merge": ["concat"],
    "gru_layers": [2],
    "attn_layers": [2],
    "attn_heads": [8],
    "dropout": [0.05],
    "pool_factor": [1],
    "seq_len": [20, 60],
    "multi_accdoa": [False],
    "lr": [1e-3, 3.2e-4, 1e-4],
    "batch_size": [32],
    "max_epochs": [200],
    "patience": [20],
    "check_val_every_n_epoch": [3],
    "optim": [torch.optim.Adam],
}

AUDIOSPHERE_PARITY_GRID: Dict[str, List[Any]] = {
    "freq_pool": ["attention"],
    "embed_dim": [768],
    "norm_position": ["pre_pool"],
    "use_projection": [False],
    "proj_dim": [None],
    "hidden_dim": [128],
    "gru_layers": [2],
    "gru_merge": ["gate"],
    "attn_layers": [2],
    "attn_heads": [8],
    "fnn_layers": [1],
    "fnn_size": [128],
    "dropout": [0.05],
    "out_dropout": [0.0],
    "pool_factor": [1],
    "seq_len": [60],
    "batch_size": [48],
    "lr": [1e-3],
    "max_epochs": [125],
    "patience": [100],
    "check_val_every_n_epoch": [1],
    "random_crop": [False],
    "optim": [torch.optim.Adam],
}


# AudioSphereSELD with proj_gru enabled: input_norm(F*D) -> attention freq pool
# -> LayerNorm(768) -> Linear(768, 256) -> GELU -> Dropout -> GRU(256, 128).
# Everything else matches the parity preset.
AUDIOSPHERE_PROJ_GRID: Dict[str, List[Any]] = {
    **{k: list(v) for k, v in AUDIOSPHERE_PARITY_GRID.items()},
    "norm_position": ["both"],
    "use_projection": [True],
    "proj_dim": [256],
}


# Multi-ACCDOA with ADPIT. Same head as the AudioSphere parity preset, but three
# tracks per class, so same-class overlap is representable and the localization
# recall ceiling that check_alignment reports no longer applies.
MULTI_ACCDOA_GRID: Dict[str, List[Any]] = {
    **{k: list(v) for k, v in AUDIOSPHERE_PARITY_GRID.items()},
    "multi_accdoa": [True],
    "n_tracks": [3],
    # parameters.py: thresh_unify=15 degrees.
    "thresh_unify": [15],
}


def _seld_scores(metadata: Dict[str, Any], label_to_idx: Dict[str, int]):
    params = metadata.get("scoring_params", {})
    return [
        available_scores["SELD"](
            label_to_idx=label_to_idx,
            doa_threshold=params.get("doa_threshold", 20),
            average=params.get("average", "macro"),
        )
    ]


def _reference_frames(
    embedding_path: Path, metadata: Dict, split_names: List[str]
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Target events + timestamps on the reference frame grid.

    With `_nb_label_frames_1s` set (dynamic sources), `load_timestamps` derives
    the grid from file *lengths*, not from the embeddings - so this grid can
    legitimately differ from the prediction grid. `get_ref_accdoa_events`
    iterates the timestamps dict and indexes the events dict, so the two must
    have identical keys or it raises KeyError; intersect them here.
    """
    timestamps: Dict[str, List[float]] = {}
    events: Dict[str, Any] = {}
    for name in split_names:
        timestamps.update(load_timestamps(embedding_path, metadata, name))
        events.update(json.load(embedding_path.joinpath(f"{name}.json").open()))

    shared = {k for k in events if k in timestamps and len(timestamps[k]) > 0}
    dropped = (set(events) | set(timestamps)) - shared
    if dropped:
        logging.getLogger(__name__).warning(
            "%d file(s) lack either events or timestamps and are excluded from "
            "scoring, e.g. %s", len(dropped), sorted(dropped)[:3]
        )
    events = {k: events[k] for k in shared}
    timestamps = {k: timestamps[k] for k in shared}
    return map_to_frames(events, timestamps, metadata), timestamps


def _trainer(conf: Dict[str, Any], gpus, log_dir: Path, deterministic: bool,
             mode: str = "min"):
    """
    `mode` must follow the score's own direction. The SELD score is
    mean(ER, 1-F, LE/180, 1-LR), so LOWER IS BETTER and mode is "min" -
    the kit derives the same thing from ScoreFunction.maximize. Getting this
    backwards makes ModelCheckpoint keep the worst epoch and EarlyStopping fire
    on improvement, while every logged number still looks plausible.
    """
    checkpoint = ModelCheckpoint(monitor="val_score", mode=mode)
    early_stop = EarlyStopping(
        monitor="val_score",
        mode=mode,
        patience=conf["patience"],
        check_on_train_epoch_end=False,
    )
    kwargs = dict(
        callbacks=[checkpoint, early_stop],
        check_val_every_n_epoch=conf["check_val_every_n_epoch"],
        max_epochs=conf["max_epochs"],
        deterministic=deterministic,
        num_sanity_val_steps=0,
        logger=CSVLogger(str(log_dir)),
    )
    try:
        # Lightning 1.x, matching the rest of hear-eval-kit.
        trainer = pl.Trainer(gpus=gpus, **kwargs)
    except TypeError:
        # Lightning 2.x removed `gpus` in favour of accelerator/devices.
        if not gpus:
            trainer = pl.Trainer(accelerator="cpu", **kwargs)
        else:
            devices = gpus if isinstance(gpus, list) else int(gpus)
            trainer = pl.Trainer(accelerator="gpu", devices=devices, **kwargs)
    return trainer, checkpoint


# --------------------------------------------------------------------------- #
def seld_predictions(
    embedding_path: Path,
    embedding_size: int,
    grid_points: int = 3,
    gpus: Optional[int] = 1,
    logger: Optional[logging.Logger] = None,
    deterministic: bool = True,
    in_memory: bool = True,
    grid: Optional[Dict[str, List[Any]]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Frozen encoder + temporal ACCDOA head over cached HEAR embeddings."""
    logger = logger or logging.getLogger(__name__)
    embedding_path = Path(embedding_path)
    if deterministic:
        pl.seed_everything(seed, workers=True)

    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    if metadata["prediction_type"] != "accdoa":
        raise ValueError(
            f"Expected prediction_type 'accdoa', got {metadata['prediction_type']!r}."
        )

    label_vocab, nlabels = label_vocab_nlabels(embedding_path)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")
    scores = _seld_scores(metadata, label_to_idx)
    # SELD score: lower is better, so maximize is False and mode is "min".
    maximize = bool(getattr(scores[0], "maximize", False))
    mode = "max" if maximize else "min"
    logger.info("model selection on val_score, mode=%s (maximize=%s)", mode, maximize)
    split = get_splits_from_metadata(metadata)[0]

    val_events, val_ts = _reference_frames(embedding_path, metadata, split["valid"])
    test_events, test_ts = _reference_frames(embedding_path, metadata, split["test"])

    def make_loader(conf, names, *, train: bool):
        dataset = build_embedding_dataset(
            embedding_path, names, label_to_idx, nlabels,
            seq_len=conf["seq_len"],
            seq_hop=None,                  # tiling stride == seq_len
            pool_factor=conf["pool_factor"],
            in_memory=in_memory,
            # Random crops for train: free augmentation, same epoch size. Exact
            # tiling for val/test regardless, so every timestamp is covered
            # exactly once - duplicates are silently dropped by the
            # timestamp-keyed dict inside get_accdoa_events.
            # conf["random_crop"] must be honoured here: the reproduction
            # presets set it False because cls_data_generator tiles.
            random_crop=train and bool(conf.get("random_crop", True)),
            multi_accdoa=bool(conf.get("multi_accdoa", False)),
        )
        return DataLoader(
            dataset,
            batch_size=conf["batch_size"],
            shuffle=train,
            collate_fn=seld_collate,
            num_workers=0,
            pin_memory=True,
        )

    confs = list(ParameterGrid(grid or SELD_PARAM_GRID))
    results: List[Dict[str, Any]] = []

    for i, conf in enumerate(confs[: min(grid_points, len(confs))]):
        logger.info("SELD grid point %d/%d: %s", i + 1, grid_points, conf)
        module = SELDSequenceModule(
            embedding_size=embedding_size,
            nlabels=nlabels,
            label_to_idx=label_to_idx,
            scores=scores,
            target_events={"val": val_events, "test": test_events},
            target_timestamps={"val": val_ts, "test": test_ts},
            conf=conf,
            source=metadata.get("source_dynamics", "static"),
            nb_label_frames_1s=metadata.get("_nb_label_frames_1s"),
        )
        trainer, checkpoint = _trainer(
            conf, gpus, Path("logs").joinpath(embedding_path.name), deterministic,
            mode=mode,
        )
        trainer.fit(
            module,
            make_loader(conf, split["train"], train=True),
            make_loader(conf, split["valid"], train=False),
        )
        logger.info(
            "grid point %d finished after %d epoch(s) of a %d budget "
            "(early stopping patience=%d, validating every %d)",
            i + 1, trainer.current_epoch, conf["max_epochs"],
            conf["patience"], conf["check_val_every_n_epoch"],
        )
        if trainer.current_epoch < conf["max_epochs"]:
            logger.warning(
                "stopped early at epoch %d of %d; if you meant to train the "
                "full budget, raise --patience or lower "
                "--check-val-every-n-epoch",
                trainer.current_epoch, conf["max_epochs"],
            )
        if checkpoint.best_model_score is None:
            logger.warning("Grid point %d produced no validation score; skipping", i + 1)
            continue
        results.append({
            "score": float(checkpoint.best_model_score),
            "conf": conf,
            "trainer": trainer,
            "ckpt": checkpoint.best_model_path,
        })

    if not results:
        raise RuntimeError("No grid point produced a validation score.")

    results.sort(key=lambda r: r["score"], reverse=maximize)
    best = results[0]
    for other in results[1:]:          # release the losers before testing
        other["trainer"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Best val SELD score %.4f with %s", best["score"], best["conf"])
    test_results = best["trainer"].test(
        ckpt_path=best["ckpt"],
        dataloaders=make_loader(best["conf"], split["test"], train=False),
    )[0]
    test_results.update({
        "validation_score": best["score"],
        "hparams": {k: str(v) for k, v in best["conf"].items()},
        "embedding_path": str(embedding_path),
        "seed": seed,
    })
    # Seed-suffixed filenames so a multi-seed sweep does not overwrite itself.
    suffix = "" if seed == 42 else f"-seed{seed}"
    embedding_path.joinpath(f"test.predicted-scores-seld{suffix}.json").write_text(
        json.dumps(test_results, indent=4)
    )
    return test_results


# --------------------------------------------------------------------------- #
def seld_finetune(
    task_path: Path,
    embedding_path: Path,
    hear_module: Any,
    encoder: torch.nn.Module,
    embedding_size: int,
    conf: Dict[str, Any],
    gpus: Optional[int] = 1,
    logger: Optional[logging.Logger] = None,
    chunk_seconds: float = 6.0,
    freeze_encoder_epochs: int = 1,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end fine-tuning on raw audio. Requires the RuntimeAudioSphere patch
    in PATCHES.md; without it the encoder silently stays frozen.

    `task_path` is the hearpreprocess task directory (holding
    `<sample_rate>/<split>/*.wav` and `<split>.json`); `embedding_path` supplies
    `task_metadata.json` and `labelvocabulary.csv`.
    """
    logger = logger or logging.getLogger(__name__)
    task_path, embedding_path = Path(task_path), Path(embedding_path)
    if deterministic:
        pl.seed_everything(42, workers=True)

    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    label_vocab, nlabels = label_vocab_nlabels(embedding_path)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")
    scores = _seld_scores(metadata, label_to_idx)
    split = get_splits_from_metadata(metadata)[0]

    sample_rate = encoder.sample_rate
    chunk_samples = int(round(chunk_seconds * sample_rate))
    frame_grid = probe_frame_grid(hear_module, encoder, chunk_samples, sample_rate)
    logger.info("Encoder emits %d frames per %.1fs chunk", len(frame_grid), chunk_seconds)

    # encode() pads up to a whole number of input_tdim windows and then strips an
    # integer number of frames off the tail, so a chunk that is not a whole
    # number of windows leaves a fractional frame and drifts the grid.
    input_tdim = getattr(encoder, "input_size", (None,))[0]
    if input_tdim:
        window_seconds = input_tdim / 100.0        # 10 ms mel hop
        ratio = chunk_seconds / window_seconds
        if abs(ratio - round(ratio)) > 1e-6:
            logger.warning(
                "chunk_seconds=%.3f is not a multiple of the encoder window "
                "(%.2fs); frame timestamps may drift at chunk boundaries.",
                chunk_seconds, window_seconds,
            )

    def dataset(name: str, hop: Optional[float]):
        return SELDAudioChunkDataset(
            task_path, name, sample_rate, label_to_idx, nlabels,
            frame_grid_ms=frame_grid,
            chunk_seconds=chunk_seconds,
            hop_seconds=hop,
        )

    train_ds = dataset(split["train"][0], conf.get("train_hop_seconds"))
    val_ds = dataset(split["valid"][0], None)     # tile exactly at eval
    test_ds = dataset(split["test"][0], None)

    val_events, val_ts = references_and_timestamps(val_ds)
    test_events, test_ts = references_and_timestamps(test_ds)

    def loader(ds, train: bool):
        return DataLoader(
            ds, batch_size=conf["batch_size"], shuffle=train,
            collate_fn=seld_collate, num_workers=conf.get("num_workers", 4),
            pin_memory=True,
        )

    module = SELDFinetuneModule(
        embedding_size,
        nlabels,
        label_to_idx,
        scores,
        {"val": val_events, "test": test_events},
        {"val": val_ts, "test": test_ts},
        conf,
        source=metadata.get("source_dynamics", "static"),
        nb_label_frames_1s=metadata.get("_nb_label_frames_1s"),
        hear_module=hear_module,
        encoder=encoder,
        freeze_encoder_epochs=freeze_encoder_epochs,
    )
    trainer, checkpoint = _trainer(
        conf, gpus, Path("logs").joinpath(embedding_path.name), deterministic
    )
    trainer.fit(module, loader(train_ds, True), loader(val_ds, False))

    test_results = trainer.test(
        ckpt_path=checkpoint.best_model_path, dataloaders=loader(test_ds, False)
    )[0]
    test_results["validation_score"] = float(checkpoint.best_model_score)
    embedding_path.joinpath("test.predicted-scores-seld-finetune.json").write_text(
        json.dumps(test_results, indent=4)
    )
    return test_results