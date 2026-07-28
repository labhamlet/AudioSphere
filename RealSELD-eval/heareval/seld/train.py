"""
SELD entry points, mirroring `heareval.predictions.task_predictions` but
sequence-aware.

    from heareval.seld.train import seld_predictions

Both reuse the kit's `available_scores["SELD"]`, its task metadata, and its
split logic, so results sit next to the existing probe results.
"""

from __future__ import annotations

import gc
import json
import random

import numpy as np
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
    TASK_SPECIFIC_PARAM_GRID,
    available_scores,
    get_splits_from_metadata,
    label_vocab_as_dict,
    label_vocab_nlabels,
    load_timestamps,
    map_to_frames,
)
from .data import build_embedding_dataset, seld_collate
from .lightning import SELDSequenceModule

__all__ = ["SELD_PARAM_GRID", "AUDIOSPHERE_PARITY_GRID",
           "AUDIOSPHERE_PROJ_GRID", "MLP_PARAM_GRID", "seld_predictions"]


# seq_len is in encoder frames. At HEAR_TIMESTAMP_HOP_MS=100 that is 100 ms per
# frame, so 20 == SELDNet's 2 s window and 60 == 6 s. Longer helps here: SELDNet
# could afford 2 s because its conv stack had already pooled time 10x, whereas
# the attention layers see one token per frame.
SELD_PARAM_GRID: Dict[str, List[Any]] = {
    "hidden_dim": [256],
    "proj_dim": [None],
    # AudioSphere's raw strategy emits F*D = 8*768 = 6144 per frame. Attention
    # pooling over the 8 frequency patches recovers a 768-wide token stream
    # instead of flattening 6144 features into one Linear. FrequencyPool falls
    # back to "none" automatically when in_dim == embed_dim, so this same config
    # is correct for encoders that already pool.
    "freq_pool": ["attention"],
    "embed_dim": [768],
    "gru_merge": ["concat"],
    "gru_layers": [2],
    "attn_layers": [2],
    "attn_heads": [8],
    "dropout": [0.05],
    "pool_factor": [1],
    "branch": ["rnn"],
    "seq_len": [20, 60],
    "lr": [1e-3, 3.2e-4, 1e-4],
    "batch_size": [32],
    "max_epochs": [200],
    "patience": [20],
    "check_val_every_n_epoch": [3],
    "optim": [torch.optim.Adam],
}


AUDIOSPHERE_PARITY_GRID: Dict[str, List[Any]] = {
    # AudioSphereSELD: LayerNorm(F*D) -> attention freq pool -> GRU directly.
    "freq_pool": ["attention"],
    "embed_dim": [768],
    "norm_position": ["pre_pool"],
    "use_projection": [False],
    "proj_dim": [None],
    # parameters.py: rnn_size=128, nb_rnn_layers=2, and SeldModel's tanh
    # multiplicative merge of the two GRU directions.
    "hidden_dim": [128],
    "gru_layers": [2],
    "gru_merge": ["gate"],
    # nb_self_attn_layers=2, nb_heads=8
    "attn_layers": [2],
    "attn_heads": [8],
    # nb_fnn_layers=1, fnn_size=128
    "fnn_layers": [1],
    "fnn_size": [128],
    "dropout": [0.05],
    # AudioSphereSELD has no dropout before the FNN head.
    "out_dropout": [0.0],
    # 100 ms frames already == the label grid, so no temporal pooling, and
    # label_sequence_length=20 is 20 frames.
    "pool_factor": [1],
    "seq_len": [60],
    "eval_seq_len": [None],       # None = evaluate at the training seq_len
    "batch_size": [48],
    "lr": [1e-3],
    "max_epochs": [125],
    # parameters.py sets patience=nb_epochs, so early stopping never fires.
    # Validate every epoch: the repo selects the best epoch by val SELD, and
    # checking every 25th would leave only four candidates out of a hundred.
    "patience": [100],
    "check_val_every_n_epoch": [1],
    # cls_data_generator tiles sequences; it does not random-crop.
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


# heareval.predictions.PARAM_GRID, mapped onto this package. The MLP branch is
# the frame-wise probe the rest of hear-eval-kit uses for every task: no
# temporal context at all, selected by randomized grid search over grid_points
# of the product below.
#
# batch_size is in SEQUENCES here, not frames. seq_len x batch_size = 1024
# reproduces heareval's frame-batch, and since the MLP is per-frame the sequence
# length has no effect on the model - only on how frames are grouped.
MLP_PARAM_GRID: Dict[str, List[Any]] = {
    "branch": ["mlp"],
    "hidden_layers": [1, 2],
    "hidden_dim": [1024],
    "dropout": [0.1],
    "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4],
    "patience": [20],
    "max_epochs": [500],
    "check_val_every_n_epoch": [3],
    "batch_size": [16],
    "seq_len": [64],
    "hidden_norm": ["batchnorm"],
    "norm_after_activation": [False],
    "embedding_norm": ["identity"],
    "initialization": ["xavier_uniform", "xavier_normal"],
    "optim": [torch.optim.Adam],
    "pool_factor": [1],
    "eval_seq_len": [None],
    "random_crop": [False],
}


def _seld_scores(metadata: Dict[str, Any], label_to_idx: Dict[str, int]):
    """
    Build the score functions named in metadata["evaluation"], mirroring
    heareval.predictions.task_predictions.

    "SELD"    - the DCASE2020+ metric, for moving-source tasks. Continuous
                DOAs, location-aware F at a doa_threshold, localization recall
                over matched tracks. Produces the classwise array this package
                reports.

    "OldSELD" - the DCASE2019-era metric, for the STATIC-SCENE tasks (tau2018
                splits, tau2019). Buckets DOA onto an azimuth x elevation grid
                at doa_resolution degrees and scores frame recall rather than
                localization recall, so LE has a floor near the bin size and
                there is no per-class breakdown.
    """
    params = metadata.get("scoring_params", {})
    scores = []
    for name in metadata.get("evaluation", ["SELD"]):
        if name == "SELD":
            scores.append(available_scores["SELD"](
                label_to_idx=label_to_idx,
                doa_threshold=params.get("doa_threshold", 20),
                average=params.get("average", "macro"),
            ))
        elif name == "OldSELD":
            missing = [k for k in ("azimuth_limits", "elevation_limits",
                                   "doa_resolution") if k not in params]
            if missing:
                raise ValueError(
                    f"OldSELD needs {missing} in scoring_params; got "
                    f"{sorted(params)}"
                )
            scores.append(available_scores["OldSELD"](
                label_to_idx=label_to_idx,
                azimuth_list=params["azimuth_limits"],
                elevation_list=params["elevation_limits"],
                _doa_resolution=params["doa_resolution"],
            ))
        else:
            raise ValueError(
                f"metadata['evaluation'] names {name!r}, which this runner does "
                f"not build. Supported: SELD, OldSELD."
            )
    if not scores:
        raise ValueError("metadata['evaluation'] is empty")
    return scores


def _apply_task_specific(
    grid: Dict[str, List[Any]], metadata: Dict[str, Any], logger
) -> Dict[str, List[Any]]:
    """
    heareval's per-task grid override, the last step of its selection procedure:

        if metadata["task_name"] in TASK_SPECIFIC_PARAM_GRID:
            final_grid.update(TASK_SPECIFIC_PARAM_GRID[metadata["task_name"]])
        if "task_specific_param_grid" in metadata.get("evaluation_params", {}):
            final_grid.update(...)

    It exists because the SELD metric is slow: for the tau tasks the kit drops
    to `check_val_every_n_epoch=25, patience=3` rather than validating every
    third epoch for 500 epochs.

    Applied to the MLP branch only. The rnn presets are reproductions of a
    specific published configuration, and silently overriding their epoch
    budget would defeat the point - the whole reason `--preset` warns when the
    command line changes it.
    """
    if grid.get("branch", ["rnn"])[0] != "mlp":
        return grid
    override: Dict[str, Any] = {}
    task_name = metadata.get("task_name")
    if task_name in (TASK_SPECIFIC_PARAM_GRID or {}):
        override.update(TASK_SPECIFIC_PARAM_GRID[task_name])
    override.update(
        metadata.get("evaluation_params", {}).get("task_specific_param_grid", {})
    )
    if not override:
        return grid
    grid = dict(grid)
    grid.update({k: list(v) for k, v in override.items()})
    logger.info("applied heareval's task-specific grid override for %s: %s",
                task_name, override)
    return grid


def _tolerates_nan(scores) -> bool:
    """
    OldSELD computes doa_error as _doa_loss_pred / _doa_loss_pred_cnt, and the
    denominator counts predictions above the 0.5 threshold. A tanh-bounded head
    starts near zero, so the first validations divide 0/0 and the score is NaN.
    Lightning's EarlyStopping defaults to check_finite=True and would stop on
    the first one. NaN never wins model selection either way - ModelCheckpoint
    compares False against it - so tolerating it costs nothing.

    SELDMetrics guards every division (LE[DE_TP == 0] = 180, eps elsewhere), so
    for the moving-source tasks the default stays on and a genuinely diverged
    run still stops.
    """
    return any(type(s).__name__ == "OldSELD" for s in scores)


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
             mode: str = "min", check_finite: bool = True):
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
        check_finite=check_finite,
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
def _aggregate(per_fold: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Mean and std over folds, as heareval.predictions.aggregate_test_results."""
    keys = {k for r in per_fold.values() for k, v in r.items()
            if isinstance(v, (int, float))}
    agg: Dict[str, float] = {}
    for k in sorted(keys):
        vals = [float(r[k]) for r in per_fold.values() if k in r]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return agg


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
    tag: str = "",
) -> Dict[str, Any]:
    """
    Temporal or per-frame ACCDOA head over cached HEAR embeddings.

    Follows heareval.predictions.task_predictions for model selection:
    randomized grid search on the FIRST data split, then the winning
    configuration retrained on every remaining split, each fold tested with its
    own model, and the results aggregated. For a `trainvaltest` task there is
    one split and the fold machinery collapses to a single run; for a k-fold
    task (tau2019) it is k trainings of the selected configuration.
    """
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
    check_finite = not _tolerates_nan(scores)
    maximize = bool(getattr(scores[0], "maximize", False))
    mode = "max" if maximize else "min"
    logger.info("model selection on val_score, mode=%s (maximize=%s)", mode, maximize)

    data_splits = get_splits_from_metadata(metadata)
    logger.info("split_mode=%s -> %d data split(s)",
                metadata.get("split_mode"), len(data_splits))
    for i, sp in enumerate(data_splits):
        logger.info("  split %d: train=%s valid=%s test=%s",
                    i, sp["train"], sp["valid"], sp["test"])

    grid = _apply_task_specific(grid or SELD_PARAM_GRID, metadata, logger)
    confs = list(ParameterGrid(grid))
    # heareval's procedure is a RANDOMIZED grid search: shuffle the full
    # product, evaluate the first grid_points. Seeded, so reproducible.
    random.Random(seed).shuffle(confs)
    confs = confs[: min(grid_points, len(confs))]
    logger.info("grid has %d point(s); evaluating %d",
                len(list(ParameterGrid(grid))), len(confs))

    log_dir = Path("logs").joinpath(embedding_path.name)

    # ---- helpers ---------------------------------------------------------- #
    def references(split):
        val = _reference_frames(embedding_path, metadata, split["valid"])
        test = _reference_frames(embedding_path, metadata, split["test"])
        return {"val": val[0], "test": test[0]}, {"val": val[1], "test": test[1]}

    def make_loader(conf, names, *, train: bool):
        # eval_seq_len decouples the inference chunk length from the training
        # one. seq_len sets both by default, which confounds a seq_len ablation:
        # a longer chunk means fewer frames sit at a boundary with no left
        # context, so longer windows look better at test time whether or not the
        # training benefited.
        seq_len = (
            conf["seq_len"] if train
            else int(conf.get("eval_seq_len") or conf["seq_len"])
        )
        dataset = build_embedding_dataset(
            embedding_path, names, label_to_idx, nlabels,
            seq_len=seq_len,
            seq_hop=None,                  # tiling stride == seq_len
            pool_factor=conf["pool_factor"],
            in_memory=in_memory,
            # Random crops for train: free augmentation, same epoch size. Exact
            # tiling for val/test regardless, so every timestamp is covered
            # exactly once - duplicates are silently dropped by the
            # timestamp-keyed dict inside get_accdoa_events.
            random_crop=train and bool(conf.get("random_crop", True)),
        )
        return DataLoader(
            dataset,
            batch_size=conf["batch_size"],
            shuffle=train,
            collate_fn=seld_collate,
            num_workers=0,
            pin_memory=True,
        )

    def build_module(conf, events, timestamps):
        return SELDSequenceModule(
            embedding_size=embedding_size,
            nlabels=nlabels,
            label_to_idx=label_to_idx,
            scores=scores,
            target_events=events,
            target_timestamps=timestamps,
            conf=conf,
            source=metadata.get("source_dynamics", "static"),
            nb_label_frames_1s=metadata.get("_nb_label_frames_1s"),
        )

    def fit(conf, split, events, timestamps, label: str):
        module = build_module(conf, events, timestamps)
        trainer, checkpoint = _trainer(conf, gpus, log_dir, deterministic,
                                       mode=mode, check_finite=check_finite)
        trainer.fit(module,
                    make_loader(conf, split["train"], train=True),
                    make_loader(conf, split["valid"], train=False))
        logger.info(
            "%s finished after %d epoch(s) of a %d budget "
            "(patience=%d, validating every %d)",
            label, trainer.current_epoch, conf["max_epochs"],
            conf["patience"], conf["check_val_every_n_epoch"],
        )
        score = checkpoint.best_model_score
        path = checkpoint.best_model_path
        # A fitted Trainer holds its dataloaders, which hold the in-memory
        # embeddings - several GB per fit. Release before the next one.
        del trainer, checkpoint, module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (None if score is None else float(score)), path

    def test(conf, split, events, timestamps, ckpt):
        module = build_module(conf, events, timestamps)
        trainer, _ = _trainer(conf, gpus, log_dir, deterministic,
                              mode=mode, check_finite=check_finite)
        out = trainer.test(module, ckpt_path=ckpt,
                           dataloaders=make_loader(conf, split["test"], train=False))[0]
        del trainer, module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    # ---- 1. grid search on the first split -------------------------------- #
    events0, timestamps0 = references(data_splits[0])
    candidates: List[Dict[str, Any]] = []
    for i, conf in enumerate(confs):
        logger.info("grid point %d/%d: %s", i + 1, len(confs), conf)
        score, ckpt = fit(conf, data_splits[0], events0, timestamps0,
                          f"grid point {i + 1}")
        if score is None:
            logger.warning("grid point %d produced no validation score; skipping",
                           i + 1)
            continue
        candidates.append({"score": score, "conf": conf, "ckpt": ckpt})

    if not candidates:
        raise RuntimeError("No grid point produced a validation score.")
    candidates.sort(key=lambda r: r["score"], reverse=maximize)
    best = candidates[0]
    logger.info("best val score %.4f with %s", best["score"], best["conf"])

    # ---- 2. retrain the winner on the remaining splits -------------------- #
    per_split = [{"split": data_splits[0], "ckpt": best["ckpt"],
                  "val": best["score"], "events": events0,
                  "timestamps": timestamps0}]
    for i, split in enumerate(data_splits[1:], start=1):
        logger.info("retraining the selected config on split %d/%d",
                    i + 1, len(data_splits))
        events, timestamps = references(split)
        score, ckpt = fit(best["conf"], split, events, timestamps, f"split {i}")
        per_split.append({"split": split, "ckpt": ckpt, "val": score,
                          "events": events, "timestamps": timestamps})

    # ---- 3. test each fold with its own model ----------------------------- #
    per_fold: Dict[str, Dict[str, Any]] = {}
    for entry in per_split:
        fold = "|".join(entry["split"]["test"])
        logger.info("testing fold %s", fold)
        result = test(best["conf"], entry["split"], entry["events"],
                      entry["timestamps"], entry["ckpt"])
        result["validation_score"] = entry["val"]
        per_fold[fold] = result

    results: Dict[str, Any] = dict(per_fold)
    if len(per_fold) > 1:
        results["aggregated_scores"] = _aggregate(per_fold)
        logger.info("aggregated over %d folds: %s",
                    len(per_fold), results["aggregated_scores"])
    else:
        # Single split: keep the metrics at the top level too, so a
        # trainvaltest task reads the same as it always has.
        results.update(next(iter(per_fold.values())))

    results.update({
        "hparams": {k: str(v) for k, v in best["conf"].items()},
        "embedding_path": str(embedding_path),
        "seed": seed,
        "tag": tag,
        "n_folds": len(per_fold),
    })

    # Suffix so a sweep does not overwrite itself: --tag for ablations,
    # --seed for repeats.
    suffix = (f"-{tag}" if tag else "") + ("" if seed == 42 else f"-seed{seed}")
    embedding_path.joinpath(f"test.predicted-scores-seld{suffix}.json").write_text(
        json.dumps(results, indent=4)
    )
    return results