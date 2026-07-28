#!/usr/bin/env python3
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import torch
from tqdm import tqdm

import heareval.gpu_max_mem as gpu_max_mem

from .train import (
    AUDIOSPHERE_PARITY_GRID,
    AUDIOSPHERE_PROJ_GRID,
    SELD_PARAM_GRID,
    seld_predictions,
)

log = logging.getLogger("heareval.seld")

_task_path_to_logger: Dict[Tuple[str, Path], logging.Logger] = {}

_PRESETS = {
    "default": SELD_PARAM_GRID,
    # "audiosphere" and "seldnet" are the same AudioSphereSELD reproduction.
    "audiosphere": AUDIOSPHERE_PARITY_GRID,
    "seldnet": AUDIOSPHERE_PARITY_GRID,
    "audiosphere-proj": AUDIOSPHERE_PROJ_GRID,
}
_PRESET_NAMES = list(_PRESETS)


def get_logger(task_name: str, log_path: Path) -> logging.Logger:
    """Returns a task level logger"""
    global _task_path_to_logger
    if (task_name, log_path) not in _task_path_to_logger:
        logger = logging.getLogger(f"seld.{task_name}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "seld - %(name)s - %(asctime)s - %(msecs)d - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
        _task_path_to_logger[(task_name, log_path)] = logger
    return _task_path_to_logger[(task_name, log_path)]


def _override_grid(preset: str = "default", **overrides: Optional[Any]):
    base = _PRESETS[preset]
    grid = {k: list(v) for k, v in base.items()}
    clobbered = []
    for key, value in overrides.items():
        if value is None:
            continue
        old = base.get(key)
        if old is not None and list(old) != [value]:
            clobbered.append(f"{key}: {old[0] if len(old) == 1 else old} -> {value}")
        grid[key] = [value]
    return grid, clobbered


@click.command()
@click.argument("task_dirs", nargs=-1, required=True)
@click.option(
    "--preset",
    default="default",
    type=click.Choice(_PRESET_NAMES),
    help="'audiosphere' reproduces seldnet_model.AudioSphereSELD exactly - one "
    "grid point, no search; 'seldnet' is an alias. 'audiosphere-proj' adds "
    "proj_gru (LayerNorm -> Linear(768,256) -> GELU -> Dropout) before the GRU. "
    "(Default: default)",
)
@click.option(
    "--grid-points", default=8, type=click.INT,
    help="Number of grid points for randomized grid search model selection. "
    "(Default: 8)",
)
@click.option(
    "--gpus", default=None if not torch.cuda.is_available() else "[0]", type=str,
    help='GPUs to use, as JSON string (default: "[0]" if any are available).',
)
@click.option(
    "--in-memory", default=True, type=click.BOOL,
    help="Load embeddings in memory, or memmap them from disk. (Default: True)",
)
@click.option(
    "--deterministic", default=True, type=click.BOOL,
    help="Deterministic or non-deterministic. (Default: True)",
)
@click.option(
    "--seed", default=42, show_default=True, type=click.INT,
    help="Random seed. Outputs are suffixed with it when it is not 42, so a "
    "multi-seed sweep does not overwrite itself.",
)
@click.option(
    "--tag", default="", type=str,
    help="Suffix for the done-file and the scores JSON. Needed for ablations: "
    "without it the second run in a sweep sees prediction-done-seld.json and "
    "skips.",
)
@click.option(
    "--shuffle", default=False, type=click.BOOL,
    help="Shuffle tasks? (Default: False)",
)
@click.option(
    "--skip-checks", default=False, type=click.BOOL,
    help="Skip the pre-flight checks. Not recommended: they catch stale files, "
    "frame-rate truncation and grid misalignment, all of which are silent. "
    "(Default: False)",
)
# ---- head architecture ---------------------------------------------------- #
@click.option("--seq-len", default=None, type=click.INT,
              help="Pin the training sequence length, in encoder frames.")
@click.option("--eval-seq-len", default=None, type=click.INT,
              help="Inference chunk length, if it should differ from --seq-len. "
              "Evaluate every arm of a seq_len sweep at one fixed value to "
              "separate 'longer context helps the model' from 'longer chunks "
              "put fewer frames at a boundary'.")
@click.option("--pool-factor", default=None, type=click.INT,
              help="Pin the head's temporal pooling factor.")
@click.option("--hidden-dim", default=None, type=click.INT,
              help="Pin the head width.")
@click.option("--gru-layers", default=None, type=click.INT,
              help="Number of BiGRU layers.")
@click.option("--attn-layers", default=None, type=click.INT,
              help="Number of self-attention blocks after the GRU. 0 removes "
              "them, leaving a pure recurrent head.")
@click.option("--attn-heads", default=None, type=click.INT,
              help="Attention heads. Must divide hidden-dim.")
@click.option("--dropout", default=None, type=click.FLOAT,
              help="Dropout through the head.")
@click.option("--proj-dim", default=None, type=click.INT,
              help="Width of proj_gru feeding the GRU. Defaults to hidden-dim.")
@click.option("--embed-dim", default=None, type=click.INT,
              help="Encoder token width, needed by --freq-pool. 768 for base.")
@click.option(
    "--freq-pool", default=None, type=click.Choice(["attention", "mean", "none"]),
    help="How to collapse the frequency-patch axis of a raw (B,T,F*D) embedding.",
)
@click.option(
    "--gru-merge", default=None, type=click.Choice(["concat", "gate"]),
    help="'gate' reproduces SELDnet's tanh multiplicative merge of the two GRU "
    "directions; 'concat' concatenates them.",
)
@click.option(
    "--use-projection/--no-projection", default=None,
    help="Insert proj_gru before the GRU: LayerNorm -> Linear(token, proj_dim) "
    "-> GELU -> Dropout. Off in the audiosphere preset, since proj_gru is "
    "commented out in seldnet_model.py.",
)
@click.option(
    "--norm-position", default=None,
    type=click.Choice(["pre_pool", "post_pool", "both", "none"]),
    help="Where the LayerNorm goes: input_norm over F*D (pre_pool), proj_gru's "
    "LayerNorm over the pooled token (post_pool), both, or neither.",
)
# ---- optimisation --------------------------------------------------------- #
@click.option(
    "--random-crop/--no-random-crop", default=None,
    help="Draw a fresh random offset into each file per epoch instead of fixed "
    "tiling. Free augmentation, same epoch size. Off in the audiosphere preset "
    "because cls_data_generator tiles.",
)
@click.option("--lr", default=None, type=click.FLOAT, help="Pin the learning rate.")
@click.option("--batch-size", default=None, type=click.INT,
              help="Pin the batch size, in sequences.")
@click.option("--max-epochs", default=None, type=click.INT,
              help="Pin the epoch budget.")
@click.option(
    "--check-val-every-n-epoch", default=None, type=click.INT,
    help="Validation interval. The SELD metric is slow, so on tau-scale tasks "
    "25 is realistic - but the baseline selects per epoch, so the reproduction "
    "presets use 1.",
)
@click.option("--patience", default=None, type=click.INT,
              help="Early-stopping patience, in validation checks, not epochs.")
def runner(
    task_dirs: List[str],
    preset: str = "default",
    grid_points: int = 8,
    gpus: Any = None if not torch.cuda.is_available() else "[0]",
    in_memory: bool = True,
    deterministic: bool = True,
    seed: int = 42,
    tag: str = "",
    shuffle: bool = False,
    skip_checks: bool = False,
    seq_len: Optional[int] = None,
    eval_seq_len: Optional[int] = None,
    pool_factor: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    gru_layers: Optional[int] = None,
    attn_layers: Optional[int] = None,
    attn_heads: Optional[int] = None,
    dropout: Optional[float] = None,
    proj_dim: Optional[int] = None,
    embed_dim: Optional[int] = None,
    freq_pool: Optional[str] = None,
    gru_merge: Optional[str] = None,
    use_projection: Optional[bool] = None,
    norm_position: Optional[str] = None,
    random_crop: Optional[bool] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    check_val_every_n_epoch: Optional[int] = None,
    patience: Optional[int] = None,
) -> None:
    import os

    logging.basicConfig(level=logging.INFO)

    # Required for torch deterministic mode on CUDA; without it the
    # deterministic=True path raises at the first matmul.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


    if gpus is not None:
        gpus = json.loads(gpus)

    if shuffle:
        task_dirs = list(task_dirs)
        random.shuffle(task_dirs)

    grid, clobbered = _override_grid(
        preset,
        seq_len=seq_len,
        eval_seq_len=eval_seq_len,
        pool_factor=pool_factor,
        hidden_dim=hidden_dim,
        gru_layers=gru_layers,
        attn_layers=attn_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        proj_dim=proj_dim,
        embed_dim=embed_dim,
        freq_pool=freq_pool,
        gru_merge=gru_merge,
        use_projection=use_projection,
        norm_position=norm_position,
        random_crop=random_crop,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        patience=patience,
    )

    if clobbered and preset != "default":
        log.warning(
            "--preset %s was overridden on the command line: %s. The preset "
            "exists to reproduce a specific configuration; these flags change "
            "it. Drop them to run the preset as intended.",
            preset, "; ".join(clobbered),
        )

    failures: List[str] = []
    for task_dir in tqdm(task_dirs):
        task_path = Path(task_dir)
        if not task_path.is_dir():
            raise ValueError(f"{task_path} should be a directory")

        suffix = (f"-{tag}" if tag else "") + ("" if seed == 42 else f"-seed{seed}")
        done_file = task_path.joinpath(f"prediction-done-seld{suffix}.json")
        if done_file.exists():
            # We already did this
            continue

        metadata = json.load(task_path.joinpath("task_metadata.json").open())

        log_path = task_path.joinpath("prediction-seld.log")
        logger = get_logger(task_name=metadata["task_name"], log_path=log_path)

        if metadata.get("prediction_type") != "accdoa":
            logger.error(
                f"{task_path.name} has prediction_type="
                f"{metadata.get('prediction_type')!r}, not 'accdoa'. This runner "
                f"is SELD-only; use heareval.predictions.runner instead."
            )
            failures.append(str(task_path))
            continue

        logger.info(f"Computing SELD predictions for {task_path.name}")
        logger.info(f"preset={preset} seed={seed} grid_points={grid_points}")


        # Get embedding sizes for all splits/folds
        embedding_sizes = []
        for split in metadata["splits"]:
            split_path = task_path.joinpath(f"{split}.embedding-dimensions.json")
            embedding_sizes.append(json.load(split_path.open())[1])

        embedding_size = embedding_sizes[0]
        if len(set(embedding_sizes)) != 1:
            raise ValueError("Embedding dimension mismatch among JSON files")

        start = time.time()
        gpu_max_mem.reset()

        seld_predictions(
            embedding_path=task_path,
            embedding_size=embedding_size,
            grid_points=grid_points,
            gpus=gpus,
            in_memory=in_memory,
            deterministic=deterministic,
            grid=grid,
            logger=logger,
            seed=seed,
            tag=tag,
        )
        sys.stdout.flush()
        gpu_max_mem_used = gpu_max_mem.measure()
        logger.info(
            f"DONE took {time.time() - start} seconds to complete seld_predictions"
            f"(embedding_path={task_path}, embedding_size={embedding_size}, "
            f"grid_points={grid_points}, gpus={gpus}, "
            f"gpu_max_mem_used={gpu_max_mem_used}, "
            f"gpu_device_name={gpu_max_mem.device_name()}, in_memory={in_memory}, "
            f"deterministic={deterministic}, preset={preset}, seed={seed})"
        )
        sys.stdout.flush()
        open(done_file, "wt").write(
            json.dumps(
                {
                    "time": time.time() - start,
                    "embedding_path": str(task_path),
                    "embedding_size": embedding_size,
                    "grid_points": grid_points,
                    "gpus": gpus,
                    "gpu_max_mem": gpu_max_mem_used,
                    "gpu_device_name": gpu_max_mem.device_name(),
                    "in_memory": in_memory,
                    "deterministic": deterministic,
                    "preset": preset,
                    "seed": seed,
                    "timestamp_hop_ms": os.environ.get("HEAR_TIMESTAMP_HOP_MS"),
                },
                indent=4,
            )
        )

    if failures:
        log.error(f"{len(failures)} task dir(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    seed = 42
    import os
    import random

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    runner()