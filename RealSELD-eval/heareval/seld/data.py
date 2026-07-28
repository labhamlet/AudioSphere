from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from intervaltree import IntervalTree
from torch.utils.data import ConcatDataset, Dataset

__all__ = [
    "FileSpan",
    "spherical_to_cartesian",
    "accdoa_targets",
    "frame_labels_from_events",
    "build_file_spans",
    "SELDSequenceEmbeddingDataset",
    "build_embedding_dataset",
    "seld_collate",
]


# geometry / label helpers
def spherical_to_cartesian(az_deg: float, el_deg: float) -> np.ndarray:
    """(azimuth, elevation) in degrees -> unit Cartesian vector, DCASE convention."""
    az = np.deg2rad(float(az_deg))
    el = np.deg2rad(float(el_deg))
    ce = np.cos(el)
    return np.array([np.cos(az) * ce, np.sin(az) * ce, np.sin(el)], dtype=np.float32)


def accdoa_targets(
    frame_labels: Sequence[Sequence[Any]],
    label_to_idx: Dict[str, int],
    nlabels: int,
) -> torch.Tensor:
    """
    frame_labels[t] is a list of (class_string, (azimuth_deg, elevation_deg)).

    Returns (T, nlabels, 3) float32.  Inactive class -> zero vector, which is the
    standard single-ACCDOA encoding: ||v|| < 0.5 means "inactive".

    NOTE: single-ACCDOA cannot represent two simultaneous instances of the same
    class.  If that happens here the last one wins, and the reference side still
    counts both, so the extra source is an unavoidable false negative.
    `check_alignment` reports how much of the data this affects.
    """
    y = torch.zeros((len(frame_labels), nlabels, 3), dtype=torch.float32)
    for t, frame in enumerate(frame_labels):
        for entry in frame:
            class_str, doa = entry[0], entry[1]
            if len(doa) != 2:
                raise ValueError(
                    f"Expected polar (az, el) direction, got {doa!r}. "
                    "hearpreprocess should emit degrees."
                )
            idx = label_to_idx[str(class_str)]
            y[t, idx] = torch.from_numpy(spherical_to_cartesian(doa[0], doa[1]))
    return y


def frame_labels_from_events(
    events: Sequence[Dict[str, Any]],
    timestamps_ms: Sequence[float],
) -> List[List[Tuple[str, Sequence[float]]]]:
    """
    Sample a list of {label, start, end, direction} events onto a frame grid.

    Mirrors `heareval.predictions.map_to_frames`, including the half-open
    interval, so the reference structures agree exactly.
    """
    tree = IntervalTree()
    for ev in events:
        if "direction" not in ev:
            raise KeyError(
                "SELD events need a 'direction' field; this task looks like plain SED."
            )
        tree.addi(ev["start"], ev["end"], (ev["label"], ev["direction"]))
    return [[iv.data for iv in tree[t]] for t in timestamps_ms]


@dataclass
class FileSpan:
    """A contiguous run of frames belonging to one audio file inside the memmap."""

    filename: str
    start: int
    n_frames: int
    timestamps: np.ndarray  # (n_frames,) milliseconds, strictly increasing


def build_file_spans(embedding_path: Path, split_name: str) -> List[FileSpan]:
    """Recover per-file frame runs from `<split>.filename-timestamps.json`."""
    pairs = json.load(
        open(embedding_path.joinpath(f"{split_name}.filename-timestamps.json"))
    )
    spans: List[FileSpan] = []
    cur_name: Optional[str] = None
    cur_start = 0
    cur_ts: List[float] = []

    for idx, (slug, ts) in enumerate(pairs):
        name = os.path.basename(slug)
        if name != cur_name:
            if cur_name is not None:
                spans.append(
                    FileSpan(cur_name, cur_start, idx - cur_start, np.asarray(cur_ts))
                )
            cur_name, cur_start, cur_ts = name, idx, []
        cur_ts.append(float(ts))
    if cur_name is not None:
        spans.append(
            FileSpan(cur_name, cur_start, len(pairs) - cur_start, np.asarray(cur_ts))
        )

    names = [s.filename for s in spans]
    if len(names) != len(set(names)):
        raise RuntimeError(
            "Frames of at least one file are not contiguous in the memmap. "
            "The sequence index assumes memmap_embeddings() wrote each file in "
            "one run; re-check heareval/embeddings.py if you patched it."
        )
    for s in spans:
        if s.n_frames > 1 and not np.all(np.diff(s.timestamps) > 0):
            raise RuntimeError(f"Timestamps for {s.filename} are not increasing.")
    return spans


class SELDSequenceEmbeddingDataset(Dataset):
    """
    Contiguous (T, D) embedding chunks with frame-aligned ACCDOA targets.

    Parameters
    ----------
    seq_len : frames per chunk, at the *embedding* frame rate.
    seq_hop : stride between chunks. Defaults to seq_len (no overlap).
              Use seq_hop == seq_len for val/test, otherwise a timestamp will be
              predicted twice and the scorer will silently keep only one of them.
    pool_factor : temporal downsampling applied by the head, e.g. 5 to go from a
              20 ms embedding hop to the 100 ms DCASE label grid. Targets,
              timestamps and the mask are pooled here so they line up with the
              head's output. Leave at 1 to evaluate at the embedding rate.
    random_crop : draw a fresh random offset into the file on every __getitem__
              instead of using the fixed tiling. Epoch size is unchanged (one
              crop per tiling slot), so this is free augmentation. Offsets are
              snapped to a multiple of pool_factor so the pooling blocks keep the
              same phase they have at eval time.
              Training only: leave it off for val/test, where the tiling has to
              cover every timestamp exactly once.
    """

    def __init__(
        self,
        embedding_path: Path,
        split_name: str,
        label_to_idx: Dict[str, int],
        nlabels: int,
        seq_len: int = 100,
        seq_hop: Optional[int] = None,
        pool_factor: int = 1,
        in_memory: bool = True,
        drop_incomplete: bool = False,
        random_crop: bool = False,
    ):
        embedding_path = Path(embedding_path)
        if seq_len % pool_factor:
            raise ValueError("seq_len must be a multiple of pool_factor")

        self.seq_len = seq_len
        self.seq_hop = seq_hop or seq_len
        self.pool_factor = pool_factor
        self.nlabels = nlabels
        self.split_name = split_name
        self.random_crop = random_crop
        self._rng: Optional[np.random.Generator] = None

        dim = tuple(
            json.load(
                open(embedding_path.joinpath(f"{split_name}.embedding-dimensions.json"))
            )
        )
        self.n_frames_total, self.embedding_dim = dim
        embeddings = np.memmap(
            filename=embedding_path.joinpath(f"{split_name}.embeddings.npy"),
            dtype=np.float32,
            mode="r",
            shape=dim,
        )
        if in_memory:
            nbytes = int(np.prod(dim)) * 4
            print(f"[seld] loading {split_name} embeddings into RAM: "
                  f"{nbytes / 1e9:.2f} GB", flush=True)
            self.embeddings = np.array(embeddings, dtype=np.float32)
        else:
            self.embeddings = embeddings

        raw_labels = pickle.load(
            open(embedding_path.joinpath(f"{split_name}.target-labels.pkl"), "rb")
        )
        if len(raw_labels) != self.n_frames_total:
            raise RuntimeError("labels and embeddings disagree on frame count")
        self.targets = accdoa_targets(raw_labels, label_to_idx, nlabels)

        self.spans = build_file_spans(embedding_path, split_name)

        # (span_index, offset_within_span)
        self.index: List[Tuple[int, int]] = []
        for si, span in enumerate(self.spans):
            if span.n_frames < seq_len and drop_incomplete:
                continue
            for off in range(0, max(span.n_frames, 1), self.seq_hop):
                if off + seq_len > span.n_frames and drop_incomplete:
                    break
                self.index.append((si, off))
                if off + seq_len >= span.n_frames:
                    break

    def __len__(self) -> int:
        return len(self.index)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _offset(self, span: FileSpan, tiled_offset: int) -> int:
        if not self.random_crop:
            return tiled_offset
        max_off = max(0, span.n_frames - self.seq_len)
        if max_off == 0:
            return 0
        off = int(self.rng.integers(0, max_off + 1))
        return off - (off % self.pool_factor)

    def _pool(self, y: torch.Tensor, mask: torch.Tensor, ts: np.ndarray):
        p = self.pool_factor
        if p == 1:
            return y, mask, ts
        t_out = y.shape[0] // p
        centre = p // 2
        sel = np.arange(t_out) * p + centre
        return y[sel], mask[sel], ts[sel]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        si, tiled_offset = self.index[i]
        span = self.spans[si]
        off = self._offset(span, tiled_offset)
        take = min(self.seq_len, span.n_frames - off)
        lo = span.start + off

        x = torch.zeros((self.seq_len, self.embedding_dim), dtype=torch.float32)
        # copy=True: a memmap slice is read-only, and torch.from_numpy warns
        # (once) about wrapping non-writable memory.
        x[:take] = torch.from_numpy(
            np.array(self.embeddings[lo : lo + take], dtype=np.float32, copy=True)
        )

        y = torch.zeros((self.seq_len,) + self.targets.shape[1:], dtype=torch.float32)
        y[:take] = self.targets[lo : lo + take]

        mask = torch.zeros(self.seq_len, dtype=torch.float32)
        mask[:take] = 1.0

        ts = np.zeros(self.seq_len, dtype=np.float64)
        ts[:take] = span.timestamps[off : off + take]

        input_mask = mask.clone()
        y, mask, ts = self._pool(y, mask, ts)
        return {
            "x": x,
            "y": y,
            "mask": mask,
            "input_mask": input_mask,
            "timestamps": torch.from_numpy(ts),
            "filename": span.filename,
        }


def build_embedding_dataset(
    embedding_path: Path,
    split_names: Sequence[str],
    label_to_idx: Dict[str, int],
    nlabels: int,
    **kwargs: Any,
) -> Dataset:
    """Concatenate several folds into one dataset."""
    parts = [
        SELDSequenceEmbeddingDataset(
            embedding_path, name, label_to_idx, nlabels, **kwargs
        )
        for name in split_names
    ]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def seld_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors, keep filenames as a plain list of strings."""
    keys = ("x", "y", "mask", "timestamps")
    keys += ("input_mask",) if "input_mask" in batch[0] else ()
    out: Dict[str, Any] = {
        key: torch.stack([b[key] for b in batch]) for key in keys
    }
    out["filename"] = [b["filename"] for b in batch]
    return out