"""
Sequence-aware datasets for SELD / ACCDOA on top of hear-eval-kit.

Why this module exists
----------------------
`heareval.predictions.SplitMemmapDataset` yields ONE frame at a time and the
training DataLoader shuffles those frames.  That is fine for a per-frame MLP
probe but destroys the temporal structure a GRU / self-attention head needs.

The important observation is that the frames are *not* scrambled on disk.
`heareval.embeddings.memmap_embeddings` shuffles the *file list* and then writes
each file's frames contiguously into the memmap, appending exactly one
`(slug, timestamp)` pair per frame to `<split>.filename-timestamps.json` in the
same order.  That JSON is therefore a faithful index into the memmap, and we can
recover per-file frame runs from it *without re-embedding anything*.

Two datasets are provided:

* `SELDSequenceEmbeddingDataset` - frozen encoder / linear-probe style. Reads
  the cached memmap and emits contiguous (T, D) chunks.
* `SELDAudioChunkDataset` - fine-tuning. Reads raw audio and emits fixed-length
  waveform chunks with frame-aligned ACCDOA targets, so the encoder can stay in
  the graph.

Both emit the same dict schema so one Lightning module and one scoring path
serve both.
"""

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
    "adpit_targets",
    "spherical_to_cartesian",
    "build_file_spans",
    "accdoa_targets",
    "frame_labels_from_events",
    "SELDSequenceEmbeddingDataset",
    "SELDAudioChunkDataset",
    "seld_collate",
    "build_embedding_dataset",
    "probe_frame_grid",
    "references_and_timestamps",
]


# --------------------------------------------------------------------------- #
# geometry / label helpers
# --------------------------------------------------------------------------- #
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
    class.  If that happens here the last one wins.  If your task has same-class
    overlap you want multi-ACCDOA + ADPIT, which is out of scope for this module.
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


def adpit_targets(
    frame_labels: Sequence[Sequence[Any]],
    label_to_idx: Dict[str, int],
    nlabels: int,
) -> torch.Tensor:
    """
    Multi-ACCDOA / ADPIT targets, shaped (T, 6, 4, nlabels) to match
    `cls_feature_class.get_adpit_labels_for_file`.

    The six slots are the baseline's dummy tracks: A0 for a class with one
    source in the frame, B0/B1 for two, C0/C1/C2 for three or more. Axis 1 of
    the size-4 dimension is activity, 1:4 are x, y, z. The ADPIT loss gates the
    coordinates by activity, so inactive slots contribute zeros.

    Same-class sources are ordered by (azimuth, elevation) rather than metadata
    order: the permutation-invariant loss makes the ordering irrelevant to
    training, and a deterministic one makes runs reproducible - IntervalTree
    query results come back unordered.
    """
    y = torch.zeros((len(frame_labels), 6, 4, nlabels), dtype=torch.float32)
    slots_for = {1: (0,), 2: (1, 2)}
    for t, frame in enumerate(frame_labels):
        by_class: Dict[int, list] = {}
        for entry in frame:
            by_class.setdefault(label_to_idx[str(entry[0])], []).append(entry[1])
        for cls, doas in by_class.items():
            doas = sorted(doas, key=lambda d: (float(d[0]), float(d[1])))
            slots = slots_for.get(len(doas), (3, 4, 5))
            for slot, doa in zip(slots, doas):
                y[t, slot, 0, cls] = 1.0
                y[t, slot, 1:, cls] = torch.from_numpy(
                    spherical_to_cartesian(doa[0], doa[1])
                )
    return y


def frame_labels_from_events(
    events: Sequence[Dict[str, Any]],
    timestamps_ms: Sequence[float],
) -> List[List[Tuple[str, Sequence[float]]]]:
    """
    Sample a list of {label, start, end, direction} events onto a frame grid.

    Mirrors `heareval.predictions.map_to_frames` so the fine-tuning path and the
    frozen-embedding path produce byte-identical reference structures.
    """
    tree = IntervalTree()
    for ev in events:
        if "direction" not in ev:
            raise KeyError(
                "SELD events need a 'direction' field; this task looks like plain SED."
            )
        tree.addi(ev["start"], ev["end"], (ev["label"], ev["direction"]))
    return [[iv.data for iv in tree[t]] for t in timestamps_ms]


# --------------------------------------------------------------------------- #
# frozen-encoder path: contiguous chunks out of the cached memmap
# --------------------------------------------------------------------------- #
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
              20 ms embedding hop to the 100 ms DCASE label grid. Targets and
              timestamps are pooled here so they line up with the head's output.
              Leave at 1 to evaluate at the embedding rate (the scorer already
              handles pred-rate != label-rate via _nb_pred_frames_1s).
    random_crop : draw a fresh random offset into the file on every __getitem__
              instead of using the fixed tiling. Epoch size is unchanged (one
              crop per tiling slot), so this is free augmentation: the head sees
              different views of the same recording each epoch. Offsets are
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
        multi_accdoa: bool = False,
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
        self.multi_accdoa = multi_accdoa
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
            # np.asarray does NOT copy an ndarray subclass - memmap is one, so
            # asarray returns a base-class view still backed by the file. That
            # silently made in_memory a no-op and passed the memmap's read-only
            # flag through to torch.from_numpy. np.array copies.
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
        self.targets = (
            adpit_targets(raw_labels, label_to_idx, nlabels) if multi_accdoa
            else accdoa_targets(raw_labels, label_to_idx, nlabels)
        )

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
        """Lazy per-worker RNG. torch.initial_seed() differs per worker and per
        epoch, so forked workers don't all draw identical crops."""
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

        y, mask, ts = self._pool(y, mask, ts)
        return {
            "x": x,
            "y": y,
            "mask": mask,
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


# --------------------------------------------------------------------------- #
# fine-tuning path: raw audio chunks
# --------------------------------------------------------------------------- #
def probe_frame_grid(
    module,
    model,
    chunk_samples: int,
    sample_rate: int,
    in_channels: Optional[int] = None,
) -> np.ndarray:
    """
    Run the HEAR module once on silence to learn its frame grid for a chunk of
    `chunk_samples`.  Returns relative timestamps in ms, shape (T,).

    Doing this instead of assuming a hop means the targets are aligned to
    whatever padding / windowing / resampling the encoder actually does. For
    AudioSphere in particular the grid depends on `input_tdim`, the patch
    tstride, and the HEAR_TIMESTAMP_HOP_MS environment variable, so guessing is
    a bad idea.

    `in_channels` defaults to `model.in_channels` when present. The dummy tensor
    is shaped like `heareval.embeddings.AudioFileDataset` output — (B, samples)
    for mono, (B, samples, channels) for multichannel — because that is what the
    encoder was fed when the cached embeddings were made.
    """
    if in_channels is None:
        in_channels = int(getattr(model, "in_channels", 1))
    device = next(model.parameters()).device
    shape = (1, chunk_samples) if in_channels == 1 else (1, chunk_samples, in_channels)
    dummy = torch.zeros(shape, device=device)
    with torch.no_grad():
        _, timestamps = module.get_timestamp_embeddings(dummy, model)
    ts = timestamps[0] if timestamps.dim() == 2 else timestamps
    return ts.detach().cpu().numpy().astype(np.float64)


class SELDAudioChunkDataset(Dataset):
    """
    Fixed-length waveform chunks + frame-aligned ACCDOA targets.

    `frame_grid_ms` must come from `probe_frame_grid` for the same
    `chunk_seconds` and encoder, otherwise labels and predictions drift apart.
    """

    def __init__(
        self,
        task_path: Path,
        split_name: str,
        sample_rate: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        frame_grid_ms: np.ndarray,
        chunk_seconds: float = 6.0,
        hop_seconds: Optional[float] = None,
    ):
        import soundfile as sf  # local import: only the finetune path needs it

        self.sf = sf
        task_path = Path(task_path)
        self.audio_dir = task_path.joinpath(str(sample_rate), split_name)
        self.sample_rate = sample_rate
        self.nlabels = nlabels
        self.label_to_idx = label_to_idx
        self.frame_grid_ms = np.asarray(frame_grid_ms, dtype=np.float64)
        self.chunk_samples = int(round(chunk_seconds * sample_rate))
        self.hop_samples = int(round((hop_seconds or chunk_seconds) * sample_rate))

        self.events: Dict[str, List[Dict[str, Any]]] = json.load(
            open(task_path.joinpath(f"{split_name}.json"))
        )

        self.index: List[Tuple[str, int, int]] = []  # (filename, start_sample, n_valid)
        for filename in self.events:
            path = self.audio_dir.joinpath(filename)
            n = sf.info(str(path)).frames
            for start in range(0, max(n, 1), self.hop_samples):
                self.index.append((filename, start, min(self.chunk_samples, n - start)))
                if start + self.chunk_samples >= n:
                    break

    def __len__(self) -> int:
        return len(self.index)

    def frame_times_ms(self, start_sample: int) -> np.ndarray:
        """Absolute frame timestamps (ms) for the chunk starting at `start_sample`."""
        return self.frame_grid_ms + 1000.0 * start_sample / self.sample_rate

    def __getitem__(self, i: int) -> Dict[str, Any]:
        filename, start, n_valid = self.index[i]
        path = self.audio_dir.joinpath(filename)
        # Same read as heareval.embeddings.AudioFileDataset: no always_2d, no
        # transpose. Multichannel comes back as (samples, channels) and mono as
        # (samples,), which is the layout the encoder saw during embedding
        # extraction. Transposing here would silently feed ambisonic channels in
        # as time.
        audio, sr = self.sf.read(
            str(path), start=start, frames=self.chunk_samples, dtype="float32"
        )
        if sr != self.sample_rate:
            raise ValueError(f"{filename}: expected {self.sample_rate} Hz, got {sr}")

        pad = self.chunk_samples - audio.shape[0]
        if pad > 0:
            pad_width = ((0, pad),) + ((0, 0),) * (audio.ndim - 1)
            audio = np.pad(audio, pad_width)
        x = torch.from_numpy(np.ascontiguousarray(audio))

        ts = self.frame_times_ms(start)
        labels = frame_labels_from_events(self.events[filename], ts)
        y = accdoa_targets(labels, self.label_to_idx, self.nlabels)

        valid_ms = 1000.0 * (start + n_valid) / self.sample_rate
        mask = torch.from_numpy((ts < valid_ms).astype(np.float32))

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "timestamps": torch.from_numpy(ts),
            "filename": filename,
        }


def references_and_timestamps(
    dataset: SELDAudioChunkDataset,
) -> Tuple[Dict[str, List[List[Any]]], Dict[str, List[float]]]:
    """
    Build the `(references, ref_timestamps)` pair that
    `heareval.predictions.get_ref_accdoa_events` expects, on exactly the frame
    grid this dataset predicts on.  Only meaningful when hop == chunk length.
    """
    per_file: Dict[str, List[float]] = {}
    for filename, start, n_valid in dataset.index:
        ts = dataset.frame_times_ms(start)
        valid_ms = 1000.0 * (start + n_valid) / dataset.sample_rate
        per_file.setdefault(filename, []).extend(ts[ts < valid_ms].tolist())

    references, timestamps = {}, {}
    for filename, ts in per_file.items():
        ts = sorted(set(ts))
        timestamps[filename] = ts
        references[filename] = frame_labels_from_events(dataset.events[filename], ts)
    return references, timestamps


# --------------------------------------------------------------------------- #
def seld_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors, keep filenames as a plain list of strings."""
    out: Dict[str, Any] = {
        key: torch.stack([b[key] for b in batch])
        for key in ("x", "y", "mask", "timestamps")
    }
    out["filename"] = [b["filename"] for b in batch]
    return out