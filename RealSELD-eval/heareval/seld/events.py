"""
Multi-ACCDOA inference: turn N predicted tracks per class into DCASE events.

Ported from the track-unification block in `train_seldnet.test_epoch`, with the
same `thresh_unify` semantics. The output structure matches
`heareval.predictions.get_accdoa_events` exactly - `{filename: {frame_idx:
[[class, 0, x, y, z, 0], ...]}}` - so the scoring path is shared with the
single-ACCDOA head.

Why unification is needed: nothing stops two tracks converging on the same
source. Emitting both would count as one true positive and one false positive.
So tracks closer than `thresh_unify` degrees are treated as duplicates of one
source and averaged; tracks further apart are kept as genuinely distinct
sources, which is what buys multi-ACCDOA its same-class recall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

__all__ = ["angular_distance", "multi_accdoa_events"]


def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle in degrees between two Cartesian DOAs. Matches
    `SELD_evaluation_metrics.distance_between_cartesian_coordinates`, including
    the 1e-10 in the norm and the clip before arccos.
    """
    n1 = np.sqrt(np.sum(v1 ** 2) + 1e-10)
    n2 = np.sqrt(np.sum(v2 ** 2) + 1e-10)
    dist = float(np.dot(v1 / n1, v2 / n2))
    return float(np.arccos(np.clip(dist, -1.0, 1.0)) * 180.0 / np.pi)


def _similar(sed_a: bool, sed_b: bool, doa_a: np.ndarray, doa_b: np.ndarray,
             class_idx: int, thresh_unify: float) -> int:
    """Both tracks active for this class and pointing within thresh_unify."""
    if not (sed_a and sed_b):
        return 0
    return int(
        angular_distance(doa_a[:, class_idx], doa_b[:, class_idx]) < thresh_unify
    )


def _decode_file(
    frames: np.ndarray, nb_classes: int, thresh_unify: float
) -> Dict[int, List[List[float]]]:
    """
    frames : (T, n_tracks*3, C)
    Returns {frame_idx: [[class, 0, x, y, z, 0], ...]}
    """
    t_len = frames.shape[0]
    n_tracks = frames.shape[1] // 3
    if n_tracks != 3:
        raise ValueError(
            f"track unification is defined for 3 tracks, got {n_tracks}. "
            "The baseline's determine_similar_location logic is hardcoded to "
            "three; a different count needs its own unification rule."
        )

    # (n_tracks, T, 3, C) and per-track activity (n_tracks, T, C)
    doa = np.stack([frames[:, 3 * k: 3 * (k + 1), :] for k in range(3)], axis=0)
    sed = np.linalg.norm(doa, axis=2) > 0.5

    out: Dict[int, List[List[float]]] = {}

    def emit(frame_idx: int, class_idx: int, vec: np.ndarray) -> None:
        out.setdefault(frame_idx, []).append(
            [int(class_idx), 0, float(vec[0]), float(vec[1]), float(vec[2]), 0]
        )

    for f in range(t_len):
        for c in range(nb_classes):
            f0 = _similar(sed[0, f, c], sed[1, f, c], doa[0, f], doa[1, f], c, thresh_unify)
            f1 = _similar(sed[1, f, c], sed[2, f, c], doa[1, f], doa[2, f], c, thresh_unify)
            f2 = _similar(sed[2, f, c], sed[0, f, c], doa[2, f], doa[0, f], c, thresh_unify)
            total = f0 + f1 + f2

            if total == 0:
                # All three distinct: every active track is its own source.
                for k in range(3):
                    if sed[k, f, c]:
                        emit(f, c, doa[k, f, :, c])
            elif total == 1:
                # One similar pair, averaged; the remaining track stands alone.
                if f0:
                    if sed[2, f, c]:
                        emit(f, c, doa[2, f, :, c])
                    emit(f, c, (doa[0, f, :, c] + doa[1, f, :, c]) / 2)
                elif f1:
                    if sed[0, f, c]:
                        emit(f, c, doa[0, f, :, c])
                    emit(f, c, (doa[1, f, :, c] + doa[2, f, :, c]) / 2)
                else:
                    if sed[1, f, c]:
                        emit(f, c, doa[1, f, :, c])
                    emit(f, c, (doa[2, f, :, c] + doa[0, f, :, c]) / 2)
            else:
                # Two or three pairs similar: all three describe one source.
                emit(f, c, (doa[0, f, :, c] + doa[1, f, :, c] + doa[2, f, :, c]) / 3)

    return out


def multi_accdoa_events(
    predictions: torch.Tensor,
    filenames: Sequence[str],
    timestamps: Sequence[float],
    nb_classes: int,
    thresh_unify: float = 15.0,
) -> Tuple[Dict[str, Dict[int, List[List[float]]]], float, Dict[str, int]]:
    """
    Drop-in counterpart to `heareval.predictions.get_accdoa_events`.

    predictions : (N, n_tracks*3, C) over frames from possibly several files
    Returns (event_dict, mean_timestamp_spacing, max_frame_index_per_file).
    """
    per_file: Dict[str, Dict[float, torch.Tensor]] = {}
    for idx, (filename, timestamp) in enumerate(zip(filenames, timestamps)):
        slug = Path(filename).name
        per_file.setdefault(slug, {})[float(timestamp)] = predictions[idx]

    event_dict: Dict[str, Dict[int, List[List[float]]]] = {}
    max_frames: Dict[str, int] = {}
    diffs: List[float] = []

    for slug, by_time in per_file.items():
        times = sorted(by_time)
        frames = np.stack([by_time[t].detach().cpu().numpy() for t in times])
        event_dict[slug] = _decode_file(frames, nb_classes, thresh_unify)
        max_frames[slug] = len(times) - 1
        if len(times) > 1:
            diffs.append(float(np.mean(np.diff(np.asarray(times)))))

    # get_accdoa_events reports the spacing of whichever file it saw last; the
    # median across files is the same number when the grid is uniform and more
    # robust when it is not.
    diff = float(np.median(diffs)) if diffs else 0.0
    return event_dict, diff, max_frames