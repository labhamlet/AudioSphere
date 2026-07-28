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
                emit(f, c, (doa[0, f, :, c] + doa[1, f, :, c] + doa[2, f, :, c]) / 3)

    return out
