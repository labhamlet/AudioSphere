from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "install_classwise_hook",
    "reset_classwise",
    "format_counts",
    "is_degenerate",
    "accdoa_magnitude_range",
    "recover_classwise",
    "format_magnitude_table",
    "format_classwise_table",
]

_CLASSWISE_ATTRS = (
    "classwise_results",
    "_classwise_results",
    "classwise_scores",
    "_classwise_scores",
)
_METRICS_ATTRS = ("seld_metrics", "_seld_metrics", "metrics", "_metrics")

_ROW_NAMES = ("ER", "F", "LE", "LR", "SELD")

_LAST_CLASSWISE: Dict[str, Any] = {"value": None}
_LAST_COUNTS: Dict[str, Any] = {"value": None}
_COUNTER_ATTRS = ("_Nref", "_TP", "_FP", "_FP_spatial", "_FN",
                  "_DE_TP", "_DE_FP", "_DE_FN", "_S", "_D", "_I")
_HOOK_STATE: Dict[str, Any] = {"installed": False, "target": None}


def install_classwise_hook() -> Optional[str]:
    if _HOOK_STATE["installed"]:
        return _HOOK_STATE["target"]

    # The score function has already been constructed by the time we run, so
    # whichever module defines SELDMetrics is in sys.modules.
    for mod_name, module in list(sys.modules.items()):
        if module is None or not mod_name.split(".")[0] in ("heareval", "hearbaseline"):
            if "SELD" not in mod_name:
                continue
        cls = getattr(module, "SELDMetrics", None)
        if cls is None or not hasattr(cls, "compute_seld_scores"):
            continue
        if getattr(cls.compute_seld_scores, "_seld_wrapped", False):
            _HOOK_STATE.update(installed=True, target=mod_name)
            return mod_name

        original = cls.compute_seld_scores

        def wrapped(self, _original=original):
            result = _original(self)
            try:
                classwise = np.asarray(result[-1])
                if classwise.ndim == 2:
                    _LAST_CLASSWISE["value"] = classwise
                    _LAST_COUNTS["value"] = {
                        a.lstrip("_"): np.asarray(getattr(self, a, 0.0)).copy()
                        for a in _COUNTER_ATTRS
                    }
            except Exception:
                pass
            return result

        wrapped._seld_wrapped = True
        cls.compute_seld_scores = wrapped
        _HOOK_STATE.update(installed=True, target=mod_name)
        return mod_name

    _HOOK_STATE["installed"] = True
    return None


def accdoa_magnitude_range(prediction: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    prediction : (N, C, 3) flattened per-frame ACCDOA vectors.
    Returns (min, max) magnitude per class, each (C,).
    """
    mag = prediction.norm(dim=-1)  # (N, C)
    return (
        mag.min(dim=0).values.cpu().numpy(),
        mag.max(dim=0).values.cpu().numpy(),
    )


def reset_classwise() -> None:
    _LAST_CLASSWISE["value"] = None
    _LAST_COUNTS["value"] = None


def format_counts(prefix: str = "  ") -> Optional[str]:
    counts = _LAST_COUNTS.get("value")
    if counts is None:
        return None
    tot = {k: float(np.sum(v)) for k, v in counts.items()}
    lines = [
        f"{prefix}metric counters: Nref={tot['Nref']:.0f}  "
        f"TP={tot['TP']:.0f}  FP={tot['FP']:.0f}  FP_spatial={tot['FP_spatial']:.0f}  "
        f"FN={tot['FN']:.0f}",
        f"{prefix}                 DE_TP={tot['DE_TP']:.0f}  DE_FN={tot['DE_FN']:.0f}  "
        f"S={tot['S']:.0f}  D={tot['D']:.0f}  I={tot['I']:.0f}",
    ]
    if tot["Nref"] == 0:
        lines.append(f"{prefix}                 Nref=0 -> every score below is vacuous")
    return "\n".join(lines)


def recover_classwise(
    score_fn: Any,
    nb_classes: int,
    expected_seld: Optional[float] = None,
    tol: float = 1e-4,
) -> Optional[np.ndarray]:
    """
    Best-effort recovery of the (5, nb_classes) classwise array from a
    ScoreFunction that has just been called. Returns None if unavailable.

    `expected_seld` is the scalar the ScoreFunction reported. Under
    average="macro" the aggregate is literally `SELD_scr.mean()` over the
    classwise row, so the two must agree exactly. If they do not, the array came
    from a different SELDMetrics instance than the one that produced the score,
    and printing it would be worse than printing nothing.
    """
    def _validate(arr: np.ndarray) -> Optional[np.ndarray]:
        if arr.ndim != 2:
            return None
        if arr.shape == (nb_classes, 5):
            arr = arr.T
        if arr.shape != (5, nb_classes):
            return None
        if expected_seld is not None:
            if abs(float(np.mean(arr[4])) - float(expected_seld)) > tol:
                return None
        return arr

    captured = _LAST_CLASSWISE.get("value")
    if captured is not None:
        checked = _validate(np.asarray(captured))
        if checked is not None:
            return checked

    for attr in _CLASSWISE_ATTRS:
        value = getattr(score_fn, attr, None)
        if value is not None:
            checked = _validate(np.asarray(value))
            if checked is not None:
                return checked

    for attr in _METRICS_ATTRS:
        metrics = getattr(score_fn, attr, None)
        if metrics is not None and hasattr(metrics, "compute_seld_scores"):
            try:
                *_, classwise = metrics.compute_seld_scores()
            except Exception:
                continue
            checked = _validate(np.asarray(classwise))
            if checked is not None:
                return checked
    return None


def format_magnitude_table(
    mag_min: np.ndarray,
    mag_max: np.ndarray,
    idx_to_label: Dict[int, str],
    prefix: str = "  ",
) -> str:
    lines = [f"{prefix}ACCDOA magnitude per class:",
             f"{prefix}Class\tmin\tmax\tlabel"]
    for c in range(len(mag_min)):
        lines.append(
            f"{prefix}{c}\t{mag_min[c]:0.4f}\t{mag_max[c]:0.4f}\t"
            f"{idx_to_label.get(c, '?')}"
        )
    if float(np.max(mag_max)) <= 0.5:
        lines.append(
            f"{prefix}WARNING: no class reaches the 0.5 detection threshold, so "
            f"SED output is empty and the metrics below are vacuous. Expected "
            f"for the first epochs of a tanh-bounded head; if it persists, the "
            f"head is collapsing to zero."
        )
    return "\n".join(lines)


def is_degenerate(classwise: np.ndarray) -> bool:
    return bool(np.allclose(classwise[0], 0.0) and np.allclose(classwise[3], 0.0))


def format_classwise_table(
    classwise: np.ndarray,
    idx_to_label: Dict[int, str],
    prefix: str = "  ",
) -> str:
    """classwise : (5, C) rows ER, F, LE, LR, SELD - as compute_seld_scores builds it."""
    nb_classes = classwise.shape[1]
    lines = [f"{prefix}Classwise results:",
             f"{prefix}Class\tER\tF\tLE\tLR\tSELD\tlabel"]
    for c in range(nb_classes):
        lines.append(
            f"{prefix}{c}\t"
            + "\t".join(f"{classwise[r][c]:0.2f}" for r in range(5))
            + f"\t{idx_to_label.get(c, '?')}"
        )
    means = [float(np.mean(classwise[r])) for r in range(5)]
    lines.append(
        f"{prefix}mean\t" + "\t".join(f"{m:0.2f}" for m in means)
    )
    if is_degenerate(classwise):
        lines.append(
            f"{prefix}ERROR: ER=0 with LR=0 is arithmetically impossible when "
            f"references exist - nothing detected gives ER=1, not 0. Both can "
            f"only be zero when Nref=0, so the metric received no ground truth. "
            f"The aggregate score is wrong too, not just this table. Check the "
            f"detection counts printed above."
        )
        return "\n".join(lines)

    worst = int(np.argmax(classwise[4]))   # highest SELD == worst class
    lines.append(
        f"{prefix}worst class: {worst} ({idx_to_label.get(worst, '?')}) "
        f"SELD={classwise[4][worst]:0.2f}, LR={classwise[3][worst]:0.2f}"
    )
    return "\n".join(lines)