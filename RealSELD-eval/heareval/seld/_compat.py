from __future__ import annotations

from importlib import import_module
from typing import Any, Sequence

__all__ = [
    "get_accdoa_events",
    "get_ref_accdoa_events",
    "get_splits_from_metadata",
    "label_vocab_as_dict",
    "label_vocab_nlabels",
    "load_timestamps",
    "map_to_frames",
    "available_scores",
]

_PREDICTIONS = (
    "heareval.predictions.task_predictions",
    "heareval.predictions",
)
_SCORE = ("heareval.score",)


def _find(name: str, modules: Sequence[str]) -> Any:
    tried = []
    for modname in modules:
        try:
            mod = import_module(modname)
        except ImportError as exc:  # module genuinely absent in this checkout
            tried.append(f"{modname} (import failed: {exc})")
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
        tried.append(f"{modname} (no attribute {name!r})")
    raise ImportError(
        f"Could not resolve {name!r}. Looked in: " + "; ".join(tried) + ". "
        "If your layout differs, add the module to heareval/seld/_compat.py."
    )


get_accdoa_events = _find("get_accdoa_events", _PREDICTIONS)
get_ref_accdoa_events = _find("get_ref_accdoa_events", _PREDICTIONS)
get_splits_from_metadata = _find("get_splits_from_metadata", _PREDICTIONS)
label_vocab_nlabels = _find("label_vocab_nlabels", _PREDICTIONS)
load_timestamps = _find("load_timestamps", _PREDICTIONS)
map_to_frames = _find("map_to_frames", _PREDICTIONS)

# label_vocab_as_dict is defined in heareval.score and imported into
# task_predictions, so it resolves from either.
label_vocab_as_dict = _find("label_vocab_as_dict", _SCORE + _PREDICTIONS)
available_scores = _find("available_scores", _SCORE + _PREDICTIONS)