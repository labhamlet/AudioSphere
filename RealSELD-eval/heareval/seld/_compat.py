"""
Resolve hear-eval-kit symbols regardless of how the package is laid out.

`python -m heareval.predictions.runner` means `heareval.predictions` is a
package, so the scoring helpers live in `heareval.predictions.task_predictions`
and are only importable as `heareval.predictions.X` if `__init__.py` re-exports
them. Some checkouts do, some don't. Rather than guess, look each symbol up
across the plausible modules and fail with a message that says what was missing
and where it was looked for.
"""

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
    "TASK_SPECIFIC_PARAM_GRID",
]

_PREDICTIONS = (
    "heareval.predictions.task_predictions",
    "heareval.predictions",
)
_SCORE = ("heareval.score",)


def _find_optional(name: str, modules: Sequence[str], default: Any = None) -> Any:
    """Like _find, but returns `default` instead of raising."""
    for modname in modules:
        try:
            mod = import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    return default


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

# Part of heareval's model-selection procedure: per-task overrides applied on
# top of the parameter grid. Optional, because a checkout may not define it.
TASK_SPECIFIC_PARAM_GRID = _find_optional(
    "TASK_SPECIFIC_PARAM_GRID", _PREDICTIONS, default={}
)