"""Sequence-aware SELD (single-ACCDOA) evaluation for hear-eval-kit.

If something fails oddly, check the files are all from one revision:

    python3 -m heareval.seld.selfcheck
"""

from .data import (  # noqa: F401
    SELDSequenceEmbeddingDataset,
    accdoa_targets,
    build_embedding_dataset,
    seld_collate,
)
from .heads import ACCDOAHead, masked_accdoa_mse  # noqa: F401
from .lightning import SELDSequenceModule  # noqa: F401