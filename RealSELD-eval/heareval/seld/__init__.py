"""Sequence-aware SELD (ACCDOA) evaluation for hear-eval-kit.

If something fails oddly, check the files are all from one revision:

    python3 -m heareval.seld.selfcheck
"""

from .data import (  # noqa: F401
    SELDAudioChunkDataset,
    adpit_targets,
    SELDSequenceEmbeddingDataset,
    build_embedding_dataset,
    seld_collate,
)
from .events import multi_accdoa_events  # noqa: F401
from .heads import ACCDOAHead, masked_accdoa_mse  # noqa: F401
from .losses import MSELossADPIT  # noqa: F401
from .lightning import SELDFinetuneModule, SELDSequenceModule  # noqa: F401