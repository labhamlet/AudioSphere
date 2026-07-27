"""
Lightning modules for SELD/ACCDOA on hear-eval-kit.

Two modules, one scoring path:

* `SELDSequenceModule`  - frozen encoder, cached embeddings, temporal head.
                          This is the supported path.
* `SELDFinetuneModule`  - encoder in the graph, trained end to end. Requires the
                          RuntimeAudioSphere patch in PATCHES.md, without which
                          the encoder silently will not train.

Both reuse `heareval.predictions.get_accdoa_events` / `get_ref_accdoa_events`
and the existing `SELD` score function, so numbers are directly comparable to
the frame-wise probe already in the kit.

Targets PyTorch Lightning 1.x (`validation_epoch_end`, `Trainer(gpus=...)`),
matching the rest of hear-eval-kit. On Lightning 2.x, rename the two
`*_epoch_end` hooks to `on_*_epoch_end` and buffer step outputs yourself.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch import nn

from .events import multi_accdoa_events
from .heads import ACCDOAHead, masked_accdoa_mse
from .losses import MSELossADPIT
from .reporting import (
    accdoa_magnitude_range,
    install_classwise_hook,
    is_degenerate,
    reset_classwise,
    format_classwise_table,
    format_counts,
    format_magnitude_table,
    recover_classwise,
)

__all__ = ["SELDSequenceModule", "SELDFinetuneModule"]


# --------------------------------------------------------------------------- #
def _scalar_hparams(conf: Dict[str, Any]) -> Dict[str, Any]:
    """CSVLogger chokes on classes and callables; keep only loggable scalars."""
    return {k: v for k, v in conf.items() if isinstance(v, (int, float, str, bool))}


def _flatten_valid_frames(outputs: Sequence[Dict[str, Any]]):
    """(B, T, ...) step outputs -> flat per-frame lists, padded frames dropped."""
    preds, filenames, timestamps = [], [], []
    for out in outputs:
        pred = out["prediction"].detach().cpu()
        mask = out["mask"].detach().cpu() > 0.5
        ts = out["timestamps"].detach().cpu()
        for b, filename in enumerate(out["filename"]):
            keep = mask[b]
            if not keep.any():
                continue
            preds.append(pred[b][keep])
            timestamps.extend(ts[b][keep].tolist())
            filenames.extend([filename] * int(keep.sum()))
    if not preds:
        raise RuntimeError("No valid frames in this epoch - check your masks.")
    return torch.cat(preds, dim=0), filenames, timestamps


class _SELDBase(pl.LightningModule):
    """Shared plumbing: head construction, steps, and SELD scoring."""

    def __init__(
        self,
        embedding_size: int,
        nlabels: int,
        label_to_idx: Dict[str, int],
        scores: List[Any],
        target_events: Dict[str, Dict[str, Any]],
        target_timestamps: Dict[str, Dict[str, List[float]]],
        conf: Dict[str, Any],
        source: str = "static",
        nb_label_frames_1s: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(_scalar_hparams(conf))
        self._optim = conf.get("optim", torch.optim.Adam)
        self._lr = conf["lr"]

        self.multi_accdoa = bool(conf.get("multi_accdoa", False))
        self.n_tracks = int(conf.get("n_tracks", 3)) if self.multi_accdoa else 1
        self.thresh_unify = float(conf.get("thresh_unify", 15.0))
        self._adpit = MSELossADPIT() if self.multi_accdoa else None

        # Build the head from the conf by signature rather than a hand-written
        # list of conf.get(...) calls. A stale copy of this file silently
        # dropping a new head option is otherwise invisible - the model builds,
        # trains, and reports plausible numbers with the wrong architecture.
        # The parameter count is the only tell, so it is printed below.
        head_args = set(inspect.signature(ACCDOAHead.__init__).parameters)
        head_args -= {"self", "in_dim", "nlabels", "n_tracks"}
        head_kwargs = {k: conf[k] for k in head_args if k in conf}
        ignored = sorted(head_args - set(head_kwargs))

        self.head = ACCDOAHead(
            in_dim=embedding_size,
            nlabels=nlabels,
            n_tracks=self.n_tracks,
            **head_kwargs,
        )

        # NOT self.print(): that touches self.trainer, which raises here.
        n_params = sum(p.numel() for p in self.head.parameters())
        print(f"[seld] head input {embedding_size} -> {self.head.freq_pool}", flush=True)
        print(f"[seld] head parameters: {n_params:,}", flush=True)
        print(f"[seld] head config: " +
              ", ".join(f"{k}={head_kwargs[k]}" for k in sorted(head_kwargs)) +
              (f"  [defaulted: {', '.join(ignored)}]" if ignored else ""),
              flush=True)

        self.scores = scores
        self.target_events = target_events        # {"val": ..., "test": ...}
        self.target_timestamps = target_timestamps
        self.label_to_idx = label_to_idx
        self.nlabels = nlabels
        self.source = source
        self._nb_label_frames_1s = nb_label_frames_1s
        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        self._warned_no_classwise = False
        # Lightning 1.x passes step outputs into *_epoch_end; 2.x removed that
        # argument and expects you to buffer them yourself. Buffer either way
        # and implement both hook spellings.
        #
        # Both hooks fire on 1.9, so scoring must be idempotent within an epoch:
        # a per-epoch flag rather than "the buffer is empty by then", which
        # would depend on Lightning's internal hook ordering. Getting that wrong
        # means running the SELD metric twice per validation - Hungarian
        # matching over every segment, so far from free - and logging each
        # score twice.
        # NOT self._buffers: that is nn.Module's registered-buffer OrderedDict,
        # and overwriting it makes .to(device) try to call .to() on a list.
        self._epoch_outputs: Dict[str, List[Dict[str, Any]]] = {"val": [], "test": []}
        self._scored: Dict[str, bool] = {"val": False, "test": False}

    # -- steps -------------------------------------------------------------- #
    def _loss(self, pred, target, mask):
        if self._adpit is not None:
            return self._adpit(pred, target, mask)
        return masked_accdoa_mse(pred, target, mask)

    def training_step(self, batch, batch_idx):
        pred = self(batch["x"], batch["mask"])
        loss = self._loss(pred, batch["y"], batch["mask"])
        self.log("train_loss", loss)
        return loss

    def _eval_step(self, batch, name: str):
        pred = self(batch["x"], batch["mask"])
        self.log(
            f"{name}_loss",
            self._loss(pred, batch["y"], batch["mask"]),
            prog_bar=True,
        )
        out = {
            "prediction": pred.detach(),
            "mask": batch["mask"],
            "timestamps": batch["timestamps"],
            "filename": batch["filename"],
        }
        self._epoch_outputs[name].append(out)
        return out

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def _maybe_score(self, name: str, outputs) -> None:
        """Score exactly once per epoch, whichever hook gets there first."""
        if self._scored[name] or not outputs:
            return
        self._scored[name] = True
        self._score(name, outputs)
        self._epoch_outputs[name].clear()

    def on_validation_epoch_start(self):
        self._scored["val"] = False
        self._epoch_outputs["val"].clear()

    def on_test_epoch_start(self):
        self._scored["test"] = False
        self._epoch_outputs["test"].clear()

    # -- Lightning 1.x: outputs are passed in ------------------------------- #
    def validation_epoch_end(self, outputs):
        self._maybe_score("val", outputs)

    def test_epoch_end(self, outputs):
        self._maybe_score("test", outputs)

    # -- Lightning 2.x: score from our own buffer --------------------------- #
    def on_validation_epoch_end(self):
        self._maybe_score("val", self._epoch_outputs["val"])

    def on_test_epoch_end(self):
        self._maybe_score("test", self._epoch_outputs["test"])

    # -- scoring ------------------------------------------------------------ #
    def _score(self, name: str, outputs: Sequence[Dict[str, Any]]) -> None:
        from ._compat import get_accdoa_events, get_ref_accdoa_events

        prediction, filenames, timestamps = _flatten_valid_frames(outputs)

        if self.multi_accdoa:
            pred_events, diff, max_frames = multi_accdoa_events(
                prediction, filenames, timestamps, self.nlabels,
                thresh_unify=self.thresh_unify,
            )
        else:
            pred_events, diff, max_frames = get_accdoa_events(
                prediction, filenames, timestamps, self.nlabels
            )
        ref_events, max_ref_frames = get_ref_accdoa_events(
            self.target_events[name],
            self.target_timestamps[name],
            self.nlabels,
            label_to_idx=self.label_to_idx,
        )

        # int(1000 // diff) is the kit's own inference and it truncates. An 80 ms
        # hop becomes 12 fps instead of 12.5 and the one-second segmentation
        # drifts against the reference. check_alignment.py catches this before
        # training; warn here too in case it was skipped.
        nb_pred_frames_1s = int(1000 // diff)
        if abs(1000.0 / diff - nb_pred_frames_1s) > 1e-6:
            self.print(
                f"WARNING: embedding hop {diff:g} ms does not divide 1000; the "
                f"scorer will use {nb_pred_frames_1s} fps instead of "
                f"{1000.0 / diff:.3f}. Set HEAR_TIMESTAMP_HOP_MS to a divisor of "
                f"1000 (100 recommended) and re-extract embeddings."
            )
        nb_label_frames_1s = (
            nb_pred_frames_1s if self.source == "static" else self._nb_label_frames_1s
        )

        # An empty comparison scores 0.75, not 1.0: with Nref == 0 the metric's
        # ER = (S+D+I)/(Nref+eps) is 0/eps = 0, F and LR are 0/eps = 0, and LE
        # hits the DE_TP == 0 guard at 180 -> mean(0, 1, 1, 1) = 0.75. That
        # looks like a mediocre model rather than a broken pipeline, so count
        # both sides and say so explicitly.
        n_pred = sum(len(d) for f in pred_events.values() for d in f.values())
        n_ref = sum(len(d) for f in ref_events.values() for d in f.values())
        self.print(
            f"  detections: {n_pred} predicted over {len(pred_events)} files, "
            f"{n_ref} reference over {len(ref_events)} files"
        )
        if n_ref == 0:
            self.print(
                "  ERROR: no reference tracks. The score will read ~0.75 and is "
                "meaningless. The events and timestamps dicts disagreed on "
                "filenames, or the split JSON was empty."
            )
        elif n_pred == 0:
            self.print(
                "  ERROR: no predictions above the 0.5 magnitude threshold, so "
                "nothing can match. Check the magnitude table below: if the max "
                "is under 0.5 the loaded weights are from an untrained or badly "
                "selected epoch."
            )

        # Anything captured on a previous epoch is stale from here on.
        reset_classwise()

        end_scores: Dict[str, float] = {}
        for score in self.scores:
            ret = score(
                pred_events,
                ref_events,
                nb_label_frames_1s,
                nb_pred_frames_1s,
                max_frames,
                max_ref_frames,
            )
            if isinstance(ret, tuple):
                end_scores[f"{name}_{score}"] = ret[0][1]
                for sub, value in ret:
                    end_scores[f"{name}_{score}_{sub}"] = value
            elif isinstance(ret, float):
                end_scores[f"{name}_{score}"] = ret
            else:
                raise ValueError(f"Unexpected score return type {type(ret)}")

        self.log(f"{name}_score", end_scores[f"{name}_{self.scores[0]}"], logger=True)
        for key, value in end_scores.items():
            self.log(key, value, prog_bar=True, logger=True)

        self._report(name, prediction, end_scores)

        if name == "test":
            self.test_predictions = {
                "prediction": prediction,
                "predicted_events": pred_events,
                "target_events": ref_events,
                "filenames": filenames,
                "timestamps": timestamps,
            }

    def _report(self, name: str, prediction: torch.Tensor,
                end_scores: Dict[str, float]) -> None:
        """Per-epoch diagnostics, in the shape train_seldnet.py prints them."""
        # log_scores emits both "{split}_{score}" and "{split}_{score}_{sub}",
        # so the primary subscore arrives twice (SELD and SELD_SELD). Strip the
        # repeated score name and keep the first value per label.
        score_name = str(self.scores[0])
        labels: Dict[str, float] = {}
        for key, value in end_scores.items():
            label = key[len(name) + 1:] if key.startswith(f"{name}_") else key
            if label.startswith(f"{score_name}_"):
                label = label[len(score_name) + 1:]
            labels.setdefault(label, value)
        header = ", ".join(f"{k}: {v:0.3f}" for k, v in labels.items())
        self.print(f"epoch {self.current_epoch} [{name}] {header}")

        mag_min, mag_max = accdoa_magnitude_range(prediction, self.n_tracks)
        self.print(format_magnitude_table(mag_min, mag_max, self.idx_to_label))

        # Cross-check against the reported aggregate: under macro averaging the
        # scalar is the mean of the classwise SELD row, so a mismatch means the
        # captured array belongs to a different SELDMetrics instance.
        primary = end_scores.get(f"{name}_{self.scores[0]}")
        classwise = recover_classwise(self.scores[0], self.nlabels, primary)
        if classwise is None:
            # Wrap SELDMetrics.compute_seld_scores so the next epoch captures it.
            # Cannot help this epoch: the score has already been computed.
            target = install_classwise_hook()
            if not self._warned_no_classwise:
                self._warned_no_classwise = True
                if target:
                    self.print(
                        f"  (classwise breakdown: hooked SELDMetrics in {target}; "
                        f"it will appear from the next validation onwards, and "
                        f"only if its mean matches the reported score)"
                    )
                else:
                    self.print(
                        "  (classwise breakdown unavailable: could not locate "
                        "SELDMetrics to hook. One-line fix in PATCHES.md "
                        "section 4.)"
                    )
            return

        counts = format_counts()
        if counts:
            self.print(counts)
        self.print(format_classwise_table(classwise, self.idx_to_label))
        if is_degenerate(classwise):
            self.print(
                f"  ERROR: {name} scoring produced no reference tracks. The "
                f"reported {name}_score is meaningless - do not use this run."
            )
        # Log per-class SELD so the CSV is analysable afterwards; the other four
        # rows stay in the printed table to keep the CSV width sane.
        for c in range(self.nlabels):
            self.log(f"{name}_SELD_class{c}", float(classwise[4][c]), logger=True)

    def configure_optimizers(self):
        return self._optim(self.parameters(), lr=self._lr)


# --------------------------------------------------------------------------- #
class SELDSequenceModule(_SELDBase):
    """Frozen encoder: cached embeddings in, temporal ACCDOA head on top."""

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.head(x, mask)


# --------------------------------------------------------------------------- #
class SELDFinetuneModule(_SELDBase):
    """
    End-to-end: raw audio -> HEAR encoder -> temporal ACCDOA head.

    `hear_module` is the imported HEAR module (the thing exposing
    `get_timestamp_embeddings`), `encoder` is the loaded model. We call the
    module directly rather than going through `heareval.embeddings.Embedding`,
    whose wrapper runs under `torch.no_grad()` and would detach the encoder.
    """

    def __init__(self, *args, hear_module: Any, encoder: nn.Module,
                 freeze_encoder_epochs: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hear_module = hear_module
        self.encoder = encoder
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self._grad_checked = False

    def _encoder_frozen(self) -> bool:
        return self.current_epoch < self.freeze_encoder_epochs

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        frozen = self._encoder_frozen()

        # Encoders that gate their own no_grad (RuntimeAudioSphere does, once
        # patched) need telling. No-op on encoders without the attribute.
        if hasattr(self.encoder, "freeze_encoder"):
            self.encoder.freeze_encoder = frozen

        if frozen:
            with torch.no_grad():
                emb, _ = self.hear_module.get_timestamp_embeddings(audio, self.encoder)
            return emb.detach()

        emb, _ = self.hear_module.get_timestamp_embeddings(audio, self.encoder)
        if not self._grad_checked:
            self._grad_checked = True
            if self.training and not emb.requires_grad:
                raise RuntimeError(
                    "Encoder embeddings came back detached, so nothing upstream "
                    "of the head will train. RuntimeAudioSphere.encode() wraps "
                    "its forward in torch.no_grad() - see PATCHES.md."
                )
        return emb

    def forward(self, audio: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.head(self.embed(audio), mask)

    def configure_optimizers(self):
        """Discriminative learning rates: the pretrained encoder gets less."""
        encoder_lr = self.hparams.get("encoder_lr", self._lr * 0.1)
        groups = [
            {"params": self.head.parameters(), "lr": self._lr},
            {"params": self.encoder.parameters(), "lr": encoder_lr},
        ]
        return self._optim(groups, lr=self._lr)