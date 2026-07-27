"""
Pre-flight checks for a SELD embedding directory.

    python -m heareval.seld.check_alignment <embedding_path> [--split valid]

Three things are verified, in decreasing order of how badly they bite:

1. The frame rate the scorer will *infer* vs the true one. `get_accdoa_events`
   returns the mean timestamp spacing and `ACCDOAPredictionModel` turns it into
   `int(1000 // diff)`. That is integer division: any hop that does not divide
   1000 exactly yields a wrong frames-per-second, and since `segment_labels`
   groups frames into one-second blocks by counting frames, the blocks drift
   further out of step with the reference the longer the recording runs.

2. Whether the prediction grid and the reference grid describe the same times.
   With `_nb_label_frames_1s` set, the reference grid comes from file *lengths*
   (`range(0, length + res, res)`), not from the embeddings, so the two can
   disagree without anything raising.

3. How much of the data single-ACCDOA structurally cannot represent, i.e.
   frames with two simultaneous sources of the same class. Those cost recall no
   matter how good the model is, and it is worth knowing the ceiling before
   reading a score.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ._compat import (
    label_vocab_as_dict,
    label_vocab_nlabels,
    load_timestamps,
)
from .data import build_file_spans, frame_labels_from_events


def _frame_rate_report(spans) -> tuple[float, bool]:
    hops = [float(np.median(np.diff(s.timestamps))) for s in spans if s.n_frames > 1]
    if not hops:
        print("  no multi-frame files; cannot infer a hop")
        return 0.0, False

    hop = float(np.median(hops))
    spread = max(hops) - min(hops)
    true_rate = 1000.0 / hop
    inferred = int(1000 // hop)

    print(f"  embedding hop           {hop:.4f} ms  (spread across files {spread:.4g})")
    print(f"  true frame rate         {true_rate:.4f} fps")
    print(f"  rate the scorer infers  {inferred} fps   [int(1000 // {hop:g})]")

    ok = abs(true_rate - inferred) < 1e-6
    if ok:
        print("  -> exact, no drift")
    else:
        err = abs(true_rate - inferred) / true_rate
        drift_per_min = 60.0 * err
        print(
            f"  -> TRUNCATED. {err:.2%} rate error means one-second blocks slide "
            f"~{drift_per_min:.2f}s per minute of audio against the reference."
        )
        print(
            "     Fix: set HEAR_TIMESTAMP_HOP_MS to a divisor of 1000 (100 ms "
            "matches the DCASE label grid) and re-extract embeddings."
        )
    return hop, ok


def _grid_report(embedding_path: Path, metadata: dict, split_name: str, spans) -> bool:
    ref_ts = load_timestamps(embedding_path, metadata, split_name)
    source = "file lengths" if metadata.get("_nb_label_frames_1s") else "cached json"
    print(f"  reference grid built from {source}")

    missing = [s.filename for s in spans if s.filename not in ref_ts]
    if missing:
        print(f"  -> {len(missing)} file(s) have embeddings but no reference grid, "
              f"e.g. {missing[:3]}")
        return False

    count_deltas, coincide = [], 0
    for s in spans:
        ref = np.asarray(ref_ts[s.filename], dtype=np.float64)
        count_deltas.append(len(ref) - s.n_frames)
        n = min(len(ref), s.n_frames)
        if n and np.allclose(ref[:n], s.timestamps[:n], atol=1e-6):
            coincide += 1

    deltas = Counter(count_deltas)
    print(f"  frame-count difference (ref - pred): {dict(sorted(deltas.items()))}")
    print(f"  files whose grids coincide exactly:  {coincide}/{len(spans)}")

    if coincide == len(spans):
        print("  -> grids identical; frame indices mean the same instant on both sides")
    else:
        print(
            "  -> grids differ. Not fatal on its own (the scorer segments each side "
            "at its own rate), but it removes your ability to reason frame-by-frame, "
            "and combines badly with any rate truncation above."
        )
    if any(abs(d) > 1 for d in count_deltas):
        print("     Differences beyond +/-1 frame are worth chasing down.")
    return True


def _polyphony_report(embedding_path: Path, metadata: dict, split_name: str) -> None:
    events = json.load(embedding_path.joinpath(f"{split_name}.json").open())
    ref_ts = load_timestamps(embedding_path, metadata, split_name)
    label_rate = int(metadata.get("_nb_label_frames_1s") or 0)

    active = same_class = dropped_frame_level = 0
    polyphony: Counter = Counter()
    seg_ref_doas = seg_achievable = 0

    for filename, evs in events.items():
        if filename not in ref_ts:
            continue
        frames = frame_labels_from_events(evs, ref_ts[filename])

        for frame in frames:
            polyphony[len(frame)] += 1
            if not frame:
                continue
            active += 1
            counts = Counter(str(entry[0]) for entry in frame)
            extra = sum(c - 1 for c in counts.values() if c > 1)
            if extra:
                same_class += 1
                dropped_frame_level += extra

        # Segment-level ceiling. SELDMetrics counts Nref per class per
        # one-second block as max-over-frames of the DOA count, and matches
        # tracks with the Hungarian algorithm. Single-ACCDOA can supply at most
        # one track per class per block, so every reference track beyond the
        # first is an unavoidable false negative. This is the number that
        # actually bounds localization recall.
        if label_rate:
            for start in range(0, len(frames), label_rate):
                block = frames[start : start + label_rate]
                per_class_max: Counter = Counter()
                for frame in block:
                    for cls, count in Counter(str(e[0]) for e in frame).items():
                        per_class_max[cls] = max(per_class_max[cls], count)
                for nb_gt in per_class_max.values():
                    seg_ref_doas += nb_gt
                    seg_achievable += 1        # min(1, nb_gt), and nb_gt >= 1

    total = sum(polyphony.values())
    print(f"  frames                  {total}")
    print(f"  active frames           {active} ({active / max(total, 1):.1%})")
    print("  polyphony distribution  " +
          ", ".join(f"{k}:{v}" for k, v in sorted(polyphony.items())))
    if not active:
        return

    max_poly = max(polyphony)
    print(f"  same-class overlap      {same_class} frames "
          f"({same_class / active:.1%} of active)")
    print(f"  sources dropped         {dropped_frame_level} frame-instances")

    if seg_ref_doas:
        ceiling = seg_achievable / seg_ref_doas
        print(f"  segment-level reference tracks     {seg_ref_doas}")
        print(f"  reachable by single-ACCDOA         {seg_achievable}")
        print(f"  -> LOCALIZATION RECALL CEILING     {ceiling:.3f}")
        print(f"     A perfect single-ACCDOA model scores LR <= {ceiling:.3f} on "
              f"this split. F is also depressed, since the forced misses enter "
              f"as FN. Neither is a bug: it is what the framework can represent.")
        if ceiling < 0.9:
            print(f"     At {ceiling:.3f} the loss is large enough to be worth "
                  f"pricing in before comparing against published numbers, and "
                  f"large enough that multi-ACCDOA + ADPIT would change the "
                  f"conclusion rather than just the decimals.")
    if max_poly >= 3:
        print(f"  NOTE: polyphony reaches {max_poly}. Single-ACCDOA represents at "
              f"most one source per class per frame; it handles different-class "
              f"overlap fine, so the ceiling above is driven only by same-class "
              f"collisions, not by polyphony as such.")


def _composition_report(embedding_path: Path, split_name: str) -> Counter:
    """
    Which source folds/rooms a split is drawn from.

    TAU filenames encode the fold (`fold6_room1_mix001.wav`), so this says
    directly whether hearpreprocess partitioned the data the way the SELDNet
    repo does (`train=[1,2,3,4], val=[5], test=[6]`). If it did not, the two
    pipelines are being scored on different test sets and any comparison
    between their numbers is meaningless regardless of how well the
    architectures match.
    """
    events = json.load(embedding_path.joinpath(f"{split_name}.json").open())
    folds = Counter(name.split("_")[0] for name in events)
    print(f"  files by fold           " +
          ", ".join(f"{k}:{v}" for k, v in sorted(folds.items())))
    return folds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_path", type=Path)
    parser.add_argument("--split", default=None, help="default: every split")
    args = parser.parse_args(argv)

    embedding_path = args.embedding_path
    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    label_vocab, nlabels = label_vocab_nlabels(embedding_path)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    print(f"task            {metadata['task_name']}")
    print(f"prediction_type {metadata['prediction_type']}")
    print(f"source_dynamics {metadata.get('source_dynamics')}")
    print(f"label rate      {metadata.get('_nb_label_frames_1s')} fps")
    print(f"classes         {nlabels}")

    if metadata["prediction_type"] != "accdoa":
        print("\nprediction_type is not 'accdoa'; the SELD path expects it.")
        return 1

    splits = [args.split] if args.split else metadata["splits"]
    all_ok = True
    fold_counts: dict = {}
    for split_name in splits:
        print(f"\n=== {split_name} ===")
        spans = build_file_spans(embedding_path, split_name)
        print(f"  files                   {len(spans)}")

        print("\n frame rate")
        _, rate_ok = _frame_rate_report(spans)
        all_ok &= rate_ok

        print("\n grid alignment")
        all_ok &= _grid_report(embedding_path, metadata, split_name, spans)

        print("\n split composition")
        fold_counts[split_name] = _composition_report(embedding_path, split_name)

        print("\n label density")
        _polyphony_report(embedding_path, metadata, split_name)

    if len(fold_counts) > 1:
        print("\n=== split disjointness ===")
        overlap = False
        names = list(fold_counts)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = set(fold_counts[a]) & set(fold_counts[b])
                if shared:
                    overlap = True
                    print(f"  {a} and {b} share folds {sorted(shared)}")
        if not overlap:
            print("  folds are disjoint across splits")
            for name, folds in fold_counts.items():
                print(f"  {name:6s} <- {sorted(folds)}")
            print("  Compare against the partition your other pipeline uses "
                  "(the SELDNet repo does train=[1,2,3,4] val=[5] test=[6] for "
                  "2021). Different partitions mean different test sets, and "
                  "the numbers are not comparable.")

    print("\n" + ("All checks passed." if all_ok else "Problems found; see above."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())