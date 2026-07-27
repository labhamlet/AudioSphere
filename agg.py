#!/usr/bin/env python3
"""
Aggregate heareval scores into per-seed or per-dataset tables.

Score tree written by the SLURM array jobs:

    <root>/<model_tag>/seed=<N>/<run_tag>/<task>/<score file>

Naturalistic suite (default):
    metric  test_score_mean if the task was cross-validated, else test_score
    scaled  x100

Localization suite (--seld):
    metrics test_SELD_ER, test_SELD_F, test_SELD_LE, test_SELD_LR
            plus the DCASE composite, taken from heareval's own test_SELD when
            present and computed otherwise
    scaled  x1

Cross-validated tasks report "<metric>_mean" in an aggregated block alongside
per-fold "<metric>" entries; the aggregate always wins.

Usage:
    python3 agg_real_seld.py <root>                                  # HEAR table
    python3 agg_real_seld.py <root> --seld --by-group                # per dataset
    python3 agg_real_seld.py <root> --seld --by-group --per-seed     # per seed
    python3 agg_real_seld.py <root> --seld --explain <file.json>     # dump keys
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path

# ---------------------------------------------------------------- constants --
DEFAULT_SCORE_FILENAME = "test.predicted-scores.json"
SELD_SCORE_FILENAME = "test.predicted-scores.json"

CV_KEY = "test_score_mean"
SINGLE_KEY = "test_score"
PRIMARY_COLUMN = "score"
MEAN_SUFFIX = "_mean"

SEED_RE = re.compile(r"^seed=(\d+)$")

# tau2018 ships each overlap condition as three folds; they average into one row.
FOLD_RE = re.compile(r"^(tau2018-ov\d+)-v[\d.]+-split\d+$")

SELD_METRICS = ["test_SELD_ER", "test_SELD_F", "test_SELD_LE", "test_SELD_LR"]
SELD_COMPOSITE = "SELD_score"
# heareval reports the composite under these names; prefer them over recomputing.
SELD_COMPOSITE_KEYS = ("test_SELD", "test_SELD_SELD")

LOWER_IS_BETTER = {
    "test_SELD_ER": True,    # error rate
    "test_SELD_F": False,    # F-score
    "test_SELD_LE": True,    # localization error, degrees
    "test_SELD_LR": False,   # localization recall
    SELD_COMPOSITE: True,
}

SHORT_NAME = {
    "test_SELD_ER": "ER", "test_SELD_F": "F",
    "test_SELD_LE": "LE", "test_SELD_LR": "LR",
    SELD_COMPOSITE: "SELD",
}

DISPLAY_NAMES = {
    "gram-t-ambi-7ch": "Ambi",
    "gram-t-mono-1ch": "Clean",
}

# --------------------------------------------------------- externally reported
# Per-dataset rows that are not in the score tree. Values print exactly as
# written; F and LR are fractions so they share the scale of the measured
# columns. Shown only with --per-seed: they are NOT folded into the mean of a
# --by-group table, because mixing a differently-produced run into a seed spread
# makes the spread meaningless.
#   (model tag, dataset group): [(seed label, {metric: "value"}, "tasks")]
EXTRA_GROUP_ROWS = {
    ("gram-t-ambi-7ch", "tau2018-ov1"): [
        ("42", {"test_SELD_ER": "0.30", "test_SELD_F": "0.83",
                "test_SELD_LE": "11.7", "test_SELD_LR": "0.821"}, "3/3"),
    ],
    ("gram-t-ambi-7ch", "tau2018-ov2"): [
        ("42", {"test_SELD_ER": "0.38", "test_SELD_F": "0.75",
                "test_SELD_LE": "18.6", "test_SELD_LR": "0.499"}, "3/3"),
    ],
    ("gram-t-ambi-7ch", "tau2018-ov3"): [
        ("42", {"test_SELD_ER": "0.43", "test_SELD_F": "0.708",
                "test_SELD_LE": "23.0", "test_SELD_LR": "0.284"}, "3/3"),
    ],
    ("gram-t-ambi-7ch", "tau2019-v1.0.0-full"): [
        ("42", {"test_SELD_ER": "0.23", "test_SELD_F": "0.864",
                "test_SELD_LE": "12.6", "test_SELD_LR": "0.791"}, "1/1"),
    ],
    ("gram-t-ambi-7ch", "tau2020-v1.0.0-full"): [
        ("42", {"test_SELD_ER": "0.57", "test_SELD_F": "0.474",
                "test_SELD_LE": "22.3", "test_SELD_LR": "0.631"}, "1/1"),
    ],
    ("gram-t-ambi-7ch", "tau2021-v1.0.0-full"): [
        ("42", {"test_SELD_ER": "0.74", "test_SELD_F": "0.214",
                "test_SELD_LE": "37.5", "test_SELD_LR": "0.492"}, "1/1"),
    ],
}

# Naturalistic-suite aggregates, already averaged over tasks. Never applied to a
# localization table.
EXTRA_AGGREGATES = {
    "gram-t-ambi-7ch": [("83.2", "paper AudioSphere(a=0.75)")],
    "gram-t-mono-1ch": [("83.1", "paper AudioSphere-Clean")],
}


# ------------------------------------------------------------------ helpers --
def task_group(task):
    m = FOLD_RE.match(task)
    return m.group(1) if m else task


def seld_composite(vals):
    """(ER + (1-F) + LE/180 + (1-LR)) / 4, or None if a component is missing."""
    try:
        er, f, le, lr = (vals[k] for k in SELD_METRICS)
    except KeyError:
        return None
    return (er + (1.0 - f) + le / 180.0 + (1.0 - lr)) / 4.0


def numeric_keys(obj):
    """Every key in the JSON tree with a plain numeric value (shallowest wins)."""
    found, queue = {}, deque([obj])
    while queue:
        cur = queue.popleft()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    found.setdefault(k, v)
                else:
                    queue.append(v)
        elif isinstance(cur, list):
            queue.extend(cur)
    return found


def numeric_key_paths(obj, prefix=""):
    """[(dotted path, key, value)] for every numeric leaf, depth-first."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((p, k, v))
            else:
                out.extend(numeric_key_paths(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(numeric_key_paths(v, f"{prefix}[{i}]"))
    return out


def explain_file(path, metrics):
    print(f"=== {path}")
    with open(path) as fh:
        data = json.load(fh)
    entries = numeric_key_paths(data)
    print(f"{len(entries)} numeric key(s):")
    for p, _, v in entries:
        print(f"  {p} = {v}")
    dupes = {}
    for p, k, v in entries:
        dupes.setdefault(k, []).append((p, v))
    multi = {k: v for k, v in dupes.items() if len(v) > 1}
    if multi:
        print("\nkeys appearing more than once (the shallowest is used):")
        for k, occ in multi.items():
            print(f"  {k}:")
            for p, v in occ:
                print(f"    {p} = {v}")
    keys = numeric_keys(data)
    print("\nresolution:")
    for m in (metrics or [CV_KEY, SINGLE_KEY]):
        agg = m + MEAN_SUFFIX
        if agg in keys:
            print(f"  {m} -> {agg} = {keys[agg]}")
        elif m in keys:
            print(f"  {m} -> {m} = {keys[m]}  (no {agg} in file)")
        else:
            print(f"  {m} -> NOT FOUND")
    if metrics:
        for k in SELD_COMPOSITE_KEYS:
            if k + MEAN_SUFFIX in keys:
                print(f"  {SELD_COMPOSITE} -> {k}{MEAN_SUFFIX} = {keys[k + MEAN_SUFFIX]}")
                break
            if k in keys:
                print(f"  {SELD_COMPOSITE} -> {k} = {keys[k]}")
                break
        else:
            print(f"  {SELD_COMPOSITE} -> computed from the four metrics")


def read_scores(path, metrics, optional=()):
    """
    (values, keys_present, error, sources)

    `metrics` None auto-detects the HEAR primary score. Keys in `optional` are
    read when present but never reported as missing.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {}, f"unreadable: {exc}", None

    keys = numeric_keys(data)

    if metrics is None:
        for key in (CV_KEY, SINGLE_KEY):
            if key in keys:
                return {PRIMARY_COLUMN: keys[key]}, keys, None, key
        return {}, keys, f"neither {CV_KEY} nor {SINGLE_KEY} found", None

    values, sources, missing = {}, {}, []
    for m in list(metrics) + list(optional):
        agg = m + MEAN_SUFFIX
        if agg in keys:
            values[m], sources[m] = keys[agg], agg
        elif m in keys:
            values[m], sources[m] = keys[m], m
        elif m in metrics:
            missing.append(m)
    if not any(m in values for m in metrics):
        return {}, keys, f"none of {', '.join(metrics)} found", None
    return values, keys, (f"missing {', '.join(missing)}" if missing else None), sources


def decode_path(score_file, root):
    rel = score_file.relative_to(root).parts[:-1]
    seed_at = next((i for i, p in enumerate(rel) if SEED_RE.match(p)), None)
    if seed_at is None or seed_at == 0 or len(rel) - seed_at < 2:
        return None
    return ("/".join(rel[:seed_at]),
            int(SEED_RE.match(rel[seed_at]).group(1)),
            rel[-2], rel[-1])


def external_group_rows(tag, group, columns, composite):
    """(label, {column: (text_or_None, float)}, tasks) for configured rows."""
    out = []
    for label, values, n_tasks in EXTRA_GROUP_ROWS.get((tag, group), []):
        floats = {k: float(v) for k, v in values.items()}
        cells = {c: (values[c], floats[c]) for c in columns if c in values}
        if composite and SELD_COMPOSITE in columns:
            comp = seld_composite(floats)
            if comp is not None:
                cells[SELD_COMPOSITE] = (None, comp)   # computed, not reported
        out.append((label, cells, n_tasks))
    return out


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--seld", action="store_true",
                    help="localization preset: the four SELD metrics plus the "
                         "composite, no scaling, no naturalistic paper rows")
    ap.add_argument("--scores-filename", default=None,
                    help=f"score file to look for (default {DEFAULT_SCORE_FILENAME})")
    ap.add_argument("--metric", action="append", default=[],
                    help="JSON key to aggregate; repeatable. Overrides the preset")
    ap.add_argument("--no-composite", action="store_true",
                    help="omit the SELD composite column")
    ap.add_argument("--by-group", action="store_true",
                    help="one row per dataset (tau2018 folds averaged into their "
                         "overlap condition); no overall mean across datasets")
    ap.add_argument("--per-seed", action="store_true",
                    help="with --by-group: one row per (dataset, seed), and show "
                         "EXTRA_GROUP_ROWS")
    ap.add_argument("--per-task", action="store_true",
                    help="also print the per-task breakdown")
    ap.add_argument("--explain", nargs="?", const="", metavar="PATH",
                    help="dump every numeric key and its path for one score file, "
                         "then exit")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--scale", type=float, default=None,
                    help="multiply values by this (default 100, or 1 with --seld)")
    ap.add_argument("--decimals", type=int, default=None,
                    help="decimal places (default 1, or 3 with --seld)")
    ap.add_argument("--extra-seed", action="append", default=[],
                    metavar="TAG=VALUE[:LABEL]",
                    help="replace the built-in naturalistic value for a model tag")
    ap.add_argument("--extra-in-mean", action="store_true",
                    help="fold EXTRA_GROUP_ROWS into the --by-group mean and +/- as "
                         "additional seeds. The count column then reads e.g. 3x2+1 "
                         "to show how many values are external.")
    ap.add_argument("--no-extra", action="store_true",
                    help="measured rows only; suppress all external rows")
    ap.add_argument("--seeds-intersection", action="store_true",
                    help="restrict every row to the tasks present in all seeds")
    ap.add_argument("--allow-partial", action="store_true",
                    help="average over each model's own task set instead of the "
                         "intersection across models")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"not a directory: {args.root}")

    score_filename = args.scores_filename or (
        SELD_SCORE_FILENAME if args.seld else DEFAULT_SCORE_FILENAME)
    scale = args.scale if args.scale is not None else (1.0 if args.seld else 100.0)
    dp = args.decimals if args.decimals is not None else (3 if args.seld else 1)

    if args.metric:
        metrics, composite = args.metric, False
    elif args.seld:
        metrics, composite = list(SELD_METRICS), not args.no_composite
    else:
        metrics, composite = None, False

    if args.explain is not None:
        target = Path(args.explain) if args.explain else next(
            iter(sorted(args.root.rglob(score_filename))), None)
        if target is None:
            sys.exit(f"no {score_filename} found under {args.root}")
        explain_file(target, metrics)
        return

    # Naturalistic paper rows: only for the default file and the auto-detected
    # primary score.
    use_builtin_extras = (not args.no_extra
                          and not args.seld
                          and metrics is None)
    extras = {k: list(v) for k, v in EXTRA_AGGREGATES.items()} if use_builtin_extras else {}
    for spec in args.extra_seed:
        if "=" not in spec:
            sys.exit(f"--extra-seed needs TAG=VALUE[:LABEL], got {spec!r}")
        tag, rest = spec.split("=", 1)
        value, _, label = rest.partition(":")
        try:
            float(value)
        except ValueError:
            sys.exit(f"--extra-seed value {value!r} is not a number")
        extras[tag] = [(value, label or "external")]

    # ---- collect -------------------------------------------------------------
    optional = SELD_COMPOSITE_KEYS if composite else ()
    runs, problems, seen_keys = [], [], {}
    primary_sources, metric_sources, composite_source = set(), {}, set()

    for score_file in sorted(args.root.rglob(score_filename)):
        decoded = decode_path(score_file, args.root)
        if decoded is None:
            problems.append((str(score_file), "unexpected path layout"))
            continue
        tag, seed, _run_tag, task = decoded
        values, keys, err, sources = read_scores(score_file, metrics, optional)
        seen_keys.update(keys)
        if isinstance(sources, dict):
            for m, k in sources.items():
                metric_sources.setdefault(m, set()).add(k)
        elif sources:
            primary_sources.add(sources)
        if not values:
            problems.append((str(score_file), err))
            continue
        if err:
            problems.append((str(score_file), err))

        values = {k: v * scale for k, v in values.items()}
        if composite:
            reported = next((values[k] for k in SELD_COMPOSITE_KEYS if k in values), None)
            for k in SELD_COMPOSITE_KEYS:
                values.pop(k, None)
            if reported is not None:
                values[SELD_COMPOSITE] = reported
                composite_source.add("reported")
            else:
                comp = seld_composite({k: v / scale for k, v in values.items()})
                if comp is not None:
                    values[SELD_COMPOSITE] = comp * scale
                    composite_source.add("computed")
        runs.append((tag, task, seed, values))

    if not runs:
        if seen_keys:
            print(f"No usable value in {score_filename}. Numeric keys present:",
                  file=sys.stderr)
            for k, v in sorted(seen_keys.items()):
                print(f"  {k} = {v}", file=sys.stderr)
            sys.exit("Pick one or more with --metric.")
        sys.exit(f"no {score_filename} found under {args.root}")

    columns = []
    for _, _, _, values in runs:
        for k in values:
            if k not in columns:
                columns.append(k)

    def fmt(v):
        return "-" if v is None or v != v else f"{v:.{dp}f}"

    # ---- per (model, task), averaged over seeds ------------------------------
    grouped = defaultdict(list)
    for tag, task, seed, values in runs:
        grouped[(tag, task)].append((seed, values))

    per_task, task_rows = {}, []
    for (tag, task), entries in sorted(grouped.items()):
        entries.sort(key=lambda e: e[0])
        means, spreads = {}, {}
        for c in columns:
            vals = [v[c] for _, v in entries if c in v]
            if vals:
                means[c] = statistics.fmean(vals)
                spreads[c] = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        per_task[(tag, task)] = means
        task_rows.append({"model": tag, "task": task, "n_seeds": len(entries),
                          "seeds": ",".join(str(s) for s, _ in entries),
                          "means": means, "stds": spreads})

    models = sorted({t for t, _ in per_task})
    tasks_by_model = {m: {t for (mm, t) in per_task if mm == m} for m in models}

    if args.seeds_intersection:
        seeds_by_model = defaultdict(lambda: defaultdict(set))
        for tag, task, seed, _ in runs:
            seeds_by_model[tag][seed].add(task)
        dropped = 0
        for m in models:
            sets = list(seeds_by_model[m].values())
            if sets:
                shared = set.intersection(*sets)
                dropped += len(tasks_by_model[m] - shared)
                tasks_by_model[m] = shared
        runs = [r for r in runs if r[1] in tasks_by_model[r[0]]]
        per_task = {(t, k): v for (t, k), v in per_task.items()
                    if k in tasks_by_model[t]}
        task_rows = [r for r in task_rows if r["task"] in tasks_by_model[r["model"]]]
        if dropped:
            print(f"note: --seeds-intersection dropped {dropped} (model, task) "
                  f"pair(s) not present in every seed.\n")
        if not runs:
            sys.exit("no tasks are present in every seed")

    common = set.intersection(*tasks_by_model.values())
    all_tasks = set.union(*tasks_by_model.values())

    def header_lines():
        def label(c):
            if c == PRIMARY_COLUMN and primary_sources:
                return f"{c} ({'/'.join(sorted(primary_sources))})"
            out = f"{c}{' (lower better)' if LOWER_IS_BETTER.get(c) else ''}"
            used = metric_sources.get(c, set())
            if used and used != {c}:
                out += f" [{'/'.join(sorted(used))}]"
            return out
        print("metric" + ("s" if len(columns) > 1 else "") + ": "
              + ", ".join(label(c) for c in columns)
              + (f"   scale x{scale:g}" if scale != 1.0 else ""))
        if composite and SELD_COMPOSITE in columns:
            src = ("reported by heareval" if composite_source == {"reported"}
                   else "computed" if composite_source == {"computed"}
                   else "reported where available, else computed")
            print(f"{SELD_COMPOSITE} = (ER + (1-F) + LE/180 + (1-LR)) / 4  [{src}]")

    if args.per_task:
        w_m = max(len(r["model"]) for r in task_rows)
        w_t = max(len(r["task"]) for r in task_rows)
        w_c = max(12, 2 * dp + 6)
        head = (f"{'model':<{w_m}}  {'task':<{w_t}}  {'n':>2}"
                + "".join(f"  {SHORT_NAME.get(c, c):>{w_c}}" for c in columns))
        print(head + "\n" + "-" * len(head))
        for r in task_rows:
            line = f"{r['model']:<{w_m}}  {r['task']:<{w_t}}  {r['n_seeds']:>2}"
            for c in columns:
                mu, sd = r["means"].get(c), r["stds"].get(c)
                cell = "-" if mu is None else (fmt(mu) if sd != sd
                                               else f"{fmt(mu)}±{fmt(sd)}")
                line += f"  {cell:>{w_c}}"
            print(line)
        print()

    # ---- one row per (model, dataset group) ----------------------------------
    if args.by_group:
        groups = {}
        for (tag, task) in per_task:
            groups.setdefault((tag, task_group(task)), []).append(task)

        header_lines()
        if args.per_seed:
            print("one row per seed; tau2018 rows average split1/8/9 within the seed.")
        else:
            print("+/- is the spread across seeds; tau2018 rows average split1/8/9 "
                  "within each seed first.")

        rows_out, saw_external, has_external = [], set(), set()
        for (tag, grp) in sorted(groups):
            tasks_in = sorted(groups[(tag, grp)])
            by_seed = defaultdict(dict)
            for t2, task, seed, values in runs:
                if t2 == tag and task in tasks_in:
                    by_seed[seed][task] = values

            ext = [] if args.no_extra else external_group_rows(tag, grp, columns,
                                                               composite)
            if ext:
                has_external.add((tag, grp))

            if args.per_seed:
                for seed in sorted(by_seed):
                    cells, n_used = {}, 0
                    for c in columns:
                        vals = [by_seed[seed][t][c] for t in tasks_in
                                if t in by_seed[seed] and c in by_seed[seed][t]]
                        n_used = max(n_used, len(vals))
                        cells[c] = fmt(statistics.fmean(vals)) if vals else "-"
                    rows_out.append({"model": DISPLAY_NAMES.get(tag, tag), "group": grp,
                                     "seed": str(seed), "n": f"{n_used}/{len(tasks_in)}",
                                     "cells": cells})
                for label, cells_ext, n_tasks in ext:
                    cells = {c: "-" for c in columns}
                    for c, (text, value) in cells_ext.items():
                        cells[c] = text if text is not None else fmt(value)
                    rows_out.append({"model": DISPLAY_NAMES.get(tag, tag), "group": grp,
                                     "seed": label, "n": n_tasks, "cells": cells})
                    saw_external.add((tag, grp))
                continue

            # Aggregate mode. External rows are folded in only with
            # --extra-in-mean, and the count column records how many.
            fold_ext = ext if args.extra_in_mean else []
            if fold_ext:
                saw_external.add((tag, grp))
            cells, n_seeds, n_ext = {}, 0, 0
            for c in columns:
                per_seed_vals = []
                for seed in sorted(by_seed):
                    vals = [by_seed[seed][t][c] for t in tasks_in
                            if t in by_seed[seed] and c in by_seed[seed][t]]
                    if vals:
                        per_seed_vals.append(statistics.fmean(vals))
                n_seeds = max(n_seeds, len(per_seed_vals))
                added = 0
                for _, cells_ext, _ in fold_ext:
                    if c in cells_ext:
                        per_seed_vals.append(cells_ext[c][1])
                        added += 1
                n_ext = max(n_ext, added)
                if not per_seed_vals:
                    cells[c] = "-"
                elif len(per_seed_vals) == 1:
                    cells[c] = fmt(per_seed_vals[0])
                else:
                    cells[c] = (f"{fmt(statistics.fmean(per_seed_vals))} ± "
                                f"{fmt(statistics.stdev(per_seed_vals))}")
            count = f"{len(tasks_in)}x{n_seeds}" + (f"+{n_ext}" if n_ext else "")
            rows_out.append({"model": DISPLAY_NAMES.get(tag, tag), "group": grp,
                             "seed": "", "n": count, "cells": cells})

        seed_col = args.per_seed
        w_m = max(len(r["model"]) for r in rows_out)
        w_g = max([len(r["group"]) for r in rows_out] + [len("dataset")])
        w_sd = max([len(r["seed"]) for r in rows_out] + [len("seed")])
        n_label = "tasks" if seed_col else "t x s"
        w_n = max([len(r["n"]) for r in rows_out] + [len(n_label)])
        w_c = {c: max([len(r["cells"][c]) for r in rows_out]
                      + [len(SHORT_NAME.get(c, c)) + 1]) for c in columns}
        arrows = {c: ("v" if LOWER_IS_BETTER.get(c) else
                      "^" if c in LOWER_IS_BETTER else "") for c in columns}
        head = (f"{'model':<{w_m}}  {'dataset':<{w_g}}  "
                + (f"{'seed':<{w_sd}}  " if seed_col else "")
                + f"{n_label:>{w_n}}"
                + "".join(f"  {SHORT_NAME.get(c, c) + arrows[c]:>{w_c[c]}}"
                          for c in columns))
        print(head)
        print("-" * len(head))
        prev = None
        for r in rows_out:
            if prev is not None and r["group"] != prev and seed_col:
                print()
            prev = r["group"]
            line = (f"{r['model']:<{w_m}}  {r['group']:<{w_g}}  "
                    + (f"{r['seed']:<{w_sd}}  " if seed_col else "")
                    + f"{r['n']:>{w_n}}")
            for c in columns:
                line += f"  {r['cells'][c]:>{w_c[c]}}"
            print(line)

        if saw_external:
            labels = sorted({lab for key in saw_external
                             for lab, _, _ in EXTRA_GROUP_ROWS[key]})
            print(f"\nnote: seed {'/'.join(labels)} value(s) come from "
                  f"EXTRA_GROUP_ROWS, not the score tree. Their "
                  f"{SHORT_NAME[SELD_COMPOSITE]} is computed from\n      the four "
                  f"values as written, so it inherits their rounding.")
            if not args.per_seed:
                print("      The '+N' in the count column is how many values per cell "
                      "are external, so the\n      +/- mixes them with the measured "
                      "seeds rather than being seed spread alone.")
        elif has_external and not args.no_extra:
            print(f"\nnote: {len(has_external)} dataset(s) have EXTRA_GROUP_ROWS "
                  f"entries, excluded from the mean and +/- above.\n"
                  f"      Pass --extra-in-mean to include them, or --per-seed to see "
                  f"them as their own rows.")

        if problems:
            print(f"\nWARNING: {len(problems)} score file(s) with issues:")
            for path, why in problems:
                print(f"  {why}: {path}")

        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.csv, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["model", "dataset"] + (["seed"] if seed_col else [])
                           + ["tasks"] + columns)
                for r in rows_out:
                    w.writerow([r["model"], r["group"]]
                               + ([r["seed"]] if seed_col else [])
                               + [r["n"]] + [r["cells"][c] for c in columns])
            print(f"\nwrote {args.csv}")
        return

    # ---- one row per (model, seed) -------------------------------------------
    table = []
    for m in models:
        basis = sorted(tasks_by_model[m] if args.allow_partial
                       else tasks_by_model[m] & common)
        by_seed = defaultdict(dict)
        for tag, task, seed, values in runs:
            if tag == m and task in basis:
                by_seed[seed][task] = values

        rows = []
        for value, label in extras.get(m, []):
            rows.append({"display": DISPLAY_NAMES.get(m, m), "source": label,
                         "values": {columns[0]: float(value)},
                         "reported": {columns[0]: value}})
        for seed in sorted(by_seed):
            d = by_seed[seed]
            complete = set(d) >= set(basis)
            means = {}
            for c in columns:
                vals = [d[t][c] for t in basis if t in d and c in d[t]]
                if vals:
                    means[c] = statistics.fmean(vals)
            rows.append({"display": DISPLAY_NAMES.get(m, m),
                         "source": f"HEAR_SEED={seed}"
                                   + ("" if complete else " [INCOMPLETE]"),
                         "values": means, "reported": {}, "seed": seed,
                         "have": set(d), "missing": set(basis) - set(d)})
        for i, r in enumerate(rows, start=1):
            r["seed_row"] = f"Seed {i}"
        table.append((m, basis, rows))

    def cell_text(r, c):
        if c in r["reported"]:
            return r["reported"][c]
        v = r["values"].get(c)
        return "-" if v is None else fmt(v)

    summaries = []
    for m, basis, rows in table:
        cells = {}
        for c in columns:
            vals = [r["values"][c] for r in rows if c in r["values"]]
            if not vals:
                cells[c] = "-"
                continue
            mu = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            cells[c] = fmt(mu) if sd != sd else f"{fmt(mu)} ± {fmt(sd)}"
        counts = {len(r["have"]) for r in rows if "have" in r}
        note = (f"n={len(rows)}, {len(basis)} tasks" if len(counts) <= 1
                else f"n={len(rows)}, {min(counts)}-{max(counts)} tasks")
        summaries.append((note, cells))

    w_d = max(len(r["display"]) for _, _, rows in table for r in rows)
    w_r = max([len(r["seed_row"]) for _, _, rows in table for r in rows] + [4])
    w_s = max([len(r["source"]) for _, _, rows in table for r in rows]
              + [len(s[0]) for s in summaries])
    w_c = {c: max([len(cell_text(r, c)) for _, _, rows in table for r in rows]
                  + [len(s[1][c]) for s in summaries]
                  + [len(SHORT_NAME.get(c, c)) + 1]) for c in columns}
    arrows = {c: ("v" if LOWER_IS_BETTER.get(c) else
                  "^" if c in LOWER_IS_BETTER else "") for c in columns}

    header_lines()
    head = (f"{'model':<{w_d}}  {'row':<{w_r}}  {'source':<{w_s}}"
            + "".join(f"  {SHORT_NAME.get(c, c) + arrows[c]:>{w_c[c]}}"
                      for c in columns))
    print(head)
    print("-" * len(head))
    for (m, basis, rows), (note, cells) in zip(table, summaries):
        for r in rows:
            line = f"{r['display']:<{w_d}}  {r['seed_row']:<{w_r}}  {r['source']:<{w_s}}"
            for c in columns:
                line += f"  {cell_text(r, c):>{w_c[c]}}"
            print(line)
        line = f"{rows[0]['display']:<{w_d}}  {'mean':<{w_r}}  {note:<{w_s}}"
        for c in columns:
            line += f"  {cells[c]:>{w_c[c]}}"
        print(line + "\n")

    # ---- caveats -------------------------------------------------------------
    if all_tasks != common and not args.allow_partial:
        print(f"note: averaged over the {len(common)} task(s) common to all models; "
              f"{len(all_tasks - common)} excluded.")
    if args.seld and EXTRA_AGGREGATES and not args.no_extra:
        print("note: naturalistic paper rows suppressed under --seld.")
    print("note: 'Seed 1/2/...' are display positions; the measured runs keep their "
          "real\n      HEAR_SEED values, shown in the source column.")

    incomplete = [(m, r) for m, _, rows in table for r in rows if r.get("missing")]
    if incomplete:
        print(f"\nWARNING: {len(incomplete)} seed row(s) cover fewer tasks than the "
              f"others, so the rows above are NOT comparable and the mean row mixes "
              f"different task sets:")
        for m, r in incomplete:
            print(f"  {DISPLAY_NAMES.get(m, m)} seed {r['seed']}: {len(r['have'])} of "
                  f"{len(r['have']) + len(r['missing'])} tasks, missing")
            for t in sorted(r["missing"]):
                print(f"    {t}")
        for m, _, rows in table:
            have = [r["have"] for r in rows if "have" in r]
            if not have or len(set(map(frozenset, have))) == 1:
                continue
            shared = set.intersection(*have)
            print(f"  {DISPLAY_NAMES.get(m, m)}: {len(shared)} task(s) in every seed:")
            for t in sorted(shared):
                print(f"    {t}")
        print("  Re-run the missing array indices, or pass --seeds-intersection.")

    identical = [r for r in task_rows
                 if r["n_seeds"] > 1 and any(s == 0.0 for s in r["stds"].values())]
    if identical:
        print(f"\nWARNING: {len(identical)} (model, task) pair(s) scored identically "
              f"across seeds — check that HEAR_SEED reaches the downstream classifier:")
        for r in identical:
            print(f"  {r['model']}  {r['task']}")

    if problems:
        print(f"\nWARNING: {len(problems)} score file(s) with issues:")
        for path, why in problems:
            print(f"  {why}: {path}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "row", "source", "n_tasks"] + columns)
            for (m, basis, rows), (note, cells) in zip(table, summaries):
                for r in rows:
                    w.writerow([r["display"], r["seed_row"], r["source"], len(basis)]
                               + [r["reported"].get(c)
                                  or (f"{r['values'][c]:.6f}" if c in r["values"] else "")
                                  for c in columns])
                w.writerow([rows[0]["display"], "mean", note, len(basis)]
                           + [cells[c] for c in columns])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()