#!/usr/bin/env python3
"""Render the main-history ABC experiment into analysis JSON and an offline HTML report."""

import argparse
import html
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "main_history_validate", HERE / "main-history-validate.py"
)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load main-history-validate.py")
validate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validate
spec.loader.exec_module(validate)

TERMS = validate.TERMS
TEMPLATE = HERE / "main-history-report.html"


def stat(values):
    """n/mean/sd/cv for one sample set; cv is None unless defined (n>=2, mean>0)."""
    n = len(values)
    mean = math.fsum(values) / n
    sd = statistics.stdev(values) if n > 1 else None
    cv = sd / mean if sd is not None and mean > 0 else None
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "cv": cv,
        "min": min(values),
        "max": max(values),
    }


def mean_terms(observations):
    return {
        term: math.fsum(o["terms"][term] for o in observations) / len(observations)
        for term in TERMS
    }


def contribution(reference, target):
    """Per-term and total deltas against the reference mean; pp = 100*delta/reference total."""
    ref = mean_terms(reference)
    new = mean_terms(target)
    total = math.fsum(ref.values())
    delta = {term: new[term] - ref[term] for term in TERMS}
    delta["total"] = math.fsum(delta[term] for term in TERMS)
    return {
        "reference_total": total,
        "reference_n": len(reference),
        "target_n": len(target),
        "delta_seconds": delta,
        "pp": {
            key: (100 * value / total if total > 0 else None)
            for key, value in delta.items()
        },
    }


def build_analysis(dataset):
    plan = dataset["plan"]
    commits = [
        {k: c[k] for k in ("id", "index", "sha", "title", "committed_at", "url")}
        for c in plan["commits"]
    ]
    observations = []
    groups = defaultdict(list)  # (cohort, config, revision, uri) -> observations
    for sample in dataset["samples"]:
        for uri, result in sorted(sample["results"].items()):
            row = {
                "job": sample["job"],
                "round": sample["round"],
                "order": sample["order"],
                "config": sample["config"],
                "revision": sample["revision"],
                "commit_index": sample["commit_index"],
                "uri": uri,
                "seconds": result["seconds"],
                "terms": result["terms"],
                "counters": result["counters"],
                "pid": result["pid"],
                "cohort": sample["cohort"],
                "machine_id": sample["machine_id"],
                "started_at": sample["started_at"],
            }
            observations.append(row)
            groups[
                (sample["cohort"], sample["config"], sample["revision"], uri)
            ].append(row)
    inventory = {entry["revision"]: entry for entry in dataset["inventories"]}
    heatmap = {}
    for cohort in dataset["cohorts"]:
        heatmap[cohort] = {}
        for config in plan["configs"]:
            uris = sorted(
                {u for (c, cf, _, u) in groups if c == cohort and cf == config}
            )
            rows = []
            for uri in uris:
                cells = []
                for commit in commits:
                    members = groups.get((cohort, config, commit["id"], uri))
                    in_inventory = commit["id"] in inventory and uri in (
                        inventory[commit["id"]]["uris"] or []
                    )
                    if members:
                        cells.append(stat([m["seconds"] for m in members]))
                    else:
                        cells.append(
                            {
                                "n": 0,
                                "mean": None,
                                "sd": None,
                                "cv": None,
                                "inventory": in_inventory,
                            }
                        )
                rows.append({"uri": uri, "cells": cells})
            heatmap[cohort][config] = rows
    comparisons = {}

    def add(
        cohort, entries, kind, label, config_from, commit_from, config_to, commit_to
    ):
        for uri in sorted({key[3] for key in groups if key[0] == cohort}):
            reference = groups.get((cohort, config_from, commit_from, uri))
            target = groups.get((cohort, config_to, commit_to, uri))
            if reference and target:
                entries.append(
                    {
                        "kind": kind,
                        "label": label,
                        "from": [config_from, commit_from],
                        "to": [config_to, commit_to],
                        "uri": uri,
                        **contribution(reference, target),
                    }
                )

    for cohort in dataset["cohorts"]:
        entries = []
        for config in plan["configs"]:
            for before, after in pairwise(commits):
                add(
                    cohort,
                    entries,
                    "adjacent",
                    f"{config}: {before['id']} -> {after['id']}",
                    config,
                    before["id"],
                    config,
                    after["id"],
                )
        for commit in commits:
            add(
                cohort,
                entries,
                "A->B",
                f"{commit['id']}: A -> B (Valgrind change)",
                "A",
                commit["id"],
                "B",
                commit["id"],
            )
            add(
                cohort,
                entries,
                "B->C",
                f"{commit['id']}: B -> C (integration + isolation, not isolated causal effect)",
                "B",
                commit["id"],
                "C",
                commit["id"],
            )
        comparisons[cohort] = entries
    return {
        "schema_version": 1,
        "experiment": dataset["experiment"],
        "plan_sha256": dataset["plan_sha256"],
        "verdict": dataset["verdict"],
        "errors": dataset["errors"],
        "warnings": dataset["warnings"],
        "coverage": dataset["coverage"],
        "cohorts": dataset["cohorts"],
        "machines": dataset["machines"],
        "inventories": dataset["inventories"],
        "plan": {key: plan[key] for key in plan if key != "jobs"},
        "commits": commits,
        "jobs": [
            {k: v for k, v in sample.items() if k not in ("results", "build")}
            | {
                "benchmark_count": len(sample["results"]),
                "build_provenance": sample["build"],
            }
            for sample in dataset["samples"]
        ],
        "observations": observations,
        "heatmap": heatmap,
        "comparisons": comparisons,
    }


def encode_json(value):
    text = json.dumps(value, allow_nan=False, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        dataset = validate.load_dataset(args.plan, args.input, None, args.allow_partial)
    except (validate.InvalidData, KeyError, TypeError) as error:
        print(f"invalid plan or input: {error}", file=sys.stderr)
        return 1
    for message in dataset["errors"]:
        print(f"ERROR {message}", file=sys.stderr)
    if dataset["errors"] or (
        dataset["verdict"] == "partial" and not args.allow_partial
    ):
        print("refusing to render invalid input", file=sys.stderr)
        return 1
    analysis = build_analysis(dataset)
    args.analysis_json.write_text(
        json.dumps(analysis, indent=1, allow_nan=False) + "\n"
    )
    title = f"{dataset['experiment']}: main-history ABC report"
    document = (
        TEMPLATE.read_text()
        .replace("__TITLE__", html.escape(title))
        .replace("__DATA_JSON__", encode_json(analysis))
    )
    args.output.write_text(document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
