#!/usr/bin/env python3
"""Validate main-history ABC captures against the frozen plan.

The parsing here is the single source of truth for both the validation
verdict and `main-history-report.py`, which imports `load_dataset`.
Captures may span measurement workflows, but must share one frozen build
workflow and byte-identical per-commit build metadata. Process counts use
active benchmark targets rather than executables that can legitimately skip
all measurements on the current platform or in Simulation mode.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_VERSION = 1
BATCH = "main-history"
COUNTERS = ("Ct", "Cl", "Dr", "Dw", "D1mr", "D1mw", "DLmr", "DLmw", "ILmr")
TERMS = ("compute", "l1", "ll", "memory")
CPU_HZ = 3_600_000_000
L1_CYCLES = 5
LL_CYCLES = 40
MEMORY_CYCLES = 200
VALGRIND_PREFIX = Path("/opt/stack-isolation/valgrind")
SAMPLE_NAME = re.compile(r"sample-(\d+)-([A-Z])-(c\d+)")
RESULT_KEYS = {"seconds", "counters", "file", "pid"}
# lscpu rows that describe the live clock rather than the CPU identity.
VOLATILE_CPU_FIELDS = re.compile(r"MHz|BogoMIPS|scaling", re.IGNORECASE)
REQUIRED_PROVENANCE = (
    "cpu_model",
    "cpu",
    "kernel",
    "glibc",
    "os_release",
    "package_versions_sha256",
    "started_at",
    "runner_version",
    "runner_sha256",
    "machine_id",
    "boot_id",
    "workflow_run_id",
    "build_workflow_run_id",
    "build_metadata_sha256",
    "plan_sha256",
    "binary_hashes",
    "valgrind_hashes",
    "valgrind_version",
    "targets",
    "command",
    "working_directory",
    "benchmark_commands",
    "environment",
    "isolation_env",
    "benchmark_count",
    "measured_process_count",
)
BUILD_FIELDS = (
    "schema_version",
    "experiment",
    "plan_sha256",
    "revision",
    "commit_index",
    "source_sha",
    "canonical_workload",
    "rust_toolchain",
    "rustc_version",
    "old_cargo_codspeed_version",
    "old_codspeed_crate_version",
    "codspeed_rust_revision",
    "input_fingerprints",
    "benchmark_files",
    "binaries",
)


class InvalidData(ValueError):
    pass


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise InvalidData(f"duplicate JSON keys {duplicates}")
    return dict(pairs)


def _reject_constant(name):
    raise InvalidData(f"non-finite JSON constant {name}")


def load_json(path):
    """Parse JSON rejecting duplicate keys and NaN/Infinity."""
    try:
        return json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (InvalidData, ValueError) as error:
        raise InvalidData(f"{path}: {error}") from error


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def score_terms(counters):
    """The exact capture score, split into its four additive cycle terms (seconds)."""
    cycles = {
        "compute": math.sqrt(counters["Ct"] * counters["Cl"]) / 100,
        "l1": L1_CYCLES
        * (counters["Dr"] + counters["Dw"] - counters["D1mr"] - counters["D1mw"]),
        "ll": LL_CYCLES
        * (counters["D1mr"] + counters["D1mw"] - counters["DLmr"] - counters["DLmw"]),
        "memory": MEMORY_CYCLES
        * (counters["DLmr"] + counters["DLmw"] + counters["ILmr"]),
    }
    return {term: value / CPU_HZ for term, value in cycles.items()}


def cpu_identity(manifest):
    rows = manifest["cpu"].get("lscpu") if isinstance(manifest["cpu"], dict) else None
    if not isinstance(rows, list):
        raise InvalidData("cpu is not lscpu --json output")
    stable = {}
    for row in rows:
        field = row["field"]
        if VOLATILE_CPU_FIELDS.search(field):
            continue
        if field in stable:
            raise InvalidData(f"duplicate CPU identity field {field!r}")
        stable[field] = row["data"]
    return {"cpu_model": manifest["cpu_model"], "lscpu": stable}


def cohort_of(manifest):
    identity = {
        "cpu": cpu_identity(manifest),
        "kernel": manifest["kernel"],
        "glibc": manifest["glibc"],
        "os_release": manifest["os_release"],
        "package_versions_sha256": manifest["package_versions_sha256"],
    }
    key = sha256_bytes(json.dumps(identity, sort_keys=True).encode())[:12]
    return key, identity


def normalized_command(command):
    """Profile folders are per-sample paths; everything else must match exactly."""
    result = list(command)
    for index, part in enumerate(result[:-1]):
        if part == "--profile-folder":
            result[index + 1] = "<profile-folder>"
    return result


class Validation:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)


def validate_plan(plan):
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise InvalidData("plan schema_version must be 1")
    configs = plan["configs"]
    commits = plan["commits"]
    if configs != ["A", "B", "C"]:
        raise InvalidData(f"plan configs must be ABC, got {configs}")
    for index, commit in enumerate(commits):
        if commit["index"] != index or commit["id"] != f"c{index:02}":
            raise InvalidData(f"plan commit {index} has inconsistent id/index")
    jobs = plan["jobs"]
    expected = len(commits) * len(configs) * plan["repetitions"]
    if [job["job"] for job in jobs] != list(range(expected)):
        raise InvalidData(f"plan jobs must be 0..{expected - 1} in order")
    by_commit = {commit["id"]: commit for commit in commits}
    rounds = defaultdict(list)
    for job in jobs:
        commit = by_commit.get(job["revision"])
        if (
            commit is None
            or job["commit_index"] != commit["index"]
            or job["source_sha"] != commit["sha"]
            or job["config"] not in configs
        ):
            raise InvalidData(f"plan job {job['job']} does not match its commit")
        rounds[job["round"]].append(job)
    pairs = {(commit["id"], config) for commit in commits for config in configs}
    for round_number in range(plan["repetitions"]):
        entries = rounds.get(round_number, [])
        if {(job["revision"], job["config"]) for job in entries} != pairs or len(
            entries
        ) != len(pairs):
            raise InvalidData(f"plan round {round_number} is not one full ABC pass")
        if sorted(job["order"] for job in entries) != list(range(len(pairs))):
            raise InvalidData(f"plan round {round_number} orders are not 0..N-1")
    if set(rounds) != set(range(plan["repetitions"])):
        raise InvalidData("plan contains rounds beyond repetitions")


def discover_samples(inputs):
    samples = []
    for root in inputs:
        if not root.is_dir():
            raise InvalidData(f"input root {root} is not a directory")
        samples.extend(
            path.parent
            for path in sorted(root.rglob("manifest.json"))
            if path.parent.name.startswith("sample-")
        )
    return samples


def load_build(capture_dir, manifest, plan, plan_sha, commit, check, label):
    path = capture_dir / "build-metadata.json"
    if not path.is_file():
        check.error(f"{label}: missing build-metadata.json")
        return None
    data = path.read_bytes()
    if sha256_bytes(data) != manifest.get("build_metadata_sha256"):
        check.error(
            f"{label}: build-metadata.json bytes do not match build_metadata_sha256"
        )
    build = load_json(path)
    missing = [field for field in BUILD_FIELDS if field not in build]
    if missing:
        check.error(f"{label}: build metadata missing {missing}")
        return None
    pins = plan["pins"]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment": plan["experiment"],
        "plan_sha256": plan_sha,
        "revision": commit["id"],
        "commit_index": commit["index"],
        "source_sha": commit["sha"],
        "canonical_workload": plan["canonical_workload"],
        "rust_toolchain": pins["rust_toolchain"],
        "old_cargo_codspeed_version": pins["old_cargo_codspeed_version"],
        "old_codspeed_crate_version": pins["old_codspeed_crate_version"],
        "codspeed_rust_revision": pins["codspeed_rust_revision"],
    }
    for field, value in expected.items():
        if build[field] != value:
            check.error(f"{label}: build {field}={build[field]!r}, expected {value!r}")
    if str(build["rustc_version"]).split()[:2] != ["rustc", pins["rust_toolchain"]]:
        check.error(
            f"{label}: rustc_version {build['rustc_version']!r} is not {pins['rust_toolchain']}"
        )
    if str(build.get("workflow_run_id")) != str(manifest["build_workflow_run_id"]):
        check.error(
            f"{label}: build workflow_run_id differs from captured build_workflow_run_id"
        )
    for key in ("input_fingerprints", "benchmark_files"):
        if not build[key] or not isinstance(build[key], dict):
            check.error(f"{label}: build {key} must be a nonempty mapping")
    for integration in ("old", "new"):
        binary = build["binaries"].get(integration) or {}
        targets = binary.get("targets")
        digests = binary.get("sha256")
        if not targets or not isinstance(digests, dict) or set(targets) != set(digests):
            check.error(
                f"{label}: build binaries.{integration} targets/sha256 inconsistent"
            )
    return build


def check_results(results, manifest, check, label):
    if not isinstance(results, dict) or not results:
        check.error(f"{label}: results.json must be a nonempty URI mapping")
        return None
    observations = {}
    target_pids = defaultdict(set)
    for uri, result in results.items():
        if "::" not in uri:
            check.error(f"{label}: result key {uri!r} is not a benchmark URI")
            continue
        target = Path(uri.split("::", 1)[0]).stem
        if target not in manifest["targets"]:
            check.error(f"{label}: unplanned measuring target {target!r}")
            continue
        if not isinstance(result, dict) or set(result) != RESULT_KEYS:
            check.error(f"{label}: {uri} must have exactly {sorted(RESULT_KEYS)}")
            continue
        counters = result["counters"]
        if not isinstance(counters, dict) or set(COUNTERS) - set(counters):
            check.error(f"{label}: {uri} missing simulation counters")
            continue
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters.values()
        ):
            check.error(f"{label}: {uri} counters must be nonnegative integers")
            continue
        terms = score_terms(counters)
        seconds = result["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            check.error(f"{label}: {uri} seconds must be finite and positive")
            continue
        if any(value < 0 for value in terms.values()):
            check.error(f"{label}: {uri} has a negative cost term {terms}")
            continue
        if not math.isclose(sum(terms.values()), seconds, rel_tol=1e-12, abs_tol=0):
            check.error(
                f"{label}: {uri} seconds {seconds} != score terms {sum(terms.values())}"
            )
            continue
        target_pids[target].add(result["pid"])
        observations[uri] = {
            "seconds": seconds,
            "terms": terms,
            "counters": counters,
            "file": result["file"],
            "pid": result["pid"],
        }
    if manifest.get("benchmark_count") != len(results):
        check.error(
            f"{label}: benchmark_count {manifest.get('benchmark_count')} != {len(results)} results"
        )
    processes = len(
        {result.get("pid") for result in results.values() if isinstance(result, dict)}
    )
    expected = len(results) if manifest["config"] == "C" else len(target_pids)
    if manifest["config"] != "C" and any(
        len(pids) != 1 for pids in target_pids.values()
    ):
        check.error(f"{label}: non-isolated target measured in multiple processes")
    if processes != expected or manifest.get("measured_process_count") != processes:
        check.error(
            f"{label}: measured processes {processes} (manifest {manifest.get('measured_process_count')}), expected {expected}"
        )
    return observations


def check_manifest(manifest, sample, plan, plan_sha, planned, check, label):
    pins = plan["pins"]
    config = manifest["config"]
    commit = plan["commits"][planned["commit_index"]]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment": plan["experiment"],
        "batch": BATCH,
        "preflight": False,
        "plan_sha256": plan_sha,
        "round": planned["round"],
        "order": planned["order"],
        "config": planned["config"],
        "revision": planned["revision"],
        "commit_index": planned["commit_index"],
        "source_sha": commit["sha"],
        "cell": planned["commit_index"] * len(plan["configs"])
        + plan["configs"].index(planned["config"]),
        "rust_integration": pins["codspeed_rust_revision"]
        if config == "C"
        else "locked-5.0.2",
        "valgrind_sha": pins["valgrind_revisions"]["old" if config == "A" else "new"],
        "valgrind_hashes": pins["valgrind_hashes"]["old" if config == "A" else "new"],
        "runner_version": f"codspeed-runner {pins['runner_version']}",
        "runner_sha256": pins["runner_binary_sha256"],
        "isolation_env": "unset",
        "working_directory": f"{plan['canonical_workload']}/crates/uv-bench",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            shown = (
                "<hash tree>"
                if field == "valgrind_hashes"
                else repr(manifest.get(field))
            )
            check.error(
                f"{label}: {field}={shown}, expected {'<pinned tree>' if field == 'valgrind_hashes' else repr(value)}"
            )
    match = SAMPLE_NAME.fullmatch(sample.name)
    if not match or (int(match[1]), match[2], match[3]) != (
        planned["round"],
        planned["config"],
        planned["revision"],
    ):
        check.error(
            f"{label}: directory name does not match planned round/config/revision"
        )
    environment = manifest["environment"]
    for key, value in {
        "CODSPEED_RUNNER_MODE": "simulation",
        "CODSPEED_PROFILER_ENABLED": "false",
        "CODSPEED_CARGO_WORKSPACE_ROOT": plan["canonical_workload"],
        "CODSPEED_LOG": "debug",
        "VALGRIND_LIB": str(VALGRIND_PREFIX / "libexec" / "valgrind"),
    }.items():
        if environment.get(key) != value:
            check.error(
                f"{label}: environment {key}={environment.get(key)!r}, expected {value!r}"
            )
    if not environment.get("PATH", "").startswith(f"{VALGRIND_PREFIX / 'bin'}:"):
        check.error(f"{label}: PATH must begin with the frozen Valgrind bin directory")
    for key in ("CODSPEED_PROCESS_ISOLATION", "CODSPEED_ISOLATED_BENCHMARK"):
        if key in environment:
            check.error(f"{label}: isolation variable {key} must be unset")
    if [
        Path(command[0]).name for command in manifest["benchmark_commands"]
    ] != manifest["targets"]:
        check.error(f"{label}: benchmark_commands do not match targets")
    if (
        "--skip-setup" not in manifest["command"]
        or "--skip-upload" not in manifest["command"]
    ):
        check.error(f"{label}: command must skip setup and upload")


def check_binaries(manifest, build, check, label):
    integration = "new" if manifest["config"] == "C" else "old"
    binary = build["binaries"][integration]
    if sorted(manifest["targets"]) != sorted(binary["targets"]):
        check.error(
            f"{label}: targets differ from build binaries.{integration}.targets"
        )
    if manifest["binary_hashes"] != binary["sha256"]:
        check.error(f"{label}: binary hashes do not exactly match build metadata")


def load_sample(sample, plan, plan_sha, planned_jobs, check):
    label = str(sample)
    manifest = load_json(sample / "manifest.json")
    job = manifest.get("job")
    if isinstance(job, bool) or not isinstance(job, int) or job not in planned_jobs:
        check.error(f"{label}: job {job!r} is not a planned job")
        return None
    planned = planned_jobs[job]
    missing = [
        field
        for field in REQUIRED_PROVENANCE
        if manifest.get(field) in (None, "", [], {})
    ]
    status = manifest.get("status")
    base = {
        "job": job,
        "round": planned["round"],
        "order": planned["order"],
        "config": planned["config"],
        "revision": planned["revision"],
        "commit_index": planned["commit_index"],
        "sample": label,
        "status": status,
        "started_at": manifest.get("started_at"),
    }
    if status != "success":
        return {**base, "failed": True, "error": manifest.get("error")}
    if missing:
        check.error(f"{label}: missing provenance {missing}")
        return None
    error_count = len(check.errors)
    check_manifest(manifest, sample, plan, plan_sha, planned, check, label)
    capture_dir = sample.parent
    commit = plan["commits"][planned["commit_index"]]
    build = load_build(capture_dir, manifest, plan, plan_sha, commit, check, label)
    packages = capture_dir / "package-versions.txt"
    if (
        not packages.is_file()
        or sha256_bytes(packages.read_bytes()) != manifest["package_versions_sha256"]
    ):
        check.error(f"{label}: package-versions.txt missing or hash mismatch")
    schedule_path = capture_dir / "schedule.json"
    if not schedule_path.is_file() or job not in [
        entry.get("job") for entry in load_json(schedule_path)
    ]:
        check.error(f"{label}: schedule.json missing or does not list job {job}")
    if build is not None:
        check_binaries(manifest, build, check, label)
    results_path = sample / "results.json"
    if not results_path.is_file():
        check.error(f"{label}: missing results.json")
        return None
    observations = check_results(load_json(results_path), manifest, check, label)
    for name in ("profiles.tar.zst", "runner.log"):
        if not (sample / name).is_file():
            check.error(f"{label}: missing {name}")
    if observations is None or len(check.errors) != error_count:
        return None
    cohort, identity = cohort_of(manifest)
    return {
        **base,
        "failed": False,
        "source_sha": manifest["source_sha"],
        "cohort": cohort,
        "cohort_identity": identity,
        "machine_id": manifest["machine_id"],
        "boot_id": manifest["boot_id"],
        "runner_name": manifest.get("runner_name"),
        "workflow_run_id": str(manifest["workflow_run_id"]),
        "build_workflow_run_id": str(manifest["build_workflow_run_id"]),
        "workflow_run_attempt": manifest.get("workflow_run_attempt"),
        "duration_seconds": manifest.get("duration_seconds"),
        "measurement_duration_seconds": manifest.get("measurement_duration_seconds"),
        "build_metadata_sha256": manifest["build_metadata_sha256"],
        "binary_hashes": manifest["binary_hashes"],
        "valgrind_hashes_sha256": sha256_bytes(
            json.dumps(manifest["valgrind_hashes"], sort_keys=True).encode()
        ),
        "valgrind_version": manifest["valgrind_version"],
        "targets": manifest["targets"],
        "command": normalized_command(manifest["command"]),
        "environment": manifest["environment"],
        "build": build,
        "results": observations,
    }


def same_within(samples, key_of, value_of, description, check):
    groups = defaultdict(dict)
    for sample in samples:
        groups[key_of(sample)].setdefault(
            json.dumps(value_of(sample), sort_keys=True), []
        ).append(sample["job"])
    for key, values in sorted(groups.items()):
        if len(values) > 1:
            detail = "; ".join(f"jobs {jobs}" for jobs in values.values())
            check.error(f"{description} differs within {key}: {detail}")


def cross_check(samples, plan, check):
    commit_ids = {commit["index"]: commit["id"] for commit in plan["commits"]}
    same_within(
        samples,
        lambda s: s["revision"],
        lambda s: s["build_metadata_sha256"],
        "build metadata",
        check,
    )
    same_within(
        samples,
        lambda s: "experiment",
        lambda s: s["build_workflow_run_id"],
        "frozen build workflow run",
        check,
    )
    same_within(
        samples,
        lambda s: (s["revision"], "new" if s["config"] == "C" else "old"),
        lambda s: s["binary_hashes"],
        "binary hashes (A/B share old binaries)",
        check,
    )
    same_within(
        samples,
        lambda s: "old" if s["config"] == "A" else "new (B/C)",
        lambda s: s["valgrind_hashes_sha256"],
        "Valgrind tree",
        check,
    )
    same_within(
        samples, lambda s: s["config"], lambda s: s["command"], "command", check
    )
    # PATH embeds the runner's own PATH; every other pinned variable must match.
    same_within(
        samples,
        lambda s: s["config"],
        lambda s: {k: v for k, v in s["environment"].items() if k != "PATH"},
        "environment",
        check,
    )
    same_within(
        samples,
        lambda s: s["config"],
        lambda s: s["valgrind_version"],
        "valgrind_version",
        check,
    )
    same_within(
        samples,
        lambda s: s["revision"],
        lambda s: sorted(s["results"]),
        "benchmark URI inventory",
        check,
    )
    inventories = []
    previous = None
    for index in sorted(commit_ids):
        revision = commit_ids[index]
        commit_samples = [s for s in samples if s["revision"] == revision]
        uris = (
            sorted(set().union(*(s["results"] for s in commit_samples)))
            if commit_samples
            else None
        )
        build = commit_samples[0]["build"] if commit_samples else None
        entry = {
            "revision": revision,
            "commit_index": index,
            "measured": bool(commit_samples),
            "configs": sorted({s["config"] for s in commit_samples}),
            "benchmark_count": len(uris) if uris is not None else None,
            "uris": uris,
            "added": None,
            "removed": None,
            "changed_benchmark_files": None,
            "changed_input_fingerprints": None,
        }
        if previous is not None and previous["uris"] is not None and uris is not None:
            entry["added"] = sorted(set(uris) - set(previous["uris"]))
            entry["removed"] = sorted(set(previous["uris"]) - set(uris))
            if entry["added"] or entry["removed"]:
                check.warn(
                    f"{revision}: inventory changed vs {previous['revision']} (+{len(entry['added'])}, -{len(entry['removed'])})"
                )
        if previous is not None and previous.get("_build") and build:
            entry["changed_benchmark_files"] = mapping_diff(
                previous["_build"]["benchmark_files"], build["benchmark_files"]
            )
            entry["changed_input_fingerprints"] = mapping_diff(
                previous["_build"]["input_fingerprints"], build["input_fingerprints"]
            )
        inventories.append({**entry, "_build": build})
        previous = inventories[-1]
    for entry in inventories:
        entry.pop("_build")
    return inventories


def mapping_diff(before, after):
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "modified": sorted(
            key for key in set(before) & set(after) if before[key] != after[key]
        ),
    }


def load_dataset(plan_path, inputs, rounds=None, allow_partial=False):
    """Parse and validate every capture. Never raises for data problems: they land in errors."""
    check = Validation()
    plan_bytes = plan_path.read_bytes()
    plan_sha = sha256_bytes(plan_bytes)
    plan = load_json(plan_path)
    validate_plan(plan)
    rounds = plan["repetitions"] if rounds is None else rounds
    if not 1 <= rounds <= plan["repetitions"]:
        raise InvalidData(f"--rounds must be in 1..{plan['repetitions']}")
    planned_jobs = {job["job"]: job for job in plan["jobs"]}
    expected = {
        job: entry for job, entry in planned_jobs.items() if entry["round"] < rounds
    }
    seen = defaultdict(list)
    samples, failed = [], []
    for sample in discover_samples(inputs):
        try:
            loaded = load_sample(sample, plan, plan_sha, planned_jobs, check)
        except (InvalidData, KeyError, TypeError, AttributeError) as error:
            check.error(f"{sample}: {type(error).__name__}: {error}")
            continue
        if loaded is None:
            continue
        seen[loaded["job"]].append(loaded["sample"])
        if loaded["failed"]:
            failed.append(loaded)
            message = f"{loaded['sample']}: failed capture (status={loaded['status']!r}): {loaded['error']}"
            (check.warn if allow_partial else check.error)(message)
        else:
            samples.append(loaded)
    for job, paths in sorted(seen.items()):
        if len(paths) > 1:
            check.error(f"job {job} appears {len(paths)} times: {paths}")
        if job not in expected:
            check.error(
                f"job {job} (round {planned_jobs[job]['round']}) is outside the selected {rounds} round(s)"
            )
    successful = {sample["job"] for sample in samples}
    missing = [expected[job] for job in sorted(set(expected) - successful)]
    if missing:
        message = f"{len(missing)} of {len(expected)} planned jobs have no successful measurement"
        (check.warn if allow_partial else check.error)(message)
    inventories = cross_check(samples, plan, check)
    for entry in inventories:
        if set(entry["configs"]) != set(plan["configs"]):
            message = f"{entry['revision']}: measured configs {entry['configs']}, expected ABC"
            (check.warn if allow_partial else check.error)(message)
    cells = Counter((sample["revision"], sample["config"]) for sample in samples)
    coverage = {
        "rounds": rounds,
        "expected_jobs": len(expected),
        "successful_jobs": len(successful & set(expected)),
        "failed_jobs": [
            {
                key: entry[key]
                for key in ("job", "round", "config", "revision", "sample", "error")
            }
            for entry in failed
        ],
        "missing_jobs": [
            {key: entry[key] for key in ("job", "round", "order", "config", "revision")}
            for entry in missing
        ],
        "cells": [
            {
                "revision": commit["id"],
                "config": config,
                "n": cells[(commit["id"], config)],
                "expected": rounds,
            }
            for commit in plan["commits"]
            for config in plan["configs"]
        ],
    }
    cohorts = {}
    machines = {}
    for sample in samples:
        cohort = cohorts.setdefault(
            sample["cohort"],
            {"identity": sample["cohort_identity"], "jobs": [], "machines": set()},
        )
        cohort["jobs"].append(sample["job"])
        cohort["machines"].add(sample["machine_id"])
        machine = machines.setdefault(
            sample["machine_id"],
            {"jobs": [], "boot_ids": set(), "cohorts": set(), "runner_names": set()},
        )
        machine["jobs"].append(sample["job"])
        machine["boot_ids"].add(sample["boot_id"])
        machine["cohorts"].add(sample["cohort"])
        machine["runner_names"].add(sample["runner_name"])
    for machine_id, machine in machines.items():
        if len(machine["cohorts"]) > 1:
            check.warn(
                f"machine {machine_id[:12]} measured under {len(machine['cohorts'])} environment cohorts"
            )
    if not samples and not check.errors:
        check.error("no successful measurements found")
    verdict = (
        "invalid" if check.errors else ("partial" if missing or failed else "valid")
    )
    return {
        "verdict": verdict,
        "experiment": plan["experiment"],
        "plan_sha256": plan_sha,
        "plan": plan,
        "job_count": len(samples),
        "errors": check.errors,
        "warnings": check.warnings,
        "coverage": coverage,
        "cohorts": {
            key: {
                "identity": value["identity"],
                "jobs": sorted(value["jobs"]),
                "machines": sorted(value["machines"]),
            }
            for key, value in sorted(cohorts.items())
        },
        "machines": {
            key: {
                "jobs": sorted(value["jobs"]),
                "boot_ids": sorted(value["boot_ids"]),
                "cohorts": sorted(value["cohorts"]),
                "runner_names": sorted(name for name in value["runner_names"] if name),
            }
            for key, value in sorted(machines.items())
        },
        "inventories": inventories,
        "samples": sorted(samples, key=lambda sample: sample["job"]),
    }


def summary(dataset):
    """Validation JSON: everything except the bulky per-observation payloads and the plan."""
    return {
        key: value for key, value in dataset.items() if key not in ("samples", "plan")
    } | {
        "jobs": [
            {
                key: sample[key]
                for key in (
                    "job",
                    "round",
                    "order",
                    "config",
                    "revision",
                    "started_at",
                    "cohort",
                    "machine_id",
                    "boot_id",
                    "workflow_run_id",
                    "sample",
                )
            }
            | {"benchmark_count": len(sample["results"])}
            for sample in dataset["samples"]
        ]
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        dataset = load_dataset(args.plan, args.input, args.rounds, args.allow_partial)
    except (InvalidData, KeyError, TypeError) as error:
        print(f"invalid plan or arguments: {error}", file=sys.stderr)
        return 1
    args.output.write_text(
        json.dumps(summary(dataset), indent=2, sort_keys=True) + "\n"
    )
    for message in dataset["errors"]:
        print(f"ERROR {message}", file=sys.stderr)
    for message in dataset["warnings"]:
        print(f"WARNING {message}", file=sys.stderr)
    print(
        f"{dataset['verdict']}: {dataset['job_count']} successful jobs, "
        f"{dataset['coverage']['successful_jobs']}/{dataset['coverage']['expected_jobs']} expected"
    )
    if dataset["verdict"] == "valid" or (
        args.allow_partial and dataset["verdict"] == "partial"
    ):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
