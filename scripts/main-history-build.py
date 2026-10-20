#!/usr/bin/env python3
"""Build one immutable per-commit bundle for the uv main-history ABC experiment.

One hosted job per selected commit. The pristine source tree is primed and
snapshotted once with the existing stack-isolation builder (artifact revision
``base``), then the old (lockfile-pinned) and new (codspeed-rust @ pinned
revision) integrations are each built exactly once at the canonical workload
path. Configurations A and B share the old binaries byte for byte; C uses the
new ones. ``main-history-build.json`` records the pins that were actually
observed, the input fingerprints and every shipped artifact hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path
from typing import NoReturn

SCRIPTS = Path(__file__).resolve().parent
REVISION = "base"
INTEGRATIONS = ("old", "new")
CODSPEED_CRATES = ("codspeed", "codspeed-criterion-compat")
BENCH_CRATE = Path("crates/uv-bench")
# Benchmark inputs, by role. Paths are relative to the source root.
INPUT_FILES = {
    "bench_crate_manifest": "crates/uv-bench/Cargo.toml",
    "workspace_manifest": "Cargo.toml",
    "cargo_lock": "Cargo.lock",
    "rust_toolchain": "rust-toolchain.toml",
    "dev_pyproject": "pyproject.toml",
    "dev_uv_lock": "uv.lock",
    # Resolved by the builder to prime `.cache` (uv.rs resolve_warm_* cases).
    "fixture_requirements_jupyter": "test/requirements/jupyter.in",
    "fixture_requirements_airflow": "test/requirements/airflow.in",
    "fixture_uv_config": "test/uv.toml",
}
# Trees read at runtime by benches/uv.rs (`../../.cache`, `../../.venv`).
INPUT_TREES = {
    "bench_definitions_tree": "crates/uv-bench/benches",
    "bench_support_tree": "crates/uv-bench/src",
    "primed_uv_cache_tree": ".cache",
    "venv_tree": ".venv",
}
UNFINGERPRINTED_INPUTS = [
    (
        "git_fetch: remote repositories fetched at runtime into target/bench-git; "
        "pinned commits live in benches/git_fixture (covered by bench_definitions_tree)"
    ),
    "uv_pypi_types / workspace_discovery: synthetic inputs generated in bench source",
    "macos_*: macOS-only benchmarks, not built on Linux",
]


def log(message: str) -> None:
    print(f"[main-history-build] {message}", flush=True)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"main-history-build: {message}")


def output(command, *, cwd=None, env=None) -> str:
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            f"command failed ({completed.returncode}): {shlex.join(map(str, command))}\n{completed.stdout}"
        )
    return completed.stdout.strip()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_digest(directory: Path) -> str:
    """Digest of every file content and symlink target below ``directory``."""
    entries = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            entries[relative] = f"symlink:{os.readlink(path)}"
        elif path.is_file():
            entries[relative] = digest(path)
    if not entries:
        fail(f"empty input tree {directory}")
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def load_plan(path: Path, commit_id: str) -> tuple[dict, dict]:
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1:
        fail(f"unsupported plan schema {plan.get('schema_version')}")
    matches = [commit for commit in plan["commits"] if commit["id"] == commit_id]
    if len(matches) != 1:
        fail(f"commit {commit_id} not in plan exactly once")
    return plan, matches[0]


def load_builder(source_sha: str, pins: dict):
    """Load the existing builder with this commit pinned as its ``base`` revision."""
    os.environ["UV_BASE_SHA"] = source_sha
    os.environ["CODSPEED_RUST_REV"] = pins["codspeed_rust_revision"]
    os.environ["VALGRIND_OLD_SHA"] = pins["valgrind_revisions"]["old"]
    os.environ["VALGRIND_NEW_SHA"] = pins["valgrind_revisions"]["new"]
    spec = importlib.util.spec_from_file_location(
        "stack_isolation_build", SCRIPTS / "stack-isolation-build.py"
    )
    if spec is None or spec.loader is None:
        fail("cannot load stack-isolation-build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.UV_REVISIONS[REVISION] != source_sha:
        fail(f"builder base revision {module.UV_REVISIONS[REVISION]} != {source_sha}")
    if module.CODSPEED_RUST_REV != pins["codspeed_rust_revision"]:
        fail(
            f"builder codspeed-rust revision {module.CODSPEED_RUST_REV} differs from plan"
        )
    return module


def lock_packages(lockfile: Path) -> dict[str, list[dict]]:
    packages: dict[str, list[dict]] = {}
    for package in tomllib.loads(lockfile.read_text()).get("package", []):
        packages.setdefault(package["name"], []).append(package)
    return packages


def check_old_lock(lockfile: Path, version: str) -> dict:
    packages = lock_packages(lockfile)
    found = {}
    for name in CODSPEED_CRATES:
        entries = packages.get(name, [])
        versions = sorted({entry["version"] for entry in entries})
        if versions != [version]:
            fail(
                f"{lockfile}: {name} versions {versions}, expected exactly [{version}]"
            )
        found[name] = version
    return found


def check_new_lock(lockfile: Path, revision: str) -> dict:
    packages = lock_packages(lockfile)
    found = {}
    for name in CODSPEED_CRATES:
        entries = packages.get(name, [])
        if len(entries) != 1 or revision not in entries[0].get("source", ""):
            fail(
                f"{lockfile}: {name} does not resolve to codspeed-rust {revision}: {entries}"
            )
        found[name] = {"version": entries[0]["version"], "source": entries[0]["source"]}
    return found


def fingerprint_inputs(workload: Path) -> tuple[dict, dict]:
    fingerprints = {}
    for key, relative in INPUT_FILES.items():
        path = workload / relative
        if not path.is_file():
            fail(f"missing benchmark input {relative}")
        fingerprints[key] = digest(path)
    for key, relative in INPUT_TREES.items():
        path = workload / relative
        if not path.is_dir():
            fail(f"missing benchmark input tree {relative}")
        fingerprints[key] = tree_digest(path)
    benchmark_files = {
        path.relative_to(workload).as_posix(): digest(path)
        for path in sorted((workload / BENCH_CRATE).rglob("*.rs"))
        if path.is_file()
    }
    if not benchmark_files:
        fail("no benchmark Rust files found")
    return fingerprints, benchmark_files


def extract(archive: Path, destination: Path, *members: str) -> None:
    command = f"zstd -dc {shlex.quote(str(archive))} | tar -xf - -C {shlex.quote(str(destination))}"
    if members:
        command += " " + " ".join(shlex.quote(member) for member in members)
    output(["bash", "-euo", "pipefail", "-c", command])


def verify_bundle(artifacts: Path, source_sha: str, workload: Path) -> dict:
    """Re-read the shipped archives and return the per-integration binary inventory."""
    scratch = Path(tempfile.mkdtemp(prefix="main-history-verify-"))
    try:
        extract(
            artifacts / "sources" / f"{REVISION}.tar.zst",
            scratch,
            f"sources/{REVISION}/experiment-source.json",
        )
        source = json.loads(
            (scratch / "sources" / REVISION / "experiment-source.json").read_text()
        )
        if source["source_sha"] != source_sha or source["workload"] != str(workload):
            fail(f"source snapshot provenance mismatch: {source}")
        binaries = {}
        for integration in INTEGRATIONS:
            key = f"{REVISION}-{integration}"
            extract(artifacts / "binaries" / f"{key}.tar.zst", scratch)
            build = json.loads((scratch / "binaries" / key / "build.json").read_text())
            directory = scratch / "binaries" / key / "uv-bench"
            actual = {
                path.name: digest(path)
                for path in sorted(directory.iterdir())
                if path.is_file() and not path.is_symlink()
            }
            if build["source_sha"] != source_sha or build["workload"] != str(workload):
                fail(f"{key}/build.json provenance mismatch")
            if (
                actual != build["sha256"]
                or sorted(actual) != sorted(build["targets"])
                or not actual
            ):
                fail(f"{key}: shipped binaries differ from build.json")
            for name in actual:
                if not os.access(directory / name, os.X_OK):
                    fail(f"{key}/{name} is not executable")
            binaries[integration] = {"targets": sorted(actual), "sha256": actual}
        if binaries["old"]["targets"] != binaries["new"]["targets"]:
            fail(
                f"old/new target sets differ: {binaries['old']['targets']} != {binaries['new']['targets']}"
            )
        if binaries["old"]["sha256"] == binaries["new"]["sha256"]:
            fail("old and new integration binaries are identical")
        return binaries
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build(args) -> None:
    plan_path = args.plan.resolve()
    plan, commit = load_plan(plan_path, args.commit)
    pins = plan["pins"]
    if plan["artifact_revision"] != REVISION:
        fail(f"plan artifact_revision {plan['artifact_revision']} != {REVISION}")
    artifacts = args.artifacts.resolve()
    workload = args.workload.resolve()
    if str(workload) != plan["canonical_workload"]:
        fail(f"workload {workload} is not the canonical {plan['canonical_workload']}")
    if (artifacts / "main-history-build.json").exists():
        fail(f"{artifacts} already holds a bundle")
    artifacts.mkdir(parents=True, exist_ok=True)
    source_sha = commit["sha"]
    builder = load_builder(source_sha, pins)
    started = time.time()

    uv_version = output(["uv", "--version"])
    if uv_version.split()[:2] != ["uv", pins["uv_version"]]:
        fail(f"uv {uv_version!r} is not the pinned {pins['uv_version']}")

    builder.cmd_sources(
        argparse.Namespace(
            artifacts=artifacts,
            workload=workload,
            git_cache=args.git_cache,
            revision=REVISION,
            prime_cache=True,
            snapshot=True,
        )
    )
    head = output(["git", "-C", workload, "rev-parse", "HEAD"])
    if head != source_sha:
        fail(f"workload HEAD {head} != {source_sha}")

    # Pristine inputs, before any integration build touches Cargo.toml/Cargo.lock.
    input_fingerprints, benchmark_files = fingerprint_inputs(workload)
    pristine_lock = (workload / "Cargo.lock").read_bytes()
    old_crates = check_old_lock(
        workload / "Cargo.lock", pins["old_codspeed_crate_version"]
    )

    env = dict(os.environ, UV_LOCKED="1")
    rustc_version = output(["rustc", "--version"], cwd=workload, env=env)
    if rustc_version.split()[:2] != ["rustc", pins["rust_toolchain"]]:
        fail(f"{rustc_version!r} is not the pinned toolchain {pins['rust_toolchain']}")
    cargo_version = output(["cargo", "--version"], cwd=workload, env=env)

    old_cargo_codspeed = output(
        [
            "uv",
            "run",
            "--only-dev",
            "python",
            "-c",
            "from importlib.metadata import version; print(version('astral-dev-toolchain-cargo-codspeed'))",
        ],
        cwd=workload,
        env=env,
    )
    if old_cargo_codspeed != pins["old_cargo_codspeed_version"]:
        fail(
            f"old cargo-codspeed {old_cargo_codspeed} != {pins['old_cargo_codspeed_version']}"
        )

    durations = {}
    for integration in INTEGRATIONS:
        began = time.monotonic()
        builder.cmd_binaries(
            argparse.Namespace(
                artifacts=artifacts,
                workload=workload,
                git_cache=args.git_cache,
                revision=REVISION,
                integration=integration,
                codspeed_source=args.codspeed_source,
                tools_root=args.tools_root,
            )
        )
        durations[integration] = time.monotonic() - began

    old_lock = artifacts / "locks" / f"{REVISION}-old-Cargo.lock"
    if old_lock.read_bytes() != pristine_lock:
        fail("old integration build changed the source Cargo.lock")
    new_crates = check_new_lock(
        artifacts / "locks" / f"{REVISION}-new-Cargo.lock",
        pins["codspeed_rust_revision"],
    )
    codspeed_head = output(
        ["git", "-C", args.codspeed_source.resolve(), "rev-parse", "HEAD"]
    )
    if codspeed_head != pins["codspeed_rust_revision"]:
        fail(f"codspeed-rust checkout {codspeed_head} != pinned revision")
    new_cargo_codspeed = output(
        [
            "cargo",
            "install",
            "--list",
            "--root",
            args.tools_root.resolve() / "cargo-codspeed-new",
        ]
    )

    binaries = verify_bundle(artifacts, source_sha, workload)
    shipped = sorted(
        path
        for pattern in (
            "sources/*.tar.zst",
            "binaries/*.tar.zst",
            "locks/*",
            "patches/*",
            "cargo-tree-*.txt",
        )
        for path in artifacts.glob(pattern)
        if path.is_file()
    )
    metadata = {
        "schema_version": 1,
        "experiment": plan["experiment"],
        "plan_sha256": digest(plan_path),
        "revision": commit["id"],
        "commit_index": commit["index"],
        "source_sha": source_sha,
        "canonical_workload": str(workload),
        "rust_toolchain": pins["rust_toolchain"],
        "rustc_version": rustc_version,
        "old_cargo_codspeed_version": old_cargo_codspeed,
        "old_codspeed_crate_version": pins["old_codspeed_crate_version"],
        "codspeed_rust_revision": pins["codspeed_rust_revision"],
        "input_fingerprints": input_fingerprints,
        "benchmark_files": benchmark_files,
        "binaries": binaries,
        # Extra provenance.
        "artifact_revision": REVISION,
        "commit_title": commit["title"],
        "committed_at": commit["committed_at"],
        "cargo_version": cargo_version,
        "uv_version": uv_version,
        "old_codspeed_crates": old_crates,
        "new_codspeed_crates": new_crates,
        "new_cargo_codspeed": new_cargo_codspeed,
        "artifact_files": {
            path.relative_to(artifacts).as_posix(): digest(path) for path in shipped
        },
        "unfingerprinted_inputs": UNFINGERPRINTED_INPUTS,
        "build_durations_seconds": durations,
        "started_at": started,
        "finished_at": time.time(),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "workflow_sha": os.environ.get("GITHUB_SHA"),
        "workflow_repository": os.environ.get("GITHUB_REPOSITORY"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "image_os": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
    }
    (artifacts / "main-history-build.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    log(
        f"{commit['id']} {source_sha}: {len(binaries['old']['targets'])} targets -> {artifacts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="plan commit id, e.g. c03")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--git-cache", type=Path, required=True)
    parser.add_argument("--codspeed-source", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
