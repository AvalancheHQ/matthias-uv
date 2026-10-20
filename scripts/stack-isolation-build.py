#!/usr/bin/env python3
"""Build the frozen artifacts for the uv stack-isolation experiment.

The experiment crosses two uv revisions (base/head) with CodSpeed Rust
integrations (old: the lockfile-pinned 5.0.1, new: codspeed-rust @304fa86,
warm: codspeed-rust @061f923 = new + warm-cache warmup) and runs every
measurement under a pinned Valgrind build (old/new refs).

This branch (config D) builds only the ``warm`` integration and the ``new``
Valgrind; the old/new binaries and the old Valgrind of configs A/B/C were
built and measured by the ``ci/uv22754-stack-isolation-20261020`` workflow.

Everything is built exactly once, in a canonical ``$GITHUB_WORKSPACE/workload``
directory, so the ``CARGO_MANIFEST_DIR`` paths compiled into the benchmark
binaries match the runtime layout used by ``stack-isolation-capture.py``.

Artifact layout (``--artifacts``); every directory is shipped as a
``*.tar.zst`` archive so that GitHub's artifact upload cannot strip file modes
or symlinks, and :func:`cmd_prepare` restores the trees before capture runs::

    sources/{base,head}/                       # pristine source tree + .cache
    sources/{base,head}/experiment-source.json # {source_sha: ...}
    binaries/{base,head}-{warm}/uv-bench/      # every benchmark executable
    binaries/{base,head}-{warm}/build.json     # {source_sha, rust_sha, sha256}
    valgrind/{new}/                            # contents of the install prefix
    valgrind/{new}.json                        # ref + sha256 of every file
    locks/{base,head}-{warm}-Cargo.lock        # retained resolved lockfiles
    cargo-tree-codspeed-{base,head}-{warm}.txt
    patches/{base,head}-codspeed.patch         # temporary [patch.crates-io] block
    build-info.json                            # merged summary (verify)
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
import sys
import tempfile
import time
from pathlib import Path


def pinned(name: str, default: str) -> str:
    """Pinned revision, overridable by the workflow env for a single source of truth."""
    return os.environ.get(name, default)


UV_REVISIONS = {
    "base": pinned("UV_BASE_SHA", "21979a0931e931b122c394aeb0bd731e785de30c"),
    "head": pinned("UV_HEAD_SHA", "c2d613f98dac62b4676c2d9b3a5eb3f9b3865ecc"),
}
VALGRIND_REVISIONS = {
    "old": pinned("VALGRIND_OLD_SHA", "e1e4bdf3fa3daa61cce684b513f874ff0735b1b8"),
    "new": pinned("VALGRIND_NEW_SHA", "5275f9f34b780c8c9320906a14ef6e160d45bf94"),
}
# Git integrations: the revision of codspeed-rust patched into the workload.
CODSPEED_RUST_REVS = {
    "new": pinned("CODSPEED_RUST_REV", "304fa86cb8ecfd95cf6d0c1e4f05aae0f17da96a"),
    "warm": pinned("CODSPEED_RUST_WARM_REV", "061f9230eac19fe27c95b8bccef186732558f35e"),
}
CODSPEED_RUST_URL = "https://github.com/CodSpeedHQ/codspeed-rust"
UV_URL = "https://github.com/astral-sh/uv"
VALGRIND_URL = "https://github.com/CodSpeedHQ/valgrind-codspeed"

# Compiled into both Valgrind builds; capture replaces this exact path per suite.
VALGRIND_PREFIX = pinned("STACK_ISOLATION_VALGRIND_PREFIX", "/opt/stack-isolation/valgrind")
# Where cargo-codspeed puts simulation (`analysis`) executables, relative to the
# canonical workload directory.
CODSPEED_TARGET_SUBDIR = Path("target/codspeed/analysis/uv-bench")
WORKLOAD_MANIFEST_DIR = Path("crates/uv-bench")
# ``.venv`` is deliberately kept: `benches/uv.rs` resolves
# `PythonEnvironment::from_root("../../.venv", ...)` next to `../../.cache`.
SOURCE_EXCLUDES = {"target"}
# Built and verified by this workflow; every other integration/Valgrind pin is
# retained only so the capture script's provenance cross-check stays complete.
INTEGRATIONS = ("warm",)
VALGRIND_BUILDS = ("new",)
REVISIONS = tuple(UV_REVISIONS)


def log(message: str) -> None:
    print(f"[stack-isolation] {message}", flush=True)


def run(
    command,
    *,
    cwd=None,
    env=None,
    capture: bool = False,
    check: bool = True,
):
    """Run a command, echoing it. Long build output streams to the job log."""
    command = [str(part) for part in command]
    log(f"$ {shlex.join(command)}" + (f"  (cwd={cwd})" if cwd else ""))
    if capture:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if check and completed.returncode != 0:
            raise SystemExit(
                f"command failed ({completed.returncode}): {shlex.join(command)}\n{completed.stdout}"
            )
        return completed.stdout
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if check and completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {shlex.join(command)}")
    return ""


def run_shell(script: str, *, cwd=None, env=None) -> None:
    run(["bash", "-euo", "pipefail", "-c", script], cwd=cwd, env=env)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_tree(directory: Path) -> dict[str, str]:
    """Hashes every regular, non-symlink file (matches capture's hashes())."""
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def git_checkout(url: str, revision: str, destination: Path) -> None:
    """Fetch a single pinned revision into ``destination`` (cache-friendly)."""
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        run(["git", "init", "--quiet", str(destination)])
        run(["git", "-C", str(destination), "remote", "add", "origin", url])
    run(["git", "-C", str(destination), "fetch", "--depth", "1", "--no-tags", "origin", revision])
    run(["git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    actual = run(["git", "-C", str(destination), "rev-parse", "HEAD"], capture=True).strip()
    if actual != revision:
        raise SystemExit(f"checkout mismatch: expected {revision}, got {actual} in {destination}")


def materialize_workload(url: str, revision: str, workload: Path) -> None:
    """Check out the exact revision at the canonical build and runtime path."""
    if workload.exists():
        shutil.rmtree(workload)
    git_checkout(url, revision, workload)
    if not (workload / "Cargo.toml").is_file():
        raise SystemExit(f"workload missing Cargo.toml: {workload}")


def make_archive(artifacts: Path, relative: Path) -> Path:
    """Archive ``artifacts/relative`` as ``artifacts/<relative>.tar.zst``."""
    source = artifacts / relative
    archive = source.with_suffix(".tar.zst")
    archive.parent.mkdir(parents=True, exist_ok=True)
    run_shell(
        f"tar -cf - -C {shlex.quote(str(artifacts))} {shlex.quote(relative.as_posix())} "
        f"| zstd -T0 -3 -q -f -o {shlex.quote(str(archive))}"
    )
    return archive


def prime_uv_cache(workload: Path) -> dict:
    """Mirror bench.yml: build the revision's uv and resolve the fixture sets.

    The interpreter is pinned to the image's ``/usr/bin/python3`` so `.venv` is
    reproducible across the build and measurement images (which share the same
    runner ImageVersion) instead of referencing a uv-managed download.
    """
    env = dict(os.environ, UV_LOCKED="1")
    interpreter = Path("/usr/bin/python3")
    if not interpreter.exists():
        raise SystemExit(f"expected a system interpreter at {interpreter}")
    run(
        [
            "cargo",
            "run",
            "--locked",
            "--bin",
            "uv",
            "--",
            "venv",
            "--python",
            str(interpreter),
            "--cache-dir",
            ".cache",
        ],
        cwd=workload,
        env=env,
    )
    uv = workload / "target" / "debug" / "uv"
    if not uv.is_file():
        raise SystemExit(f"primed uv binary missing: {uv}")
    for requirement in ("jupyter.in", "airflow.in"):
        run(
            [
                uv,
                "--config-file",
                "test/uv.toml",
                "pip",
                "compile",
                f"test/requirements/{requirement}",
                "--universal",
                "--exclude-newer",
                "2024-08-08",
                "--cache-dir",
                ".cache",
            ],
            cwd=workload,
            env=env,
            capture=True,  # the compiled requirements are noise; failures still abort
        )
    # Keep .venv: it is part of the runtime fixture the benchmarks load.
    return {"path": str(interpreter), "version": run([interpreter, "--version"], capture=True).strip()}


def snapshot_sources(
    artifacts: Path, workload: Path, revision: str, source_sha: str, interpreter: dict | None = None
) -> None:
    destination = artifacts / "sources" / revision
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        workload,
        destination,
        symlinks=True,
        ignore=lambda directory, names: SOURCE_EXCLUDES.intersection(names) if Path(directory) == workload else set(),
    )
    write_json(
        destination / "experiment-source.json",
        {
            "schema_version": 1,
            "revision": revision,
            "source_sha": source_sha,
            "workload": str(workload),
            "compiled_manifest_dir": str(workload / WORKLOAD_MANIFEST_DIR),
            "interpreter": interpreter,
            "venv_python": str((workload / ".venv" / "bin" / "python").resolve()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    archive = make_archive(artifacts, Path("sources") / revision)
    log(f"sources snapshot {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    # Only the archive is uploaded; prepare-artifacts restores the tree.
    shutil.rmtree(destination)


def cmd_sources(args) -> None:
    artifacts = args.artifacts.resolve()
    workload = args.workload.resolve()
    source_sha = UV_REVISIONS[args.revision]
    materialize_workload(UV_URL, source_sha, workload)
    log(f"workload {workload} at {args.revision} {source_sha}")
    interpreter = prime_uv_cache(workload) if args.prime_cache else None
    if args.snapshot:
        snapshot_sources(artifacts, workload, args.revision, source_sha, interpreter)


def copy_targets(artifacts: Path, workload: Path, revision: str, integration: str, metadata: dict) -> None:
    source = workload / CODSPEED_TARGET_SUBDIR
    if not source.is_dir():
        raise SystemExit(f"codspeed build produced no target directory: {source}")
    targets = sorted(
        path for path in source.iterdir() if path.is_file() and not path.is_symlink()
    )
    if not targets:
        raise SystemExit(f"no benchmark executables in {source}")
    names = [path.name for path in targets]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate benchmark target names: {names}")

    destination = artifacts / "binaries" / f"{revision}-{integration}"
    if destination.exists():
        shutil.rmtree(destination)
    binaries = destination / "uv-bench"
    binaries.mkdir(parents=True)
    for path in targets:
        target = binaries / path.name
        shutil.copy2(path, target)
        target.chmod(target.stat().st_mode | 0o755)

    hashes = hash_tree(binaries)
    write_json(
        destination / "build.json",
        {
            "schema_version": 1,
            "revision": revision,
            "source_sha": UV_REVISIONS[revision],
            "integration": integration,
            "rust_sha": metadata["rust_sha"],
            "rust_integration": metadata["rust_integration"],
            "cargo_codspeed_version": metadata["cargo_codspeed_version"],
            "workload": str(workload),
            "compiled_manifest_dir": str(workload / WORKLOAD_MANIFEST_DIR),
            "targets": names,
            "sha256": hashes,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "extra": metadata.get("extra", {}),
        },
    )
    archive = make_archive(artifacts, Path("binaries") / f"{revision}-{integration}")
    size = archive.stat().st_size / 1e6
    log(f"{revision}-{integration}: {len(names)} targets {names} -> {archive} ({size:.1f} MB)")
    shutil.rmtree(destination)


def cargo_tree_inverse(workload: Path, env: dict, crate: str) -> str:
    """`cargo tree -i <crate>` stdout only.

    stderr is kept apart on purpose: cargo's "patch ... was not used in the
    crate graph" warning quotes the patched revision, so a merged stream would
    make a substring check pass exactly when the patch was ignored.
    """
    command = ["cargo", "tree", "-i", crate]
    log(f"$ {shlex.join(command)}  (cwd={workload})")
    completed = subprocess.run(
        command, cwd=workload, env=env, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {shlex.join(command)}\n{completed.stderr}")
    if "was not used in the crate graph" in completed.stderr:
        raise SystemExit(f"[patch.crates-io] ignored by cargo:\n{completed.stderr}")
    return completed.stdout


def expect_patched(tree: str, crate: str, revision: str) -> None:
    """The inverse tree must root at the git-sourced crate, not a registry one."""
    root = tree.splitlines()[0] if tree else ""
    expected = f"{crate} v"
    source = f"({CODSPEED_RUST_URL}?rev={revision}#"
    if not (root.startswith(expected) and source in root):
        raise SystemExit(f"{crate} is not built from {CODSPEED_RUST_URL}@{revision}:\n{tree}")



def cmd_binaries(args) -> None:
    artifacts = args.artifacts.resolve()
    workload = args.workload.resolve()
    revision = args.revision
    integration = args.integration
    if not (workload / "Cargo.toml").is_file():
        raise SystemExit(f"no workload sources at {workload}; run `sources` first")
    env = dict(os.environ, UV_LOCKED="1")
    metadata = {"rust_integration": integration}

    if integration == "old":
        # Exactly the locked upstream build: uv's dev environment provides the
        # 5.0.1 cargo-codspeed, and Cargo.lock keeps criterion compat at 5.0.1.
        run(["cargo", "metadata", "--locked", "--format-version", "1"], cwd=workload, env=env, capture=True)
        run(
            ["uv", "run", "--only-dev", "cargo", "codspeed", "build", "--locked", "--profile", "profiling", "-p", "uv-bench"],
            cwd=workload,
            env=env,
        )
        metadata["cargo_codspeed_version"] = run(
            ["uv", "run", "--only-dev", "python", "-c",
             "from importlib.metadata import version; print(version('astral-dev-toolchain-cargo-codspeed'))"],
            cwd=workload, env=env, capture=True,
        ).strip()
        metadata["rust_sha"] = "5.0.1"
    else:
        codspeed_rust_rev = CODSPEED_RUST_REVS[integration]
        codspeed_src = args.codspeed_source.resolve()
        git_checkout(CODSPEED_RUST_URL, codspeed_rust_rev, codspeed_src)
        run(["git", "-C", codspeed_src, "submodule", "update", "--init", "--recursive"])
        # Kept outside the artifact root: only binaries/valgrind/sources ship.
        tools_root = args.tools_root.resolve() / f"cargo-codspeed-{integration}"
        run(
            [
                "cargo",
                "install",
                "--locked",
                "--path",
                codspeed_src / "crates" / "cargo-codspeed",
                "--root",
                tools_root,
            ],
            cwd=workload,
            env=env,
        )
        cargo_codspeed = tools_root / "bin" / "cargo-codspeed"
        metadata["cargo_codspeed_version"] = run(
            ["cargo", "install", "--list", "--root", tools_root], capture=True
        ).strip()

        manifest = workload / "Cargo.toml"
        original = manifest.read_text()
        patch = (
            "\n[patch.crates-io]\n"
            f'codspeed = {{ git = "{CODSPEED_RUST_URL}", rev = "{codspeed_rust_rev}" }}\n'
            f'codspeed-criterion-compat = {{ git = "{CODSPEED_RUST_URL}", rev = "{codspeed_rust_rev}" }}\n'
        )
        patch_file = artifacts / "patches" / f"{revision}-codspeed.patch"
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file.write_text(patch)
        try:
            manifest.write_text(original.rstrip() + "\n" + patch)
            env_new = dict(env, PATH=f"{tools_root / 'bin'}:{env['PATH']}")
            # A [patch.crates-io] entry is only used when the lockfile selects
            # its version. The locked registry 5.0.2 satisfies uv's `5.0.1`
            # requirement, so a patch at any other version is silently ignored
            # ("patch ... was not used in the crate graph"). Re-lock exactly
            # the patched crates so the lockfile moves them to the git source;
            # the resulting lockfile is retained as an artifact.
            run(
                ["cargo", "update", "-p", "codspeed-criterion-compat", "-p", "codspeed"],
                cwd=workload,
                env=env_new,
            )
            run(
                [cargo_codspeed, "build", "--locked", "--profile", "profiling", "-p", "uv-bench"],
                cwd=workload,
                env=env_new,
            )
            for crate in ("codspeed-criterion-compat", "codspeed"):
                tree = cargo_tree_inverse(workload, env_new, crate)
                expect_patched(tree, crate, codspeed_rust_rev)
                suffix = "" if crate == "codspeed-criterion-compat" else "-core"
                (artifacts / f"cargo-tree-codspeed{suffix}-{revision}-{integration}.txt").write_text(tree)
            metadata["extra"] = {"codspeed_rust_path": str(codspeed_src), "patch_crates_io": True}
            lockfile = (artifacts / f"locks/{revision}-{integration}-Cargo.lock")
            lockfile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workload / "Cargo.lock", lockfile)
        finally:
            manifest.write_text(original)
        metadata["rust_sha"] = codspeed_rust_rev

    if integration == "old":
        tree = run(
            ["cargo", "tree", "-i", "codspeed-criterion-compat", "--locked"], cwd=workload, env=env, capture=True
        )
        (artifacts / f"cargo-tree-codspeed-{revision}-{integration}.txt").write_text(tree)
        lockfile = artifacts / f"locks/{revision}-{integration}-Cargo.lock"
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workload / "Cargo.lock", lockfile)

    copy_targets(artifacts, workload, revision, integration, metadata)


def cmd_valgrind(args) -> None:
    artifacts = args.artifacts.resolve()
    name = args.name
    revision = VALGRIND_REVISIONS[name]
    checkout = args.git_cache.resolve() / "valgrind-codspeed"
    git_checkout(VALGRIND_URL, revision, checkout)
    run(
        [
            "git",
            "-C",
            str(checkout),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--depth",
            "1",
            "third_party/capstone",
        ]
    )
    run(["./autogen.sh"], cwd=checkout)
    run(["./configure", f"--prefix={VALGRIND_PREFIX}", "--enable-only64bit"], cwd=checkout)
    run(["make", "-j", str(os.cpu_count() or 1)], cwd=checkout)

    staging = Path(tempfile.mkdtemp(prefix=f"valgrind-install-{name}-"))
    try:
        run(["make", "install", f"DESTDIR={staging}"], cwd=checkout)
        installed = staging / VALGRIND_PREFIX.lstrip("/")
        if not (installed / "bin" / "valgrind").is_file():
            raise SystemExit(f"valgrind install missing launcher: {installed}")

        destination = artifacts / "valgrind" / name
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(installed, destination, symlinks=True)

        hashes = hash_tree(destination)
        env = dict(os.environ, VALGRIND_LIB=str(destination / "libexec" / "valgrind"))
        version = run([destination / "bin" / "valgrind", "--version"], env=env, capture=True).strip()
        callgrind = sorted(destination.rglob("callgrind-amd64-linux"))
        if len(callgrind) != 1:
            raise SystemExit(f"expected exactly one callgrind-amd64-linux, found {callgrind}")
        write_json(
            artifacts / "valgrind" / f"{name}.json",
            {
                "schema_version": 1,
                "name": name,
                "source_sha": revision,
                "prefix": VALGRIND_PREFIX,
                "version": version,
                "callgrind": callgrind[0].relative_to(destination).as_posix(),
                "sha256": hashes,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        archive = make_archive(artifacts, Path("valgrind") / name)
        log(f"valgrind {name} {version} -> {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
        shutil.rmtree(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def cmd_prepare(args) -> None:
    """Extract every archive in place, restoring modes and symlinks."""
    artifacts = args.artifacts.resolve()
    # Only the archives this script produced: source trees may legitimately
    # contain their own `.tar.zst` fixtures, which must not be extracted.
    archives = sorted(
        path
        for pattern in ("sources/*.tar.zst", "binaries/*.tar.zst", "valgrind/*.tar.zst")
        for path in artifacts.glob(pattern)
    )
    if not archives:
        raise SystemExit(f"no stack-isolation archives under {artifacts}")
    for archive in archives:
        log(f"extract {archive}")
        run_shell(f"zstd -dc {shlex.quote(str(archive))} | tar -xf - -C {shlex.quote(str(artifacts))}")


def capture_constants() -> dict | None:
    """Load the capture script's pins so the two files cannot drift apart."""
    path = Path(__file__).resolve().with_name("stack-isolation-capture.py")
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("stack_isolation_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "uv_revisions": dict(module.REVISIONS),
        "valgrind_revisions": dict(module.VALGRIND_REVISIONS),
        "codspeed_rust_revs": dict(module.RUST_REVISIONS),
        "valgrind_prefix": str(module.VALGRIND_PREFIX),
    }


def cmd_verify(args) -> None:
    artifacts = args.artifacts.resolve()
    problems: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    shared = capture_constants()
    if shared is None:
        log("capture script not present next to the build script; skipping pin cross-check")
    else:
        expect(shared["uv_revisions"] == UV_REVISIONS, f"capture UV revisions differ: {shared['uv_revisions']}")
        expect(
            shared["valgrind_revisions"] == VALGRIND_REVISIONS,
            f"capture Valgrind revisions differ: {shared['valgrind_revisions']}",
        )
        expect(
            shared["codspeed_rust_revs"] == CODSPEED_RUST_REVS,
            f"capture codspeed-rust revs differ: {shared['codspeed_rust_revs']}",
        )
        expect(
            shared["valgrind_prefix"] == VALGRIND_PREFIX,
            f"capture Valgrind prefix differs: {shared['valgrind_prefix']}",
        )

    expected_targets: dict[str, list[str]] = {}
    for revision, source_sha in UV_REVISIONS.items():
        source = artifacts / "sources" / revision
        marker = source / "experiment-source.json"
        expect(source.is_dir(), f"missing source tree {source}")
        if marker.is_file():
            recorded = json.loads(marker.read_text()).get("source_sha")
            expect(recorded == source_sha, f"{marker}: source_sha {recorded} != {source_sha}")
            expect(
                (source / WORKLOAD_MANIFEST_DIR / "Cargo.toml").is_file(),
                f"{source}: missing {WORKLOAD_MANIFEST_DIR}/Cargo.toml",
            )
        else:
            problems.append(f"missing {marker}")

    for revision in REVISIONS:
        for integration in INTEGRATIONS:
            key = f"{revision}-{integration}"
            binaries = artifacts / "binaries" / key / "uv-bench"
            build_json = artifacts / "binaries" / key / "build.json"
            expect(binaries.is_dir(), f"missing {binaries}")
            if not build_json.is_file():
                problems.append(f"missing {build_json}")
                continue
            recorded = json.loads(build_json.read_text())
            expect(recorded.get("source_sha") == UV_REVISIONS[revision], f"{build_json}: wrong source_sha")
            expect(bool(recorded.get("rust_sha")), f"{build_json}: missing rust_sha")
            actual = hash_tree(binaries) if binaries.is_dir() else {}
            expect(actual == recorded.get("sha256"), f"{build_json}: binary hashes changed")
            names = sorted(actual)
            expect(names == sorted(recorded.get("targets", [])), f"{build_json}: target names changed")
            expect(bool(names), f"{binaries}: no executables")
            for name in names:
                expect(os.access(binaries / name, os.X_OK), f"{binaries / name}: not executable")
            expected_targets[key] = names

    if expected_targets:
        reference = next(iter(expected_targets.values()))
        for key, names in expected_targets.items():
            expect(names == reference, f"{key}: target list differs: {names} != {reference}")

    for name in VALGRIND_BUILDS:
        revision = VALGRIND_REVISIONS[name]
        directory = artifacts / "valgrind" / name
        metadata_file = artifacts / "valgrind" / f"{name}.json"
        expect(directory.is_dir(), f"missing {directory}")
        if not metadata_file.is_file():
            problems.append(f"missing {metadata_file}")
            continue
        recorded = json.loads(metadata_file.read_text())
        expect(recorded.get("source_sha") == revision, f"{metadata_file}: wrong source_sha")
        expect(hash_tree(directory) == recorded.get("sha256"), f"{metadata_file}: valgrind files changed")
        callgrind = sorted(directory.rglob("callgrind-amd64-linux"))
        expect(len(callgrind) == 1, f"{directory}: expected one callgrind-amd64-linux, found {callgrind}")

    write_json(
        artifacts / "build-info.json",
        {
            "schema_version": 1,
            "uv_revisions": UV_REVISIONS,
            "valgrind_revisions": VALGRIND_REVISIONS,
            "codspeed_rust_revs": CODSPEED_RUST_REVS,
            "valgrind_prefix": VALGRIND_PREFIX,
            "targets": expected_targets,
            "verified": not problems,
        },
    )
    if problems:
        for problem in problems:
            print(f"::error::{problem}", file=sys.stderr)
        raise SystemExit(f"artifact verification failed with {len(problems)} problem(s)")
    log(f"verified artifacts: {sorted(expected_targets)} with targets {reference}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--artifacts", type=Path, required=True, help="artifact output root")
        sub.add_argument(
            "--workload",
            type=Path,
            default=Path.cwd() / "workload",
            help="canonical build/runtime directory (default: ./workload)",
        )
        sub.add_argument(
            "--git-cache",
            type=Path,
            default=Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "stack-isolation-git",
            help="scratch directory for pinned checkouts",
        )

    sources = subparsers.add_parser("sources", help="materialize a pinned revision into the workload")
    common(sources)
    sources.add_argument("--revision", choices=REVISIONS, required=True)
    sources.add_argument("--prime-cache", action="store_true", help="prime .cache with the jupyter/airflow fixtures")
    sources.add_argument("--snapshot", action="store_true", help="archive the pristine source tree")
    sources.set_defaults(func=cmd_sources)

    binaries = subparsers.add_parser("binaries", help="build one revision x integration benchmark binaries")
    common(binaries)
    binaries.add_argument("--revision", choices=REVISIONS, required=True)
    binaries.add_argument("--integration", choices=INTEGRATIONS, required=True)
    binaries.add_argument(
        "--codspeed-source",
        type=Path,
        default=Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "codspeed-rust",
        help="codspeed-rust checkout path (one pinned revision per integration)",
    )
    binaries.add_argument(
        "--tools-root",
        type=Path,
        default=Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "stack-isolation-tools",
        help="where the pinned cargo-codspeed is installed (never uploaded)",
    )
    binaries.set_defaults(func=cmd_binaries)

    valgrind = subparsers.add_parser("valgrind", help="build one pinned Valgrind into the artifact prefix layout")
    common(valgrind)
    valgrind.add_argument("--name", choices=VALGRIND_BUILDS, required=True)
    valgrind.set_defaults(func=cmd_valgrind)

    prepare = subparsers.add_parser("prepare", help="extract artifact archives restoring modes and symlinks")
    prepare.add_argument("--artifacts", type=Path, required=True)
    prepare.set_defaults(func=cmd_prepare)

    verify = subparsers.add_parser("verify", help="validate the complete artifact set before measurement")
    verify.add_argument("--artifacts", type=Path, required=True)
    verify.set_defaults(func=cmd_verify)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
