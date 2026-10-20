#!/usr/bin/env python3
"""Build the frozen artifacts for one variance-lab candidate.

Subcommands:
  sources-import  Keep only ``sources/head.tar.zst`` from an imported artifact tree.
  binaries        Build ``uv-bench`` Simulation executables with a pinned codspeed-rust
                  revision patched over crates.io, from the frozen ``sources/head`` tree.
  valgrind        Build valgrind-codspeed at a pinned revision into the fixed prefix.
  prepare         Extract every lab archive in place (sources/binaries/valgrind).

The uv source tree (with its primed ``.venv``/``.cache``) is never rebuilt: it is the
frozen ``head`` snapshot produced by the earlier stack-isolation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

UV_HEAD_SHA = "c2d613f98dac62b4676c2d9b3a5eb3f9b3865ecc"
VALGRIND_PREFIX = Path("/opt/stack-isolation/valgrind")
CODSPEED_TARGET_SUBDIR = Path("target/codspeed/analysis/uv-bench")
WORKLOAD_MANIFEST_DIR = Path("crates/uv-bench")
BINARIES_NAME = "head-new"
VALGRIND_NAME = "new"
ARCHIVE_GLOBS = ("sources/*.tar.zst", "binaries/*.tar.zst", "valgrind/*.tar.zst")


def log(message: str) -> None:
    print(f"[variance-lab] {message}", flush=True)


def run(command, *, cwd=None, env=None, capture=False) -> str:
    printable = " ".join(shlex.quote(str(part)) for part in command)
    log(f"$ {printable}" + (f"  (cwd={cwd})" if cwd else ""))
    if capture:
        return subprocess.run(
            [str(part) for part in command], cwd=cwd, env=env, check=True, text=True, capture_output=True
        ).stdout
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)
    return ""


def run_shell(script: str, *, cwd=None, env=None) -> None:
    run(["bash", "-euo", "pipefail", "-c", script], cwd=cwd, env=env)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_tree(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def git_checkout(url: str, revision: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        run(["git", "init", "--quiet", destination])
        run(["git", "-C", destination, "remote", "add", "origin", url])
    else:
        run(["git", "-C", destination, "remote", "set-url", "origin", url])
    run(["git", "-C", destination, "fetch", "--depth", "1", "--no-tags", "origin", revision])
    run(["git", "-C", destination, "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    actual = run(["git", "-C", destination, "rev-parse", "HEAD"], capture=True).strip()
    if actual != revision:
        raise SystemExit(f"checkout mismatch: expected {revision}, got {actual} in {destination}")


def make_archive(artifacts: Path, relative: Path) -> Path:
    archive = (artifacts / relative).with_suffix(".tar.zst")
    archive.parent.mkdir(parents=True, exist_ok=True)
    run_shell(
        f"tar -cf - -C {shlex.quote(str(artifacts))} {shlex.quote(relative.as_posix())} "
        f"| zstd -T0 -3 -q -f -o {shlex.quote(str(archive))}"
    )
    return archive


def extract_archives(artifacts: Path, patterns=ARCHIVE_GLOBS) -> list[Path]:
    archives = sorted(path for pattern in patterns for path in artifacts.glob(pattern))
    for archive in archives:
        log(f"extract {archive}")
        run_shell(f"zstd -dc {shlex.quote(str(archive))} | tar -xf - -C {shlex.quote(str(artifacts))}")
    return archives


def restore_workload(artifacts: Path, workload: Path) -> dict:
    """Copy the frozen head source tree to the canonical workload path."""
    source = artifacts / "sources" / "head"
    metadata = json.loads((source / "experiment-source.json").read_text())
    if metadata["source_sha"] != UV_HEAD_SHA:
        raise SystemExit(f"sources artifact is {metadata['source_sha']}, expected {UV_HEAD_SHA}")
    if metadata["workload"] != str(workload):
        raise SystemExit(f"sources were primed at {metadata['workload']}, not {workload}")
    if workload.exists():
        shutil.rmtree(workload)
    shutil.copytree(source, workload, symlinks=True)
    head = run(["git", "-C", workload, "rev-parse", "HEAD"], capture=True).strip()
    if head != UV_HEAD_SHA:
        raise SystemExit(f"restored workload HEAD is {head}, expected {UV_HEAD_SHA}")
    return metadata


def cmd_sources_import(args) -> None:
    artifacts = args.artifacts.resolve()
    archive = artifacts / "sources" / "head.tar.zst"
    if not archive.is_file():
        raise SystemExit(f"imported artifact has no {archive}")
    for entry in sorted(artifacts.iterdir()):
        if entry.name == "sources":
            continue
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    for entry in sorted((artifacts / "sources").iterdir()):
        if entry != archive:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    write_json(
        artifacts / "sources" / "import.json",
        {"source_sha": UV_HEAD_SHA, "sha256": sha256_file(archive), "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )
    log(f"kept {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


def cmd_binaries(args) -> None:
    artifacts = args.artifacts.resolve()
    workload = args.workload.resolve()
    extract_archives(artifacts, ("sources/*.tar.zst",))
    restore_workload(artifacts, workload)
    env = dict(os.environ, UV_LOCKED="1")

    codspeed_src = args.codspeed_source.resolve()
    git_checkout(args.codspeed_url, args.codspeed_rev, codspeed_src)
    run(["git", "-C", codspeed_src, "submodule", "update", "--init", "--recursive"])
    tools_root = args.tools_root.resolve() / "cargo-codspeed-lab"
    run(["cargo", "install", "--locked", "--path", codspeed_src / "crates" / "cargo-codspeed", "--root", tools_root], cwd=workload, env=env)
    cargo_codspeed = tools_root / "bin" / "cargo-codspeed"

    manifest = workload / "Cargo.toml"
    original = manifest.read_text()
    patch = (
        "\n[patch.crates-io]\n"
        f'codspeed = {{ git = "{args.codspeed_url}", rev = "{args.codspeed_rev}" }}\n'
        f'codspeed-criterion-compat = {{ git = "{args.codspeed_url}", rev = "{args.codspeed_rev}" }}\n'
    )
    try:
        manifest.write_text(original.rstrip() + "\n" + patch)
        env_new = dict(env, PATH=f"{tools_root / 'bin'}:{env['PATH']}")
        run(["cargo", "metadata", "--format-version", "1"], cwd=workload, env=env_new, capture=True)
        run([cargo_codspeed, "build", "--locked", "--profile", "profiling", "-p", "uv-bench"], cwd=workload, env=env_new)
        for crate in ("codspeed-criterion-compat", "codspeed"):
            tree = run(["cargo", "tree", "-i", crate], cwd=workload, env=env_new, capture=True)
            if args.codspeed_rev not in tree:
                raise SystemExit(f"{crate} patch did not take effect:\n{tree}")
            (artifacts / f"cargo-tree-{crate}.txt").write_text(tree)
        lockfile = artifacts / "locks" / "Cargo.lock"
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workload / "Cargo.lock", lockfile)
    finally:
        manifest.write_text(original)

    source = workload / CODSPEED_TARGET_SUBDIR
    targets = sorted(path for path in source.iterdir() if path.is_file() and not path.is_symlink())
    if not targets:
        raise SystemExit(f"no benchmark executables in {source}")
    destination = artifacts / "binaries" / BINARIES_NAME
    if destination.exists():
        shutil.rmtree(destination)
    binaries = destination / "uv-bench"
    binaries.mkdir(parents=True)
    for path in targets:
        target = binaries / path.name
        shutil.copy2(path, target)
        target.chmod(target.stat().st_mode | 0o755)
    write_json(
        destination / "build.json",
        {
            "schema_version": 1,
            "revision": "head",
            "source_sha": UV_HEAD_SHA,
            "integration": "new",
            "rust_sha": args.codspeed_rev,
            "rust_url": args.codspeed_url,
            "rust_integration": "new",
            "cargo_codspeed_version": run(["cargo", "install", "--list", "--root", tools_root], capture=True).strip(),
            "workload": str(workload),
            "compiled_manifest_dir": str(workload / WORKLOAD_MANIFEST_DIR),
            "targets": [path.name for path in targets],
            "sha256": hash_tree(binaries),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "extra": {"codspeed_rust_path": str(codspeed_src), "patch_crates_io": True},
        },
    )
    archive = make_archive(artifacts, Path("binaries") / BINARIES_NAME)
    log(f"{BINARIES_NAME}: {[p.name for p in targets]} -> {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(destination)
    # Only the binaries archive ships from this job.
    shutil.rmtree(artifacts / "sources", ignore_errors=True)


def cmd_valgrind(args) -> None:
    artifacts = args.artifacts.resolve()
    checkout = args.git_cache.resolve() / "valgrind-codspeed"
    git_checkout(args.valgrind_url, args.valgrind_rev, checkout)
    run(["git", "-C", checkout, "submodule", "update", "--init", "--recursive", "--depth", "1", "third_party/capstone"])
    run(["./autogen.sh"], cwd=checkout)
    run(["./configure", f"--prefix={VALGRIND_PREFIX}", "--enable-only64bit"], cwd=checkout)
    run(["make", "-j", str(os.cpu_count() or 1)], cwd=checkout)
    staging = Path(tempfile.mkdtemp(prefix="valgrind-install-"))
    try:
        run(["make", "install", f"DESTDIR={staging}"], cwd=checkout)
        installed = staging / VALGRIND_PREFIX.relative_to("/")
        if not (installed / "bin" / "valgrind").is_file():
            raise SystemExit(f"valgrind install missing launcher: {installed}")
        destination = artifacts / "valgrind" / VALGRIND_NAME
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(installed, destination, symlinks=True)
        env = dict(os.environ, VALGRIND_LIB=str(destination / "libexec" / "valgrind"))
        version = run([destination / "bin" / "valgrind", "--version"], env=env, capture=True).strip()
        callgrind = sorted(destination.rglob("callgrind-amd64-linux"))
        if len(callgrind) != 1:
            raise SystemExit(f"expected exactly one callgrind-amd64-linux, found {callgrind}")
        write_json(
            artifacts / "valgrind" / f"{VALGRIND_NAME}.json",
            {
                "schema_version": 1,
                "name": VALGRIND_NAME,
                "source_sha": args.valgrind_rev,
                "source_url": args.valgrind_url,
                "prefix": str(VALGRIND_PREFIX),
                "version": version,
                "callgrind": callgrind[0].relative_to(destination).as_posix(),
                "sha256": hash_tree(destination),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        archive = make_archive(artifacts, Path("valgrind") / VALGRIND_NAME)
        log(f"valgrind {version} -> {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
        shutil.rmtree(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def cmd_prepare(args) -> None:
    artifacts = args.artifacts.resolve()
    archives = extract_archives(artifacts)
    if not archives:
        raise SystemExit(f"no lab archives under {artifacts}")
    for archive in archives:
        archive.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sources-import")
    p.add_argument("--artifacts", type=Path, required=True)
    p.set_defaults(func=cmd_sources_import)

    p = sub.add_parser("binaries")
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--codspeed-source", type=Path, required=True)
    p.add_argument("--tools-root", type=Path, required=True)
    p.add_argument("--codspeed-url", required=True)
    p.add_argument("--codspeed-rev", required=True)
    p.set_defaults(func=cmd_binaries)

    p = sub.add_parser("valgrind")
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--git-cache", type=Path, required=True)
    p.add_argument("--valgrind-url", required=True)
    p.add_argument("--valgrind-rev", required=True)
    p.set_defaults(func=cmd_valgrind)

    p = sub.add_parser("prepare")
    p.add_argument("--artifacts", type=Path, required=True)
    p.set_defaults(func=cmd_prepare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
