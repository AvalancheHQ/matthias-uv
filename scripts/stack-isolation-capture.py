#!/usr/bin/env python3
"""Measure frozen uv benchmark artifacts with pinned Valgrind builds."""

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path

REVISIONS = {
    "base": "21979a0931e931b122c394aeb0bd731e785de30c",
    "head": "c2d613f98dac62b4676c2d9b3a5eb3f9b3865ecc",
}
VALGRIND_REVISIONS = {
    "old": "e1e4bdf3fa3daa61cce684b513f874ff0735b1b8",
    "new": "5275f9f34b780c8c9320906a14ef6e160d45bf94",
}
RUST_REVISION = "304fa86cb8ecfd95cf6d0c1e4f05aae0f17da96a"
VALGRIND_PREFIX = Path("/opt/stack-isolation/valgrind")
CELLS = [(config, revision) for config in "ABC" for revision in REVISIONS]
# Williams ordering balances immediate predecessors over the first six rounds.
BALANCED_ORDER = (0, 1, 5, 2, 4, 3)
COUNTERS = ("Ct", "Cl", "Dr", "Dw", "D1mr", "D1mw", "DLmr", "DLmw", "ILmr")
CPU_HZ = 3_600_000_000
L1_CYCLES = 5
LL_CYCLES = 40
MEMORY_CYCLES = 200


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(directory):
    return {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def score(counters):
    missing = set(COUNTERS) - counters.keys()
    if missing:
        raise ValueError(f"Missing simulation counters: {sorted(missing)}")
    components = {
        "compute": math.sqrt(counters["Ct"] * counters["Cl"]) / 100,
        "l1": L1_CYCLES
        * (counters["Dr"] + counters["Dw"] - counters["D1mr"] - counters["D1mw"]),
        "ll": LL_CYCLES
        * (counters["D1mr"] + counters["D1mw"] - counters["DLmr"] - counters["DLmw"]),
        "memory": MEMORY_CYCLES
        * (counters["DLmr"] + counters["DLmw"] + counters["ILmr"]),
    }
    return sum(components.values()) / CPU_HZ


def summarize(directory):
    results = {}
    for path in sorted(directory.glob("*.out")):
        name = None
        events = []
        pid = path.stem
        with path.open() as stream:
            for line in stream:
                if line.startswith("pid:"):
                    pid = line.split(":", 1)[1].strip()
                elif line.startswith("part:"):
                    name = None
                elif line.startswith("desc: Trigger: Client Request: "):
                    name = line.removeprefix("desc: Trigger: Client Request: ").strip()
                elif line.startswith("events:"):
                    events = line.split()[1:]
                    if len(events) != len(set(events)):
                        raise ValueError(f"Duplicate counter names in {path}")
                elif line.startswith("totals:") and name and "::" in name:
                    values = [int(value) for value in line.split()[1:]]
                    if len(values) > len(events):
                        raise ValueError(f"Too many counter values in {path}")
                    counters = dict(zip(events, values + [0] * (len(events) - len(values)), strict=True))
                    if name in results:
                        raise ValueError(f"Duplicate measured benchmark: {name}")
                    results[name] = {
                        "seconds": score(counters),
                        "counters": counters,
                        "file": path.name,
                        "pid": pid,
                    }
    if not results:
        raise ValueError(f"No measured benchmarks in {directory}")
    return results


def schedule(job, rounds, preflight):
    if preflight:
        return [(0, position, config, "base") for position, config in enumerate("ABC")]
    rows = []
    for local_round in range(rounds):
        global_round = job * rounds + local_round
        order = [(index + global_round) % len(CELLS) for index in BALANCED_ORDER]
        if global_round >= len(CELLS):
            order.reverse()
        rows.extend((local_round, position, *CELLS[index]) for position, index in enumerate(order))
    return rows


def output_of(command):
    return subprocess.check_output(command, text=True).strip()


def install_valgrind(source):
    # Only this experiment's dedicated prefix is replaced; the system install stays intact.
    subprocess.run(["sudo", "mkdir", "-p", str(VALGRIND_PREFIX.parent)], check=True)
    subprocess.run(["sudo", "rm", "-rf", str(VALGRIND_PREFIX)], check=True)
    subprocess.run(["sudo", "cp", "-a", str(source), str(VALGRIND_PREFIX)], check=True)
    tools = list(VALGRIND_PREFIX.rglob("callgrind-amd64-linux"))
    if len(tools) != 1:
        raise ValueError(f"Expected exactly one installed Callgrind executable, found {tools}")
    return tools[0].parent


def prepare_workload(artifacts, workload, revision, integration):
    source_metadata = json.loads((artifacts / "sources" / revision / "experiment-source.json").read_text())
    build_metadata = json.loads((artifacts / "binaries" / f"{revision}-{integration}" / "build.json").read_text())
    if source_metadata["source_sha"] != REVISIONS[revision] or build_metadata["source_sha"] != REVISIONS[revision]:
        raise ValueError("Artifact source SHA does not match the pinned revision")
    if integration == "new" and build_metadata["rust_sha"] != RUST_REVISION:
        raise ValueError("Artifact Rust integration SHA does not match PR #194")
    if build_metadata["workload"] != str(workload):
        raise ValueError("Runtime workload path differs from the compiled artifact path")
    if workload.exists():
        shutil.rmtree(workload)
    shutil.copytree(artifacts / "sources" / revision, workload, symlinks=True)
    if output_of(["git", "-C", str(workload), "rev-parse", "HEAD"]) != REVISIONS[revision]:
        raise ValueError("Restored workload Git HEAD differs from the pinned revision")
    binaries = artifacts / "binaries" / f"{revision}-{integration}" / "uv-bench"
    if hashes(binaries) != build_metadata["sha256"]:
        raise ValueError("Benchmark artifact checksums do not match build provenance")
    destination = workload / "target" / "codspeed" / "analysis" / "uv-bench"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(binaries, destination)
    targets = sorted(path for path in destination.iterdir() if path.is_file() and os.access(path, os.X_OK))
    if not targets:
        raise ValueError(f"No executable targets in {destination}")
    return destination, targets


def capture(args):
    artifacts = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workload = Path.cwd() / "workload"
    runner = Path(shutil.which("codspeed") or "codspeed").resolve()
    runner_version = output_of([str(runner), "--version"])
    if runner_version != "codspeed-runner 5.2.1":
        raise ValueError(f"Unexpected runner: {runner_version}")
    cpu = json.loads(output_of(["lscpu", "--json"]))
    cpu_model = next(row["data"] for row in cpu["lscpu"] if row["field"] == "Model name:")
    plan = schedule(args.job, args.rounds, args.preflight)
    save(output / "schedule.json", [dict(round=r, order=o, config=c, revision=v) for r, o, c, v in plan])
    expected_names = None
    expected_targets = None
    failures = []
    for round_number, position, config, revision in plan:
        sample = output / f"sample-{round_number:02}-{config}-{revision}"
        sample.mkdir()
        profiles = sample / "profiles"
        profiles.mkdir()
        integration = "new" if config == "C" else "old"
        valgrind = "old" if config == "A" else "new"
        manifest = {
            "schema_version": 1,
            "job": args.job,
            "round": round_number,
            "order": position,
            "preflight": args.preflight,
            "config": config,
            "revision": revision,
            "source_sha": REVISIONS[revision],
            "rust_integration": RUST_REVISION if integration == "new" else "locked-5.0.1",
            "valgrind_sha": VALGRIND_REVISIONS[valgrind],
            "cpu_model": cpu_model,
            "cpu": cpu,
            "kernel": platform.release(),
            "glibc": output_of(["getconf", "GNU_LIBC_VERSION"]),
            "runner_version": runner_version,
            "runner_sha256": digest(runner),
            "image_os": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "workflow_url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}",
            "workflow_sha": os.environ.get("GITHUB_SHA"),
            "upload": "skipped; raw profiles retained in artifact",
            "status": "running",
        }
        save(sample / "manifest.json", manifest)
        started = time.monotonic()
        try:
            binary_dir, targets = prepare_workload(artifacts, workload, revision, integration)
            target_names = [path.name for path in targets]
            if expected_targets is None:
                expected_targets = target_names
            elif target_names != expected_targets:
                raise ValueError(f"Target coverage changed: {target_names} != {expected_targets}")
            binary_hashes = hashes(binary_dir)
            if binary_hashes != hashes(artifacts / "binaries" / f"{revision}-{integration}" / "uv-bench"):
                raise ValueError("Copied benchmark binary checksum mismatch")
            valgrind_source = artifacts / "valgrind" / valgrind
            valgrind_lib = install_valgrind(valgrind_source)
            valgrind_hashes = hashes(VALGRIND_PREFIX)
            if valgrind_hashes != hashes(valgrind_source):
                raise ValueError("Installed Valgrind checksum mismatch")
            environment = dict(os.environ)
            for key in ("CODSPEED_PROCESS_ISOLATION", "CODSPEED_ISOLATED_BENCHMARK"):
                environment.pop(key, None)
            environment.update(
                PATH=f"{VALGRIND_PREFIX / 'bin'}:{os.environ['PATH']}",
                VALGRIND_LIB=str(valgrind_lib),
                CODSPEED_CARGO_WORKSPACE_ROOT=str(workload),
                CODSPEED_PROFILER_ENABLED="false",
                CODSPEED_RUNNER_MODE="simulation",
                CODSPEED_LOG="debug",
            )
            run_script = workload / "run-frozen-suite.sh"
            run_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(shlex.quote(str(path)) for path in targets) + "\n")
            command = [str(runner), "run", "--mode", "simulation", "--skip-setup", "--skip-upload", "--profile-folder", str(profiles), "--show-full-output", "--", f"bash {shlex.quote(str(run_script))}"]
            manifest.update(
                binary_hashes=binary_hashes,
                valgrind_hashes=valgrind_hashes,
                targets=target_names,
                command=command,
                working_directory=str(workload / "crates" / "uv-bench"),
                benchmark_commands=[[str(path)] for path in targets],
                environment={key: environment[key] for key in ("PATH", "VALGRIND_LIB", "CODSPEED_CARGO_WORKSPACE_ROOT", "CODSPEED_PROFILER_ENABLED", "CODSPEED_RUNNER_MODE", "CODSPEED_LOG")},
                isolation_env="unset",
                valgrind_version=output_of([str(VALGRIND_PREFIX / "bin" / "valgrind"), "--version"]),
            )
            save(sample / "manifest.json", manifest)
            measured_start = time.monotonic()
            with (sample / "runner.log").open("w") as log:
                subprocess.run(command, cwd=workload / "crates" / "uv-bench", env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest["measurement_duration_seconds"] = time.monotonic() - measured_start
            results = summarize(profiles)
            save(sample / "results.json", results)
            names = set(results)
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise ValueError(f"Measured coverage changed: missing={sorted(expected_names - names)}, extra={sorted(names - expected_names)}")
            pids = {result["pid"] for result in results.values()}
            if config == "C" and len(pids) != len(results):
                raise ValueError(f"Isolation not verified: {len(results)} benchmarks measured by {len(pids)} processes")
            if config != "C" and len(pids) >= len(results):
                raise ValueError("Shared-process control unexpectedly has one measuring process per benchmark")
            if hashes(VALGRIND_PREFIX) != valgrind_hashes:
                raise ValueError("Valgrind installation changed during measurement")
            manifest.update(status="success", benchmark_count=len(results), measured_process_count=len(pids))
        except Exception as error:
            manifest.update(status="failed", error=str(error))
            failures.append(f"{sample.name}: {error}")
            print(f"FAILED {sample.name}: {error}", flush=True)
        finally:
            manifest["duration_seconds"] = time.monotonic() - started
            save(sample / "manifest.json", manifest)
            subprocess.run(["tar", "--zstd", "-cf", str(sample / "profiles.tar.zst"), "-C", str(sample), "profiles"], check=True)
            shutil.rmtree(profiles)
        print(json.dumps({key: manifest.get(key) for key in ("job", "round", "order", "config", "revision", "status", "benchmark_count", "duration_seconds")}), flush=True)
        # Stop on failed preflight instead of spending the remaining CI allocation.
        if failures and args.preflight:
            break
    if failures:
        raise RuntimeError("\n".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
