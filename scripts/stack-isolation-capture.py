#!/usr/bin/env python3
"""Run one frozen Simulation suite per macro job.

Config D = config C (new integration, process isolation, new Valgrind) plus
the warm-cache warmup of codspeed-rust @061f923: instrumentation is switched
on before the warmup iterations and only the counters are zeroed before the
measured iteration, so the measurement starts from a warm simulated cache.

Twenty jobs measure each D cell (global jobs 300..339; A/B/C used 0..299).
Physical machines may be reused after reboot; samples are not paired by job
or replicate number.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
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
RUST_REVISIONS = {
    "new": "304fa86cb8ecfd95cf6d0c1e4f05aae0f17da96a",
    "warm": "061f9230eac19fe27c95b8bccef186732558f35e",
}
EXPERIMENT = "macro-abcd"
BATCH = "warmup-20"
# integration (binaries) and Valgrind build per configuration.
CONFIGS = {
    "A": ("old", "old"),
    "B": ("old", "new"),
    "C": ("new", "new"),
    "D": ("warm", "new"),
}
VALGRIND_PREFIX = Path("/opt/stack-isolation/valgrind")
# Cell indices continue the A/B/C numbering (0..5) of the earlier batches.
ALL_CELLS = [(config, revision) for config in CONFIGS for revision in REVISIONS]
CELLS = [(config, revision) for config in "D" for revision in REVISIONS]
RUNS_PER_CELL = 20
FIRST_MEASUREMENT_JOB = 300
MEASUREMENTS = [
    (round_number, 0, config, revision)
    for config, revision in CELLS
    for round_number in range(RUNS_PER_CELL)
]
MEASUREMENT_JOB_COUNT = FIRST_MEASUREMENT_JOB + len(MEASUREMENTS)
EXPECTED_BENCHMARK_COUNT = 68
SHARED_PROCESS_COUNT = 6
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
                    counters = dict(
                        zip(
                            events,
                            values + [0] * (len(events) - len(values)),
                            strict=True,
                        )
                    )
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


# Benchmark URIs may contain spaces (e.g. version specifiers), so the URI is
# everything between the prefix and the trailing warmup suffix.
WARMUP_LINE = re.compile(r"^Measured: (?P<uri>.+?)(?: \(group: [^)]*\))? \(warmup: (?P<iterations>\d+) iterations\)$")


def attach_warmup_iterations(results, runner_log, required):
    """Record how many unmeasured iterations preceded each measured one.

    Only the warm integration prints the count (``required``); benchmarks
    without a line keep no entry so the report can tell "not reported" from a
    count. A warm suite without any count means the binaries were not built
    from the warm integration, which a hash check cannot tell apart.
    """
    counts = {}
    with runner_log.open(errors="replace") as stream:
        for line in stream:
            match = WARMUP_LINE.match(line.strip())
            if not match:
                continue
            uri = match.group("uri")
            if uri in counts:
                raise ValueError(f"Duplicate warmup line for {uri}")
            counts[uri] = int(match.group("iterations"))
    for uri, iterations in counts.items():
        if uri not in results:
            raise ValueError(f"Warmup line for unmeasured benchmark {uri}")
        results[uri]["warmup_iterations"] = iterations
    if required and not counts:
        raise ValueError("Warm integration printed no warmup iteration counts")
    if counts and len(counts) != len(results):
        raise ValueError(
            f"Warmup counts for {len(counts)} of {len(results)} measured benchmarks"
        )


def schedule(job):
    if not FIRST_MEASUREMENT_JOB <= job < MEASUREMENT_JOB_COUNT:
        raise ValueError(
            f"measurement job must be in [{FIRST_MEASUREMENT_JOB}, {MEASUREMENT_JOB_COUNT - 1}], got {job}"
        )
    return [MEASUREMENTS[job - FIRST_MEASUREMENT_JOB]]


def output_of(command):
    return subprocess.check_output(command, text=True).strip()


def install_valgrind(source):
    # Only this experiment's dedicated prefix is replaced; the system install stays intact.
    subprocess.run(["sudo", "mkdir", "-p", str(VALGRIND_PREFIX.parent)], check=True)
    subprocess.run(["sudo", "rm", "-rf", str(VALGRIND_PREFIX)], check=True)
    subprocess.run(["sudo", "cp", "-a", str(source), str(VALGRIND_PREFIX)], check=True)
    tools = list(VALGRIND_PREFIX.rglob("callgrind-amd64-linux"))
    if len(tools) != 1:
        raise ValueError(
            f"Expected exactly one installed Callgrind executable, found {tools}"
        )
    return tools[0].parent


def prepare_workload(artifacts, workload, revision, integration):
    source_metadata = json.loads(
        (artifacts / "sources" / revision / "experiment-source.json").read_text()
    )
    build_metadata = json.loads(
        (
            artifacts / "binaries" / f"{revision}-{integration}" / "build.json"
        ).read_text()
    )
    if (
        source_metadata["source_sha"] != REVISIONS[revision]
        or build_metadata["source_sha"] != REVISIONS[revision]
    ):
        raise ValueError("Artifact source SHA does not match the pinned revision")
    if build_metadata["workload"] != str(workload):
        raise ValueError(
            "Runtime workload path differs from the compiled artifact path"
        )
    if workload.exists():
        shutil.rmtree(workload)
    shutil.copytree(artifacts / "sources" / revision, workload, symlinks=True)
    if (
        output_of(["git", "-C", str(workload), "rev-parse", "HEAD"])
        != REVISIONS[revision]
    ):
        raise ValueError("Restored workload Git HEAD differs from the pinned revision")
    binaries = artifacts / "binaries" / f"{revision}-{integration}" / "uv-bench"
    if hashes(binaries) != build_metadata["sha256"]:
        raise ValueError("Benchmark artifact checksums do not match build provenance")
    destination = workload / "target" / "codspeed" / "analysis" / "uv-bench"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(binaries, destination)
    targets = sorted(
        path
        for path in destination.iterdir()
        if path.is_file() and os.access(path, os.X_OK)
    )
    if not targets:
        raise ValueError(f"No executable targets in {destination}")
    return destination, targets


def capture(args):
    artifacts = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workload = Path(
        os.environ.get("STACK_ISOLATION_WORKLOAD", Path.cwd() / "workload")
    ).resolve()
    runner = Path(shutil.which("codspeed") or "codspeed").resolve()
    runner_version = output_of([str(runner), "--version"])
    if runner_version != "codspeed-runner 5.2.1":
        raise ValueError(f"Unexpected runner: {runner_version}")
    cpu = json.loads(output_of(["lscpu", "--json"]))
    cpu_model = next(
        row["data"] for row in cpu["lscpu"] if row["field"] == "Model name:"
    )
    product_uuid = Path("/sys/class/dmi/id/product_uuid")
    machine_id = (
        hashlib.sha256(
            output_of(["sudo", "cat", str(product_uuid)]).encode()
        ).hexdigest()
        if product_uuid.exists()
        else None
    )
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    packages = output_of(["dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"])
    package_file = output / "package-versions.txt"
    package_file.write_text("\n".join(sorted(packages.splitlines())) + "\n")
    plan = schedule(args.job)
    save(
        output / "schedule.json",
        [
            {
                "job": args.job,
                "round": r,
                "order": o,
                "cell": ALL_CELLS.index((c, v)),
                "config": c,
                "revision": v,
            }
            for r, o, c, v in plan
        ],
    )
    failures = []
    for round_number, position, config, revision in plan:
        sample = output / f"sample-{round_number:02}-{config}-{revision}"
        sample.mkdir()
        profiles = sample / "profiles"
        profiles.mkdir()
        integration, valgrind = CONFIGS[config]
        manifest = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "batch": BATCH,
            "job": args.job,
            "round": round_number,
            "order": position,
            "cell": ALL_CELLS.index((config, revision)),
            "preflight": args.preflight,
            "config": config,
            "revision": revision,
            "source_sha": REVISIONS[revision],
            "rust_integration": RUST_REVISIONS.get(integration, "locked-5.0.2"),
            "valgrind_sha": VALGRIND_REVISIONS[valgrind],
            "cpu_model": cpu_model,
            "cpu": cpu,
            "kernel": platform.release(),
            "glibc": output_of(["getconf", "GNU_LIBC_VERSION"]),
            "os_release": platform.freedesktop_os_release(),
            "package_versions_sha256": digest(package_file),
            "started_at": time.time(),
            "runner_version": runner_version,
            "runner_sha256": digest(runner),
            "machine_id": machine_id,
            "boot_id": boot_id,
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "hostname": platform.node() or None,
            "image_os": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "workflow_url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}",
            "workflow_sha": os.environ.get("GITHUB_SHA"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
            "workflow_run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
            "upload": "skipped; raw profiles retained in artifact",
            "status": "running",
        }
        save(sample / "manifest.json", manifest)
        started = time.monotonic()
        try:
            binary_dir, targets = prepare_workload(
                artifacts, workload, revision, integration
            )
            target_names = [path.name for path in targets]
            binary_hashes = hashes(binary_dir)
            if binary_hashes != hashes(
                artifacts / "binaries" / f"{revision}-{integration}" / "uv-bench"
            ):
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
            run_script.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                + "\n".join(shlex.quote(str(path)) for path in targets)
                + "\n"
            )
            command = [
                str(runner),
                "run",
                "--mode",
                "simulation",
                "--skip-setup",
                "--skip-upload",
                "--profile-folder",
                str(profiles),
                "--show-full-output",
                "--",
                f"bash {shlex.quote(str(run_script))}",
            ]
            manifest.update(
                binary_hashes=binary_hashes,
                valgrind_hashes=valgrind_hashes,
                targets=target_names,
                command=command,
                working_directory=str(workload / "crates" / "uv-bench"),
                benchmark_commands=[[str(path)] for path in targets],
                environment={
                    key: environment[key]
                    for key in (
                        "PATH",
                        "VALGRIND_LIB",
                        "CODSPEED_CARGO_WORKSPACE_ROOT",
                        "CODSPEED_PROFILER_ENABLED",
                        "CODSPEED_RUNNER_MODE",
                        "CODSPEED_LOG",
                    )
                },
                isolation_env="unset",
                valgrind_version=output_of(
                    [str(VALGRIND_PREFIX / "bin" / "valgrind"), "--version"]
                ),
            )
            save(sample / "manifest.json", manifest)
            measured_start = time.monotonic()
            with (sample / "runner.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=workload / "crates" / "uv-bench",
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            manifest["measurement_duration_seconds"] = time.monotonic() - measured_start
            results = summarize(profiles)
            attach_warmup_iterations(
                results, sample / "runner.log", required=integration == "warm"
            )
            save(sample / "results.json", results)
            names = set(results)
            if len(names) != EXPECTED_BENCHMARK_COUNT:
                raise ValueError(
                    f"Expected {EXPECTED_BENCHMARK_COUNT} measured benchmarks, got {len(names)}"
                )
            pids = {result["pid"] for result in results.values()}
            expected_process_count = (
                EXPECTED_BENCHMARK_COUNT if integration in ("new", "warm") else SHARED_PROCESS_COUNT
            )
            if len(pids) != expected_process_count:
                raise ValueError(
                    f"Expected {expected_process_count} measuring processes for {config}, got {len(pids)}"
                )
            if hashes(VALGRIND_PREFIX) != valgrind_hashes:
                raise ValueError("Valgrind installation changed during measurement")
            manifest.update(
                status="success",
                benchmark_count=len(results),
                measured_process_count=len(pids),
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            manifest.update(status="failed", error=str(error))
            failures.append(f"{sample.name}: {error}")
            print(f"FAILED {sample.name}: {error}", flush=True)
        finally:
            manifest["duration_seconds"] = time.monotonic() - started
            save(sample / "manifest.json", manifest)
            subprocess.run(
                [
                    "tar",
                    "--zstd",
                    "-cf",
                    str(sample / "profiles.tar.zst"),
                    "-C",
                    str(sample),
                    "profiles",
                ],
                check=True,
            )
            shutil.rmtree(profiles)
        print(
            json.dumps(
                {
                    key: manifest.get(key)
                    for key in (
                        "job",
                        "round",
                        "order",
                        "config",
                        "revision",
                        "status",
                        "benchmark_count",
                        "duration_seconds",
                    )
                }
            ),
            flush=True,
        )
    if failures:
        raise RuntimeError("\n".join(failures))


def job_id(value):
    try:
        job = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"measurement job must be an integer, got {value!r}"
        ) from error
    if not FIRST_MEASUREMENT_JOB <= job < MEASUREMENT_JOB_COUNT:
        raise argparse.ArgumentTypeError(
            f"measurement job must be in [{FIRST_MEASUREMENT_JOB}, {MEASUREMENT_JOB_COUNT - 1}], got {job}"
        )
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", type=job_id, default=FIRST_MEASUREMENT_JOB)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
