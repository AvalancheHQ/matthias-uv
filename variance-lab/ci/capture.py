#!/usr/bin/env python3
"""Run one variance-lab candidate once on this macro job and keep the raw evidence.

Every job restores the same frozen uv ``head`` workload, the same benchmark
executables and the same Valgrind prefix, then executes the selected benchmark
targets exactly once under ``codspeed run --mode simulation``. The candidate
spec controls only the measurement environment: extra environment variables,
environment sanitisation, a command wrapper, runner flags and setup commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import BINARIES_NAME, VALGRIND_NAME, VALGRIND_PREFIX, hash_tree, restore_workload  # noqa: E402

RUNNER_VERSION = "codspeed-runner 5.2.1"
COUNTERS = ("Ct", "Cl", "Dr", "Dw", "D1mr", "D1mw", "DLmr", "DLmw", "ILmr")
CPU_HZ = 3_600_000_000
L1_CYCLES = 5
LL_CYCLES = 40
MEMORY_CYCLES = 200
# Environment kept for the benchmark process when ``env_mode == "fixed"``.
FIXED_ENV_EXACT = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR")
FIXED_ENV_PREFIXES = ("CODSPEED_", "VALGRIND_", "LD_", "_RJEM_", "MALLOC_", "RUST", "CARGO", "TOKIO_", "UV_")


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_of(command, **kwargs) -> str:
    return subprocess.check_output(command, text=True, **kwargs).strip()


def score(counters: dict) -> float:
    missing = set(COUNTERS) - counters.keys()
    if missing:
        raise ValueError(f"Missing simulation counters: {sorted(missing)}")
    compute = math.sqrt(counters["Ct"] * counters["Cl"]) / 100
    l1 = L1_CYCLES * (counters["Dr"] + counters["Dw"] - counters["D1mr"] - counters["D1mw"])
    ll = LL_CYCLES * (counters["D1mr"] + counters["D1mw"] - counters["DLmr"] - counters["DLmw"])
    memory = MEMORY_CYCLES * (counters["DLmr"] + counters["DLmw"] + counters["ILmr"])
    return (compute + l1 + ll + memory) / CPU_HZ


def summarize(directory: Path) -> dict:
    results = {}
    for path in sorted(directory.glob("*.out")):
        name = None
        events: list[str] = []
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
                    results[name] = {"seconds": score(counters), "counters": counters, "file": path.name, "pid": pid}
    if not results:
        raise ValueError(f"No measured benchmarks in {directory}")
    return results


def install_valgrind(source: Path) -> Path:
    subprocess.run(["sudo", "mkdir", "-p", str(VALGRIND_PREFIX.parent)], check=True)
    subprocess.run(["sudo", "rm", "-rf", str(VALGRIND_PREFIX)], check=True)
    subprocess.run(["sudo", "cp", "-a", str(source), str(VALGRIND_PREFIX)], check=True)
    tools = list(VALGRIND_PREFIX.rglob("callgrind-amd64-linux"))
    if len(tools) != 1:
        raise ValueError(f"Expected exactly one installed Callgrind executable, found {tools}")
    return tools[0].parent


def machine_identity() -> dict:
    cpu = json.loads(output_of(["lscpu", "--json"]))
    product_uuid = Path("/sys/class/dmi/id/product_uuid")
    return {
        "cpu_model": next(row["data"] for row in cpu["lscpu"] if row["field"] == "Model name:"),
        "cpu": cpu,
        "kernel": platform.release(),
        "glibc": output_of(["getconf", "GNU_LIBC_VERSION"]),
        "os_release": platform.freedesktop_os_release(),
        "machine_id": hashlib.sha256(output_of(["sudo", "cat", str(product_uuid)]).encode()).hexdigest()
        if product_uuid.exists()
        else None,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "hostname": platform.node() or None,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "image_os": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
    }


def expand_lab_dir(value: str, lab: Path) -> str:
    return value.replace("$VARIANCE_LAB_DIR", str(lab))


def write_run_script(path: Path, targets: list[Path], candidate: dict, lab: Path) -> None:
    parts = [expand_lab_dir(str(part), lab) for part in candidate.get("wrapper", [])]
    for part in parts:
        if part.startswith("LD_PRELOAD="):
            for library in part.removeprefix("LD_PRELOAD=").replace(":", " ").split():
                if not Path(library).is_file():
                    raise ValueError(f"LD_PRELOAD object does not exist on this runner: {library}")
    wrapper = " ".join(shlex.quote(part) for part in parts)
    env_mode = candidate.get("env_mode", "inherit")
    if env_mode not in ("inherit", "fixed"):
        raise ValueError(f"unknown env_mode {env_mode!r}")
    exact = " ".join(shlex.quote(name) for name in FIXED_ENV_EXACT)
    prefixes = "|".join(f"{prefix}*" for prefix in FIXED_ENV_PREFIXES)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "run_target() {",
        f'  if [ "{env_mode}" = fixed ]; then',
        "    local -a keep=()",
        "    local name entry",
        "    while IFS= read -r -d '' entry; do",
        '      name="${entry%%=*}"',
        '      case "$name" in',
        f"        {' | '.join(exact.split())} | {prefixes}) keep+=(\"$entry\");;",
        "      esac",
        "    done < <(env -0 | sort -z)",
        f'    env -i "${{keep[@]}}" {wrapper} "$@"',
        "  else",
        f'    {wrapper} "$@"',
        "  fi",
        "}",
    ]
    lines += [f"run_target {shlex.quote(str(target))}" for target in targets]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def capture(args) -> None:
    artifacts = args.artifacts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(args.candidate.read_text())
    lab = args.lab.resolve()
    workload = Path(os.environ.get("VARIANCE_LAB_WORKLOAD", Path.cwd() / "workload")).resolve()
    runner = Path(shutil.which("codspeed") or "codspeed").resolve()
    runner_version = output_of([str(runner), "--version"])
    if runner_version != RUNNER_VERSION:
        raise ValueError(f"Unexpected runner: {runner_version}")

    sample = output / "sample"
    sample.mkdir()
    profiles = sample / "profiles"
    profiles.mkdir()
    manifest = {
        "schema_version": 2,
        "experiment": "variance-lab",
        "job": args.job,
        "candidate": candidate,
        "runner_version": runner_version,
        "runner_sha256": digest(runner),
        "started_at": time.time(),
        "workflow_url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}",
        "workflow_sha": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        "status": "running",
        **machine_identity(),
    }
    save(sample / "manifest.json", manifest)
    started = time.monotonic()
    try:
        restore_workload(artifacts, workload)
        build_metadata = json.loads((artifacts / "binaries" / BINARIES_NAME / "build.json").read_text())
        if build_metadata["workload"] != str(workload):
            raise ValueError("Runtime workload path differs from the compiled artifact path")
        binaries = artifacts / "binaries" / BINARIES_NAME / "uv-bench"
        if hash_tree(binaries) != build_metadata["sha256"]:
            raise ValueError("Benchmark artifact checksums do not match build provenance")
        destination = workload / "target" / "codspeed" / "analysis" / "uv-bench"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(binaries, destination)
        available = {path.name: path for path in destination.iterdir() if path.is_file() and os.access(path, os.X_OK)}
        missing = [name for name in candidate["targets"] if name not in available]
        if missing:
            raise ValueError(f"Targets {missing} not in built executables {sorted(available)}")
        targets = [available[name] for name in candidate["targets"]]

        valgrind_metadata = json.loads((artifacts / "valgrind" / f"{VALGRIND_NAME}.json").read_text())
        valgrind_source = artifacts / "valgrind" / VALGRIND_NAME
        if hash_tree(valgrind_source) != valgrind_metadata["sha256"]:
            raise ValueError("Valgrind artifact checksums do not match build provenance")
        valgrind_lib = install_valgrind(valgrind_source)
        valgrind_hashes = hash_tree(VALGRIND_PREFIX)

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
            VARIANCE_LAB_DIR=str(lab),
        )
        environment.update({key: str(value) for key, value in candidate.get("env", {}).items()})
        setup_log = []
        for command in candidate.get("setup_commands", []):
            setup_log.append({"command": command})
            completed = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", command],
                cwd=workload / "crates" / "uv-bench",
                env=environment,
                text=True,
                capture_output=True,
            )
            setup_log[-1].update(returncode=completed.returncode, stdout=completed.stdout[-4000:], stderr=completed.stderr[-4000:])
            if completed.returncode != 0:
                raise ValueError(f"setup command failed ({completed.returncode}): {command}\n{completed.stderr[-2000:]}")
        manifest["setup_commands"] = setup_log

        run_script = workload / "run-lab-suite.sh"
        write_run_script(run_script, targets, candidate, lab)
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
            *[str(part) for part in candidate.get("runner_args", [])],
            "--",
            f"bash {shlex.quote(str(run_script))}",
        ]
        manifest.update(
            source_sha=build_metadata["source_sha"],
            rust_sha=build_metadata["rust_sha"],
            valgrind_sha=valgrind_metadata["source_sha"],
            valgrind_version=output_of([str(VALGRIND_PREFIX / "bin" / "valgrind"), "--version"], env=environment),
            binary_hashes=build_metadata["sha256"],
            valgrind_hashes=valgrind_hashes,
            targets=[path.name for path in targets],
            command=command,
            run_script=run_script.read_text(),
            working_directory=str(workload / "crates" / "uv-bench"),
            environment={key: environment[key] for key in sorted(environment) if not key.startswith(("ACTIONS_", "GITHUB_TOKEN"))},
        )
        save(sample / "manifest.json", manifest)
        measured_start = time.monotonic()
        with (sample / "runner.log").open("w") as log:
            subprocess.run(command, cwd=workload / "crates" / "uv-bench", env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["measurement_duration_seconds"] = time.monotonic() - measured_start
        results = summarize(profiles)
        save(sample / "results.json", results)
        required = candidate.get("required_benchmarks", [])
        absent = [name for name in required if not any(uri.endswith("::" + name) for uri in results)]
        if absent:
            raise ValueError(f"Required benchmarks missing from profiles: {absent}")
        # Candidates that inject code (LD_PRELOAD) must prove it reached the benchmark
        # processes: callgrind names every object that executed collected instructions.
        # Presence per benchmark is recorded; absence everywhere fails the run.
        for needle in candidate.get("required_profile_substrings", []):
            missing_in = []
            for name in required:
                _, result = next((u, r) for u, r in results.items() if u.endswith("::" + name))
                if needle not in (profiles / result["file"]).read_text(errors="replace"):
                    missing_in.append(name)
            manifest.setdefault("profile_substring_missing", {})[needle] = missing_in
            if len(missing_in) == len(required):
                raise ValueError(f"{needle!r} not present in any required benchmark profile")
        if hash_tree(VALGRIND_PREFIX) != valgrind_hashes:
            raise ValueError("Valgrind installation changed during measurement")
        manifest.update(status="success", benchmark_count=len(results), measured_process_count=len({r["pid"] for r in results.values()}))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        manifest.update(status="failed", error=str(error))
        print(f"FAILED: {error}", flush=True)
    finally:
        manifest["duration_seconds"] = time.monotonic() - started
        save(sample / "manifest.json", manifest)
        subprocess.run(["tar", "--zstd", "-cf", str(sample / "profiles.tar.zst"), "-C", str(sample), "profiles"], check=True)
        shutil.rmtree(profiles)
    print(json.dumps({key: manifest.get(key) for key in ("job", "status", "benchmark_count", "measured_process_count", "duration_seconds", "error")}), flush=True)
    if manifest["status"] != "success":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
