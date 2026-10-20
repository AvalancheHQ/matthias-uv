#!/usr/bin/env python3
"""Run one planned main-history measurement (commit x configuration) per macro job.

Consumes the per-commit bundle produced by ``main-history-build.py`` plus the
two frozen Valgrind artifacts. Nothing is rebuilt: sources, binaries and
Valgrind trees are verified against the plan pins and the build metadata, then
the suite is measured with the established runner command.

Configurations: A = old Valgrind + old integration binaries, B = new Valgrind +
the same old binaries, C = new Valgrind + new integration binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ARTIFACT_REVISION = "base"
BATCH = "main-history"
COUNTER_TERMS = ("Ct", "Cl")
VALGRIND_FOR_CONFIG = {"A": "old", "B": "new", "C": "new"}
INTEGRATION_FOR_CONFIG = {"A": "old", "B": "old", "C": "new"}
ENVIRONMENT_KEYS = (
    "PATH",
    "VALGRIND_LIB",
    "CODSPEED_CARGO_WORKSPACE_ROOT",
    "CODSPEED_PROFILER_ENABLED",
    "CODSPEED_RUNNER_MODE",
    "CODSPEED_LOG",
)
ISOLATION_VARIABLES = ("CODSPEED_PROCESS_ISOLATION", "CODSPEED_ISOLATED_BENCHMARK")


def load_legacy():
    """Reuse the pure helpers of the established capture script."""
    spec = importlib.util.spec_from_file_location(
        "stack_isolation_capture", SCRIPTS / "stack-isolation-capture.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load stack-isolation-capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_legacy()
save, digest, hashes, summarize, output_of, install_valgrind = (
    legacy.save,
    legacy.digest,
    legacy.hashes,
    legacy.summarize,
    legacy.output_of,
    legacy.install_valgrind,
)
VALGRIND_PREFIX: Path = legacy.VALGRIND_PREFIX


def strict_json(text: str):
    def pairs(items):
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"duplicate JSON keys: {sorted({key for key in keys if keys.count(key) > 1})}"
            )
        return dict(items)

    return json.loads(text, object_pairs_hook=pairs)


def load_json(path: Path):
    return strict_json(path.read_text())


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def extract(archive: Path, destination: Path) -> None:
    if not archive.is_file():
        raise ValueError(f"missing archive {archive}")
    subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            f"zstd -dc {shlex.quote(str(archive))} | tar -xf - -C {shlex.quote(str(destination))}",
        ],
        check=True,
    )


def plan_job(plan: dict, job: int) -> tuple[dict, dict]:
    entries = [entry for entry in plan["jobs"] if entry["job"] == job]
    if len(entries) != 1:
        raise ValueError(f"job {job} must appear exactly once in the plan")
    entry = entries[0]
    commits = [
        commit for commit in plan["commits"] if commit["id"] == entry["revision"]
    ]
    if len(commits) != 1:
        raise ValueError(f"job {job}: revision {entry['revision']} not a plan commit")
    commit = commits[0]
    if (commit["index"], commit["sha"]) != (entry["commit_index"], entry["source_sha"]):
        raise ValueError(f"job {job}: plan entry disagrees with commit {commit['id']}")
    if entry["config"] not in plan["configs"]:
        raise ValueError(f"job {job}: unknown config {entry['config']}")
    return entry, commit


def check_build_metadata(
    metadata: dict, plan: dict, plan_hash: str, commit: dict
) -> None:
    pins = plan["pins"]
    expected = {
        "schema_version": 1,
        "experiment": plan["experiment"],
        "plan_sha256": plan_hash,
        "revision": commit["id"],
        "commit_index": commit["index"],
        "source_sha": commit["sha"],
        "canonical_workload": plan["canonical_workload"],
        "rust_toolchain": pins["rust_toolchain"],
        "old_cargo_codspeed_version": pins["old_cargo_codspeed_version"],
        "old_codspeed_crate_version": pins["old_codspeed_crate_version"],
        "codspeed_rust_revision": pins["codspeed_rust_revision"],
    }
    for key, value in expected.items():
        expect(
            metadata.get(key) == value,
            f"build metadata {key}={metadata.get(key)!r}, expected {value!r}",
        )
    expect(
        str(metadata.get("rustc_version", "")).split()[:2]
        == ["rustc", pins["rust_toolchain"]],
        f"build metadata rustc {metadata.get('rustc_version')!r} is not {pins['rust_toolchain']}",
    )
    for key in ("input_fingerprints", "benchmark_files"):
        expect(
            isinstance(metadata.get(key), dict) and metadata[key],
            f"build metadata {key} missing or empty",
        )
    binaries = metadata.get("binaries", {})
    for integration in ("old", "new"):
        entry = binaries.get(integration, {})
        expect(
            bool(entry.get("targets"))
            and sorted(entry["targets"]) == sorted(entry.get("sha256", {})),
            f"build metadata {integration} targets disagree with hashes",
        )
    expect(
        binaries["old"]["targets"] == binaries["new"]["targets"],
        "old/new target sets differ",
    )
    expect(
        isinstance(metadata.get("workflow_run_id"), str)
        and metadata["workflow_run_id"],
        "build workflow_run_id missing",
    )


def verify_valgrind(artifacts: Path, plan: dict, name: str) -> Path:
    pins = plan["pins"]
    directory = artifacts / "valgrind" / name
    recorded = load_json(artifacts / "valgrind" / f"{name}.json")
    expect(
        recorded.get("source_sha") == pins["valgrind_revisions"][name],
        f"valgrind {name}: source_sha differs from pin",
    )
    expect(
        recorded.get("prefix") == str(VALGRIND_PREFIX),
        f"valgrind {name}: prefix {recorded.get('prefix')} differs",
    )
    actual = hashes(directory)
    expect(
        actual == pins["valgrind_hashes"][name],
        f"valgrind {name}: tree differs from the plan's frozen hashes",
    )
    expect(
        actual == recorded.get("sha256"),
        f"valgrind {name}: tree differs from its metadata",
    )
    return directory


def prepare_workload(
    artifacts: Path, workload: Path, commit: dict, metadata: dict, integration: str
):
    sources = artifacts / "sources" / ARTIFACT_REVISION
    extract(artifacts / "sources" / f"{ARTIFACT_REVISION}.tar.zst", artifacts)
    source = load_json(sources / "experiment-source.json")
    expect(
        source["source_sha"] == commit["sha"],
        "source snapshot SHA differs from the plan commit",
    )
    expect(
        source["workload"] == str(workload),
        "source snapshot was taken at a different workload path",
    )

    key = f"{ARTIFACT_REVISION}-{integration}"
    extract(artifacts / "binaries" / f"{key}.tar.zst", artifacts)
    build = load_json(artifacts / "binaries" / key / "build.json")
    expect(build["source_sha"] == commit["sha"], f"{key}: build source SHA differs")
    expect(
        build["workload"] == str(workload),
        f"{key}: compiled workload path differs from runtime path",
    )
    binaries = artifacts / "binaries" / key / "uv-bench"
    actual = hashes(binaries)
    expect(actual == build["sha256"], f"{key}: binaries differ from build.json")
    expect(
        actual == metadata["binaries"][integration]["sha256"],
        f"{key}: binaries differ from main-history-build.json",
    )

    if workload.exists():
        shutil.rmtree(workload)
    workload.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sources, workload, symlinks=True)
    head = output_of(["git", "-C", str(workload), "rev-parse", "HEAD"])
    expect(head == commit["sha"], f"restored workload HEAD {head} != {commit['sha']}")
    destination = workload / "target" / "codspeed" / "analysis" / "uv-bench"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(binaries, destination)
    targets = sorted(
        path
        for path in destination.iterdir()
        if path.is_file() and os.access(path, os.X_OK)
    )
    expect(
        [path.name for path in targets]
        == sorted(metadata["binaries"][integration]["targets"]),
        f"{key}: executable targets differ from build metadata",
    )
    expect(hashes(destination) == actual, f"{key}: copied binaries differ")
    return destination, targets


def validate_results(
    results: dict, config: str, expected_targets: list[str]
) -> set[str]:
    expect(bool(results), "no measured benchmarks")
    for uri, result in results.items():
        counters = result["counters"]
        expect(
            "::" in uri
            and result["seconds"] > 0
            and all(counters.get(term, 0) > 0 for term in COUNTER_TERMS),
            f"invalid measurement for {uri!r}: seconds={result['seconds']}, counters={counters}",
        )
        expect(
            all(isinstance(value, int) and value >= 0 for value in counters.values()),
            f"invalid counters for {uri!r}",
        )
    pids = {result["pid"] for result in results.values()}
    expected = len(results) if config == "C" else len(expected_targets)
    expect(
        len(pids) == expected,
        f"expected {expected} measuring processes for {config}, got {len(pids)}",
    )
    return pids


def capture(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = args.plan.resolve()
    plan = load_json(plan_path)
    plan_hash = digest(plan_path)
    entry, commit = plan_job(plan, args.job)
    config = entry["config"]
    revision = commit["id"]
    integration = INTEGRATION_FOR_CONFIG[config]
    valgrind = VALGRIND_FOR_CONFIG[config]
    pins = plan["pins"]
    config_index = plan["configs"].index(config)
    round_number, position = entry["round"], entry["order"]

    save(
        output / "schedule.json",
        [
            {
                "job": args.job,
                "round": round_number,
                "order": position,
                "cell": commit["index"] * len(plan["configs"]) + config_index,
                "config": config,
                "revision": revision,
                "commit_index": commit["index"],
                "source_sha": commit["sha"],
            }
        ],
    )
    sample = output / f"sample-{round_number:02}-{config}-{revision}"
    sample.mkdir()
    profiles = sample / "profiles"
    profiles.mkdir()
    manifest = {
        "schema_version": 1,
        "experiment": plan["experiment"],
        "batch": BATCH,
        "job": args.job,
        "round": round_number,
        "order": position,
        "cell": commit["index"] * len(plan["configs"]) + config_index,
        "preflight": False,
        "config": config,
        "revision": revision,
        "commit_index": commit["index"],
        "source_sha": commit["sha"],
        "plan_sha256": plan_hash,
        "rust_integration": pins["codspeed_rust_revision"]
        if integration == "new"
        else "locked-5.0.2",
        "valgrind_sha": pins["valgrind_revisions"][valgrind],
        "started_at": time.time(),
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
    failure = None
    try:
        artifacts = args.artifacts.resolve()
        workload = Path(plan["canonical_workload"])

        # Build provenance: the exact bytes are retained next to the samples.
        metadata_source = artifacts / "main-history-build.json"
        shutil.copyfile(metadata_source, output / "build-metadata.json")
        manifest["build_metadata_sha256"] = digest(output / "build-metadata.json")
        metadata = strict_json((output / "build-metadata.json").read_text())
        check_build_metadata(metadata, plan, plan_hash, commit)
        for relative in (
            f"sources/{ARTIFACT_REVISION}.tar.zst",
            f"binaries/{ARTIFACT_REVISION}-{integration}.tar.zst",
        ):
            expect(
                digest(artifacts / relative) == metadata["artifact_files"][relative],
                f"{relative}: archive differs from build metadata",
            )
        manifest["build_workflow_run_id"] = metadata["workflow_run_id"]
        expect(
            os.environ.get("GITHUB_RUN_ID") == metadata["workflow_run_id"],
            f"capture run {os.environ.get('GITHUB_RUN_ID')} differs from build run {metadata['workflow_run_id']}",
        )
        save(sample / "manifest.json", manifest)

        runner = Path(shutil.which("codspeed") or "codspeed").resolve()
        runner_version = output_of([str(runner), "--version"])
        expect(
            runner_version == f"codspeed-runner {pins['runner_version']}",
            f"unexpected runner {runner_version!r}",
        )
        runner_sha = digest(runner)
        expect(
            runner_sha == pins["runner_binary_sha256"],
            f"runner binary SHA {runner_sha} differs from the plan pin",
        )
        cpu = json.loads(output_of(["lscpu", "--json"]))
        product_uuid = Path("/sys/class/dmi/id/product_uuid")
        packages = output_of(
            ["dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"]
        )
        package_file = output / "package-versions.txt"
        package_file.write_text("\n".join(sorted(packages.splitlines())) + "\n")
        manifest.update(
            cpu_model=next(
                row["data"] for row in cpu["lscpu"] if row["field"] == "Model name:"
            ),
            cpu=cpu,
            kernel=platform.release(),
            glibc=output_of(["getconf", "GNU_LIBC_VERSION"]),
            os_release=platform.freedesktop_os_release(),
            package_versions_sha256=digest(package_file),
            runner_version=runner_version,
            runner_sha256=runner_sha,
            machine_id=hashlib.sha256(
                output_of(["sudo", "cat", str(product_uuid)]).encode()
            ).hexdigest()
            if product_uuid.exists()
            else None,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        )
        save(sample / "manifest.json", manifest)

        # Frozen Valgrind artifact: must match the plan's pinned tree before it is installed.
        extract(artifacts / "valgrind" / f"{valgrind}.tar.zst", artifacts)
        valgrind_source = verify_valgrind(artifacts, plan, valgrind)

        binary_dir, targets = prepare_workload(
            artifacts, workload, commit, metadata, integration
        )
        binary_hashes = hashes(binary_dir)
        valgrind_lib = install_valgrind(valgrind_source)
        valgrind_hashes = hashes(VALGRIND_PREFIX)
        expect(
            valgrind_hashes == pins["valgrind_hashes"][valgrind],
            "installed Valgrind differs from the plan's frozen hashes",
        )

        environment = dict(os.environ)
        for key in ISOLATION_VARIABLES:
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
            targets=[path.name for path in targets],
            command=command,
            working_directory=str(workload / "crates" / "uv-bench"),
            benchmark_commands=[[str(path)] for path in targets],
            environment={key: environment[key] for key in ENVIRONMENT_KEYS},
            isolation_env="unset",
            valgrind_version=output_of(
                [str(VALGRIND_PREFIX / "bin" / "valgrind"), "--version"]
            ),
        )
        save(sample / "manifest.json", manifest)

        measured_start = time.monotonic()
        with (sample / "runner.log").open("w") as runner_log:
            subprocess.run(
                command,
                cwd=workload / "crates" / "uv-bench",
                env=environment,
                stdout=runner_log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        manifest["measurement_duration_seconds"] = time.monotonic() - measured_start
        results = summarize(profiles)
        save(sample / "results.json", results)
        pids = validate_results(results, config, manifest["targets"])
        expect(
            hashes(VALGRIND_PREFIX) == valgrind_hashes,
            "Valgrind installation changed during measurement",
        )
        expect(
            hashes(binary_dir) == binary_hashes,
            "benchmark binaries changed during measurement",
        )
        manifest.update(
            status="success",
            benchmark_count=len(results),
            measured_process_count=len(pids),
        )
    except Exception as error:  # noqa: BLE001 - every failure must be recorded, then re-raised below
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        failure = f"{sample.name}: {type(error).__name__}: {error}"
        print(f"FAILED {failure}", flush=True)
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
    if failure:
        raise SystemExit(failure)


def job_number(value: str) -> int:
    try:
        job = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"job must be an integer, got {value!r}"
        ) from error
    if job < 0:
        raise argparse.ArgumentTypeError(f"job must be non-negative, got {job}")
    return job


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", type=job_number, required=True)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
