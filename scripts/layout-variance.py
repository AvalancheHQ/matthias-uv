#!/usr/bin/env python3
"""Sweep process-environment size without rebuilding or changing the uv workload."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / ".agents/syn-variance"
PADDING_BYTES = (0, 16, 32, 48, 64, 256, 4096, 65536)
STAGING_UPLOAD = "https://api.staging.preview.codspeed.io/upload"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--padding-bytes", nargs="+", type=int, default=PADDING_BYTES)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--target", choices=("pep508", "wheel"), default="pep508")
    parser.add_argument("--upload", choices=("staging", "none"), default="staging")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = SOURCE / "tools/codspeed-runner-x86_64-unknown-linux-musl/codspeed"
    if args.target == "pep508":
        binary = SOURCE / "benchmark-A"
        manifest = json.loads((SOURCE / "build-A.json").read_text())
        working_directory = ROOT
        expected_cases = 25
    else:
        binary = ROOT / "target/codspeed/analysis/uv-bench/uv"
        manifest = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_binary": str(binary),
        }
        working_directory = ROOT / "crates/uv-bench"
        expected_cases = 18
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (expected := manifest.get("sha256")) and digest != expected:
        raise ValueError("Saved benchmark hash differs from its build manifest")
    manifest["sha256"] = digest
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location(
        "capture", ROOT / "scripts/ci-variance.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the profile parser")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    focus = (
        tuple(name for name in capture.FOCUS if name != "install_wheel_many_files")
        if args.target == "pep508"
        else ("install_wheel_many_files",)
    )
    capture.__dict__["FOCUS"] = focus
    environment = dict(
        os.environ,
        CODSPEED_PROFILE="staging",
        CODSPEED_API_URL="https://gql.staging.preview.codspeed.io/",
        CODSPEED_RUNNER_MODE="simulation",
        CODSPEED_CARGO_WORKSPACE_ROOT=str(ROOT),
    )
    upload_arguments = (
        ["--skip-upload"] if args.upload == "none" else ["--upload-url", STAGING_UPLOAD]
    )
    observations = {name: [] for name in focus}
    for repetition, size in itertools.product(
        range(args.repetitions), args.padding_bytes
    ):
        sample = output / f"r{repetition:02}-pad-{size:05}"
        sample.mkdir()
        (sample / "profiles").mkdir()
        environment["UV_VARIANCE_PADDING"] = "x" * size
        (sample / "control.json").write_text(
            json.dumps({"environment_padding_bytes": size, "binary_sha256": digest})
            + "\n"
        )
        with (sample / "runner.log").open("w") as log:
            subprocess.run(
                [
                    str(runner),
                    "run",
                    "--mode",
                    "simulation",
                    "--repository",
                    "AvalancheHQ/matthias-uv",
                    *upload_arguments,
                    "--profile-folder",
                    str(sample / "profiles"),
                    "--show-full-output",
                    "--",
                    str(binary),
                ],
                cwd=working_directory,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        results = capture.summarize(sample / "profiles", sample)
        if len(results) != expected_cases:
            raise ValueError(f"Expected all {expected_cases} cases, got {len(results)}")
        for uri, result in results.items():
            name = uri.split("::")[-1]
            if name in observations:
                observations[name].append((size, result))
                print(
                    json.dumps({"padding_bytes": size, "uri": uri, **result}),
                    flush=True,
                )

    within_padding_ranges = []
    for name, rows in observations.items():
        if args.target == "pep508":
            for counter in ("Ir", "Dr", "Dw", "Ct", "Cl"):
                if len({result["counters"][counter] for _, result in rows}) != 1:
                    raise ValueError(f"Work changed for {name}: {counter}")
        values = [result["seconds"] for _, result in rows]
        metric = re.sub(r"\W+", "_", name).strip("_")
        print(
            f"METRIC range_{metric}_percent={100 * (max(values) / min(values) - 1):.9f}"
        )
        for size in args.padding_bytes:
            values = [result["seconds"] for padding, result in rows if padding == size]
            within_padding_ranges.append(100 * (max(values) / min(values) - 1))
    print(f"METRIC max_within_padding_range_percent={max(within_padding_ranges):.9f}")


if __name__ == "__main__":
    main()
