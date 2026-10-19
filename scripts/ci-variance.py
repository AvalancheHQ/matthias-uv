#!/usr/bin/env python3
"""Capture repeated simulation runs and forward their uploads to staging."""

import argparse
import base64
import hashlib
import http.client
import json
import math
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

FOCUS = (
    "imbibe_marker_conflict_dependent[32]",
    "imbibe_marker_environment_only[32]",
    "imbibe_marker_conflict_dependent[16]",
    "imbibe_marker_environment_only[16]",
    "install_wheel_many_files",
    "imbibe_and_format_marker_environment_only[32]",
)


def summarize(directory, output):
    results = {}
    for path in sorted(directory.glob("*.out")):
        name = None
        events = []
        with path.open() as stream:
            for line in stream:
                if line.startswith("part:"):
                    name = None
                elif line.startswith("desc: Trigger: Client Request: "):
                    name = line.removeprefix("desc: Trigger: Client Request: ").strip()
                elif line.startswith("events:"):
                    events = line.split()[1:]
                elif line.startswith("totals:") and name and "::" in name:
                    values = [int(value) for value in line.split()[1:]]
                    counters = dict(
                        zip(events, values + [0] * (len(events) - len(values)), strict=True)
                    )
                    if name in results:
                        raise ValueError(f"Duplicate measured benchmark: {name}")
                    cycles = (
                        math.sqrt(counters["Ct"] * counters["Cl"]) / 100
                        + 5
                        * (
                            counters["Dr"]
                            + counters["Dw"]
                            - counters["D1mr"]
                            - counters["D1mw"]
                        )
                        + 40
                        * (
                            counters["D1mr"]
                            + counters["D1mw"]
                            - counters["DLmr"]
                            - counters["DLmw"]
                        )
                        + 200 * (counters["DLmr"] + counters["DLmw"] + counters["ILmr"])
                    )
                    results[name] = {
                        "seconds": cycles / 3_600_000_000,
                        "counters": counters,
                        "file": path.name,
                    }
    for benchmark in FOCUS:
        matches = [name for name in results if name.endswith("::" + benchmark)]
        if len(matches) != 1:
            raise ValueError(f"Expected one measurement for {benchmark}, got {matches}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def capture(output, repetitions, command):
    class Capture(BaseHTTPRequestHandler):
        def do_POST(self):
            sample = output / self.path.split("/")[1]
            metadata = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            (sample / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            request = urllib.request.Request(
                "https://api.staging.preview.codspeed.io/upload",
                data=json.dumps(metadata).encode(),
                headers={"Content-Type": "application/json"},
            )
            if token := self.headers.get("Authorization"):
                request.add_header("Authorization", token)
            with urllib.request.urlopen(request, timeout=120) as response:
                upload = json.load(response)
            destinations[sample.name] = upload["uploadUrl"]
            (sample / "staging-run.txt").write_text(upload["runId"] + "\n")
            upload["uploadUrl"] = (
                f"http://127.0.0.1:{server.server_port}/{sample.name}/profile"
            )
            body = json.dumps(upload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self):
            sample = output / self.path.split("/")[1]
            remaining = int(self.headers["Content-Length"])
            digest = hashlib.md5()
            with (sample / "profile.tar").open("wb") as stream:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("Incomplete profile upload")
                    stream.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            metadata = json.loads((sample / "metadata.json").read_text())
            if base64.b64encode(digest.digest()).decode() != metadata["profileMd5"]:
                self.send_error(400, "Profile checksum mismatch")
                return
            destination = urllib.parse.urlsplit(destinations.pop(sample.name))
            if destination.scheme != "https":
                raise ValueError("Expected HTTPS upload destination")
            connection = http.client.HTTPSConnection(destination.netloc, timeout=120)
            try:
                connection.putrequest("PUT", destination.path + "?" + destination.query)
                connection.putheader("Content-Type", "application/x-tar")
                connection.putheader(
                    "Content-Length", str((sample / "profile.tar").stat().st_size)
                )
                connection.endheaders()
                with (sample / "profile.tar").open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        connection.send(chunk)
                response = connection.getresponse()
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"Staging profile upload failed: {response.status}"
                    )
                response.read()
            finally:
                connection.close()
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    output.mkdir(parents=True, exist_ok=True)
    destinations = {}
    server = HTTPServer(("127.0.0.1", 0), Capture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for repetition in range(repetitions):
            sample = output / f"sample-{repetition:02}"
            sample.mkdir()
            profile = sample / "profiles"
            with (sample / "runner.log").open("w") as log:
                subprocess.run(
                    [
                        "codspeed",
                        "run",
                        "--mode",
                        "simulation",
                        "--profile-folder",
                        str(profile),
                        "--show-full-output",
                        "--upload-url",
                        f"http://127.0.0.1:{server.server_port}/{sample.name}/upload",
                        "--",
                        command,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not (sample / "profile.tar").is_file():
                raise ValueError("Runner did not upload its measured profile")
            results = summarize(profile, sample)
            for name, result in results.items():
                if name.split("::")[-1] in FOCUS:
                    print(
                        json.dumps({"sample": repetition, "uri": name, **result}),
                        flush=True,
                    )
            # The checksum-verified archive retains the complete raw profile.
            shutil.rmtree(profile)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--command", default="uv run --only-dev cargo codspeed run")
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.output)
    else:
        capture(args.output.resolve(), args.repetitions, args.command)
