#!/usr/bin/env python3
"""Build the pinned public CIMFlow simulator without system installation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_foundry.cimflow import REVISION, REPOSITORY  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"external/cimflow-simulator")
    parser.add_argument("--record", type=Path, default=ROOT/"runs/cimflow-build")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("jobs must be 1..4")
    args.source, args.record = args.source.resolve(), args.record.resolve()
    if args.record.exists():
        parser.error("record directory already exists; use a new directory")
    args.record.mkdir(parents=True)
    commands = []

    def run(command, log):
        commands.append(command)
        with (args.record/log).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
        if result.returncode:
            raise RuntimeError(f"command failed; see {args.record/log}")

    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        run(["git","clone",REPOSITORY,str(args.source)],"clone.log")
        run(["git","-C",str(args.source),"checkout","--detach",REVISION],"checkout.log")
        run(["git","-C",str(args.source),"submodule","update","--init","--recursive"],"submodules.log")
    head = subprocess.check_output(["git","-C",str(args.source),"rev-parse","HEAD"],text=True).strip()
    dirty = subprocess.check_output(["git","-C",str(args.source),"diff","--name-only","HEAD"],text=True).strip()
    if head != REVISION or dirty:
        raise ValueError("source must be an unmodified pinned checkout")
    submodules = subprocess.check_output(["git","-C",str(args.source),"submodule","status","--recursive"],text=True)
    if any(line and line[0] != " " for line in submodules.splitlines()):
        raise ValueError("submodule checkout does not match committed gitlinks")
    build = args.source/"build"
    run(["cmake","-S",str(args.source),"-B",str(build),"-G","Ninja","-DCMAKE_BUILD_TYPE=Release"],"configure.log")
    run(["cmake","--build",str(build),"--target","cim-simulator-bin","CoreTest","-j",str(args.jobs)],"build.log")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    tracked = subprocess.check_output(["git","-C",str(args.source),"ls-files","-z"],text=True).split("\0")
    files = {name:digest(args.source/name) for name in tracked if name and (args.source/name).is_file()}
    binary_paths = [build/"cim-simulator",build/"core-test"] + [
        path for path in build.rglob("*.so.*") if not path.is_symlink()]
    binaries = {str(path.relative_to(build)):digest(path) for path in binary_paths}
    record = {"schema":1,"repository":REPOSITORY,"revision":head,"submodules":submodules.splitlines(),
              "source_files":files,"binaries":binaries,"commands":commands,
              "compiler":subprocess.check_output(["c++","--version"],text=True),
              "cmake":subprocess.check_output(["cmake","--version"],text=True),
              "cmake_cache_sha256":digest(build/"CMakeCache.txt"),
              "dynamic_dependencies":subprocess.check_output(["ldd",str(build/"cim-simulator")],text=True),
              "hardware_calibrated":False,"native_numerical":False}
    (args.record/"provenance.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(build/"cim-simulator")


if __name__ == "__main__":
    main()
