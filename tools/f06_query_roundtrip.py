#!/usr/bin/env python3
"""Run one solver case through the MYSTRAN-compatible F06 query parser."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--query", required=True, type=Path)
    parser.add_argument("--bdf", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.query.parent))
    from f06_query import F06Query

    with tempfile.TemporaryDirectory(prefix="vibestran_f06_query_") as tmp:
        work = Path(tmp)
        deck = work / args.bdf.name
        shutil.copy2(args.bdf, deck)
        result = subprocess.run(
            [str(args.solver), deck.name],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            return result.returncode

        f06_path = deck.with_suffix(".F06")
        if not f06_path.is_file():
            print(f"missing uppercase F06 output: {f06_path}", file=sys.stderr)
            return 1

        parsed = F06Query(str(f06_path)).parsed_f06
        subcase = parsed["SC"]["1"]
        required = {"DISPLACEMENTS", "SHELLSTRESSES"}
        missing = required.difference(subcase)
        if missing:
            print(f"missing parsed result blocks: {sorted(missing)}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
