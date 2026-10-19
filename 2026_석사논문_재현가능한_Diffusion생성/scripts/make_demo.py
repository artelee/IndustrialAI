#!/usr/bin/env python
"""Create a synthetic demo workspace (noise images + stand-in IPG) for trying the tools on CPU.

Example:
    python scripts/make_demo.py outputs/demo
    python scripts/run_batch.py --config outputs/demo/sets_demo.yaml --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reprogen.demo import make_demo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="outputs/demo")
    ap.add_argument("--n-refs", type=int, default=5)
    args = ap.parse_args(argv)
    print(f"demo config: {make_demo(args.root, n_refs=args.n_refs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
