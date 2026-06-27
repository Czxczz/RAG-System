#!/usr/bin/env python3
"""Remove exact-duplicate chunks from the FAISS index (no re-upload needed).

The EC2 user guide contains repeated boilerplate headers/footers across pages.
This compacts the index by keeping one copy of each identical text body.

Usage:
    python scripts/compact_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import get_orchestrator


def main() -> int:
    orch = get_orchestrator()
    before = orch.store.num_chunks
    removed = orch.store.compact_exact_duplicates()
    after = orch.store.num_chunks
    print(f"Chunks before: {before}")
    print(f"Duplicates removed: {removed}")
    print(f"Chunks after: {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
