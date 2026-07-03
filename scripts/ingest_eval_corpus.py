#!/usr/bin/env python3
"""Ingest the multi-document eval corpora into the local FAISS index.

The eval dataset ``eval/dataset.multidoc.json`` expects these canonical names:

  * ``ec2-ug.pdf`` — Amazon EC2 User Guide
  * ``ec2-types.pdf`` — Amazon EC2 Instance Types guide

Example (rename on ingest so eval cases match):

  python scripts/ingest_eval_corpus.py \\
    --file ~/Downloads/ec2-ug.pdf \\
    --as ec2-ug.pdf

  python scripts/ingest_eval_corpus.py \\
    --file ~/Downloads/"Amazon EC2 Instance Types.pdf" \\
    --as ec2-types.pdf

Re-ingesting the same ``--as`` filename replaces the previous document with
that name in the registry/index.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import get_orchestrator

DEFAULT_CORPUS = (
    ("ec2-ug.pdf", "Amazon EC2 User Guide"),
    ("ec2-types.pdf", "Amazon EC2 Instance Types"),
)


def _delete_by_filename(orch, filename: str) -> None:
    for record in orch.registry.list():
        if record.filename == filename:
            orch.delete_document(record.id)
            print(f"Removed existing {filename} ({record.id})", file=sys.stderr)


def ingest_one(orch, source: Path, filename: str) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    _delete_by_filename(orch, filename)
    result = orch.ingest(
        source,
        filename=filename,
        content_type="application/pdf",
    )
    record = result.record
    print(
        f"Ingested {filename}: id={record.id} chunks={record.num_chunks} "
        f"chars={record.num_chars}",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest multi-doc eval PDFs.")
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        default=[],
        help="PDF path to ingest (repeatable).",
    )
    parser.add_argument(
        "--as",
        dest="as_name",
        action="append",
        default=[],
        help="Canonical filename for the preceding --file (eval dataset key).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "eval" / "corpus",
        help="Directory containing ec2-ug.pdf and ec2-types.pdf.",
    )
    parser.add_argument(
        "--list-expected",
        action="store_true",
        help="Print expected corpus filenames and exit.",
    )
    args = parser.parse_args()

    if args.list_expected:
        for filename, label in DEFAULT_CORPUS:
            print(f"  {filename:28s}  {label}")
        return 0

    orch = get_orchestrator()
    pairs: list[tuple[Path, str]] = []

    if args.file:
        if len(args.file) != len(args.as_name):
            print("Provide one --as name for each --file.", file=sys.stderr)
            return 1
        pairs.extend(zip(args.file, args.as_name))
    else:
        for filename, _label in DEFAULT_CORPUS:
            path = args.corpus_dir / filename
            if path.exists():
                pairs.append((path, filename))

    if not pairs:
        print(
            "No PDFs found. Place files in eval/corpus/ or pass --file/--as.",
            file=sys.stderr,
        )
        print("Expected:", file=sys.stderr)
        for filename, label in DEFAULT_CORPUS:
            print(f"  {filename} ({label})", file=sys.stderr)
        return 1

    for source, filename in pairs:
        ingest_one(orch, source, filename)

    print(
        f"Done. {orch.registry.count} document(s), {orch.store.num_chunks} chunk(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
