#!/usr/bin/env python3
"""Compare the custom orchestrator vs the LangChain (LCEL) RAG path.

Both engines share the same FAISS index, embeddings, reranker, and LLM router,
so this isolates the *composition* layer. Useful for verifying parity while
adopting LangChain.

Usage:
    python scripts/compare_engines.py --mode gemini --top-k 3
    python scripts/compare_engines.py --query "How do I require IMDSv2?" --mode ollama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chains.langchain_rag import build_langchain_rag
from app.dependencies import get_orchestrator
from app.eval.schemas import EvalDataset


def _summarize(label: str, answer: str, provider: str, grounded: bool, hits) -> None:
    print(f"\n=== {label} ===")
    print(f"provider={provider} grounded={grounded} chunks={len(hits)}")
    citations = sorted({h.chunk.id for h in hits})
    print(f"cited chunks: {citations}")
    print(answer[:600] + ("…" if len(answer) > 600 else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Single ad-hoc query to compare.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "dataset.ec2.json",
        help="Eval dataset to pull queries from (when --query is omitted).",
    )
    parser.add_argument("--case-id", action="append", help="Limit to these case ids.")
    parser.add_argument("--mode", default="auto", help="LLM mode.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=3, help="Max dataset cases.")
    parser.add_argument(
        "--rewrite-llm",
        action="store_true",
        help=(
            "Allow LLM query rewriting during comparison. Off by default so "
            "retrieval is deterministic and isolates the composition layer "
            "(LLM rewrites can vary between calls when the provider is flaky)."
        ),
    )
    args = parser.parse_args()

    orch = get_orchestrator()
    # Both engines share this query engine; disabling LLM rewrite keeps retrieval
    # deterministic so any difference reflects the composition layer, not
    # provider randomness (429s falling back to heuristics mid-comparison).
    if not args.rewrite_llm:
        orch.settings.query_rewrite_use_llm = False
        print("LLM query rewriting disabled for deterministic comparison.")

    lc = build_langchain_rag(orch)
    top_k = args.top_k or orch.settings.top_k

    if args.query:
        queries = [("ad-hoc", args.query)]
    else:
        ds = EvalDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
        cases = ds.cases
        if args.case_id:
            wanted = set(args.case_id)
            cases = [c for c in cases if c.id in wanted]
        queries = [(c.id, c.query) for c in cases[: args.limit]]

    agreements = 0
    for case_id, query in queries:
        print("\n" + "#" * 70)
        print(f"# {case_id}: {query}")
        custom = orch.answer(query=query, mode=args.mode, top_k=top_k)
        lc_result = lc.answer(query=query, mode=args.mode, top_k=top_k)

        _summarize("CUSTOM orchestrator", custom.answer, custom.provider, custom.grounded, custom.hits)
        _summarize("LANGCHAIN (LCEL)", lc_result.answer, lc_result.provider, lc_result.grounded, lc_result.hits)

        same_chunks = {h.chunk.id for h in custom.hits} == {h.chunk.id for h in lc_result.hits}
        print(f"\n→ same retrieved chunks: {same_chunks}")
        agreements += int(same_chunks)

    print("\n" + "=" * 70)
    print(f"Retrieval agreement: {agreements}/{len(queries)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
