"""LangChain integration layer.

A parallel RAG path that *wraps* the existing, tuned components (query engine,
context grouping, LLM router, answer validation) as LangChain primitives. The
custom orchestrator in ``app.core.orchestrator`` remains the default; this layer
lets us learn / adopt LangChain (and later LangGraph) without forking the
retrieval logic that the EC2 eval was tuned against.
"""
from app.chains.langchain_rag import LangChainRAG, QueryEngineRetriever

__all__ = ["LangChainRAG", "QueryEngineRetriever"]
