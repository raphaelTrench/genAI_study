"""Adapters for retrieval implementations used by :class:`Evaluator`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

RankedDocuments: TypeAlias = Sequence[Document | str]
RetrievalResults: TypeAlias = RankedDocuments | Sequence[RankedDocuments]


class RetrievalStrategy(Protocol):
    """A retrieval strategy returning a ranking or multiple rankings per query."""

    def retrieve(self, query: str) -> RetrievalResults:
        """Return ranked documents for a query."""


class BatchRetrievalStrategy(RetrievalStrategy, Protocol):
    """Optional batching interface for efficient retrieval over query sets."""

    def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        max_concurrency: int,
    ) -> Sequence[RetrievalResults]:
        """Retrieve a batch of queries, preserving input order."""


class BaseRetrieverStrategy:
    """Adapt a standard LangChain retriever to the strategy interface."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: str) -> RankedDocuments:
        return self.retriever.invoke(query)

    def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        max_concurrency: int,
    ) -> Sequence[RetrievalResults]:
        return self.retriever.batch(
            list(queries),
            config={"max_concurrency": max_concurrency},
        )


class RunnableStrategy:
    """Adapt a LangChain Runnable using explicit query and result conversions."""

    def __init__(
        self,
        runnable: Any,
        *,
        input_mapper: Callable[[str], Any] = lambda query: query,
        result_extractor: Callable[[Any], RetrievalResults] = lambda result: result,
    ) -> None:
        self.runnable = runnable
        self.input_mapper = input_mapper
        self.result_extractor = result_extractor

    def retrieve(self, query: str) -> RetrievalResults:
        result = self.runnable.invoke(self.input_mapper(query))
        return self.result_extractor(result)

    def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        max_concurrency: int,
    ) -> Sequence[RetrievalResults]:
        results = self.runnable.batch(
            [self.input_mapper(query) for query in queries],
            config={"max_concurrency": max_concurrency},
        )
        return [self.result_extractor(result) for result in results]


def as_retrieval_strategy(
    strategy: BaseRetriever | RetrievalStrategy,
) -> RetrievalStrategy:
    """Wrap LangChain retrievers while accepting custom strategy implementations."""
    if isinstance(strategy, BaseRetriever):
        return BaseRetrieverStrategy(strategy)
    retrieve = getattr(strategy, "retrieve", None)
    if not callable(retrieve):
        raise TypeError(
            "A retrieval strategy must be a BaseRetriever or expose "
            "a callable retrieve(query) method."
        )
    return strategy


__all__ = [
    "BatchRetrievalStrategy",
    "BaseRetrieverStrategy",
    "RankedDocuments",
    "RetrievalResults",
    "RetrievalStrategy",
    "RunnableStrategy",
    "as_retrieval_strategy",
]
