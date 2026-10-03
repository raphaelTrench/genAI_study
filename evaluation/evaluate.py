"""Reusable retrieval evaluation helpers for RAG experiments."""

import csv
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import EVALUATION_RESULTS_CSV_PATH, GROUND_TRUTH_JSON_PATH
from langchain_core.retrievers import BaseRetriever
from ranx import Qrels, Run, evaluate as ranx_evaluate


class Evaluator:
    """Evaluate LangChain retrievers against labeled query datasets."""

    def __init__(
        self,
        ground_truth_path: Path = GROUND_TRUTH_JSON_PATH,
        k_values: Sequence[int] = (1, 3, 5),
        max_workers: int = 4,
        experiment_name: str | None = None,
        results_csv_path: Path = EVALUATION_RESULTS_CSV_PATH,
    ) -> None:
        self.k_values = tuple(k_values)
        if not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must contain positive integers")
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        if experiment_name is not None and not str(experiment_name).strip():
            raise ValueError("experiment_name must be a non-empty string when provided")
        self.max_workers = max_workers
        self.experiment_name = experiment_name or "unnamed_experiment"
        self.results_csv_path = Path(results_csv_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth_dataset = self._load_ground_truth()

    def _write_metrics_to_csv(
        self,
        metrics: Mapping[str, float],
        experiment_name: str | None = None,
    ) -> None:
        """Append the metrics for one run to the shared experiment CSV."""
        row_experiment_name = (
            str(experiment_name).strip()
            if experiment_name is not None
            else self.experiment_name
        )
        if not str(row_experiment_name).strip():
            row_experiment_name = "unnamed_experiment"

        self.results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        metric_names = list(metrics.keys())
        fieldnames = [
            "experiment_name",
            "timestamp",
            "ground_truth_path",
            *metric_names,
        ]

        file_exists = self.results_csv_path.exists()
        with self.results_csv_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists or self.results_csv_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "experiment_name": row_experiment_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ground_truth_path": str(self.ground_truth_path),
                    **{metric_name: float(metrics[metric_name]) for metric_name in metric_names},
                }
            )

    def _load_ground_truth(self) -> dict[str, Any]:
        try:
            dataset = json.loads(self.ground_truth_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise OSError(
                f"Could not read ground-truth dataset at "
                f"{self.ground_truth_path}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Ground-truth dataset at {self.ground_truth_path} is invalid JSON: "
                f"{error}"
            ) from error

        if not isinstance(dataset, dict):
            raise ValueError(
                f"Ground-truth dataset at {self.ground_truth_path} "
                "must be a JSON object"
            )
        return dataset

    def retrieval_metrics(
        self,
        retrieved_ids: Sequence[str],
        relevant_ids: Collection[str],
    ) -> dict[str, float]:
        """Calculate precision, recall, and reciprocal rank at each configured k."""
        if not relevant_ids:
            raise ValueError("relevant_ids must contain at least one relevant ID")

        ranked_ids = list(dict.fromkeys(retrieved_ids))
        return self._evaluate_rankings(
            {"q": {item_id: 1 for item_id in relevant_ids}},
            {"q": self._score_ranked_ids(ranked_ids)},
        )

    def mean_retrieval_metrics(
        self,
        cases: Iterable[tuple[Sequence[str], Collection[str]]],
    ) -> dict[str, float]:
        """Return macro-averaged metrics over labeled query cases."""
        qrels: dict[str, dict[str, int | float]] = {}
        run: dict[str, dict[str, float]] = {}
        for case_count, (retrieved_ids, relevant_ids) in enumerate(cases):
            if not relevant_ids:
                raise ValueError("relevant_ids must contain at least one relevant ID")
            query_id = f"q{case_count}"
            qrels[query_id] = {item_id: 1 for item_id in relevant_ids}
            run[query_id] = self._score_ranked_ids(
                list(dict.fromkeys(retrieved_ids))
            )

        if not qrels:
            raise ValueError("cases must contain at least one labeled query")

        return self._evaluate_rankings(qrels, run)

    def _score_ranked_ids(self, ranked_ids: Sequence[str]) -> dict[str, float]:
        """Encode retrieval order as descending scores for ``ranx.Run``."""
        unique_ids = list(dict.fromkeys(ranked_ids))
        return {
            item_id: float(len(unique_ids) - rank)
            for rank, item_id in enumerate(unique_ids)
        }

    def _evaluate_rankings(
        self,
        qrels: dict[str, dict[str, int | float]],
        run: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        metrics = [
            f"{metric}@{k}"
            for k in self.k_values
            for metric in ("precision", "recall", "mrr")
        ]
        results = ranx_evaluate(Qrels(qrels), Run(run), metrics=metrics)
        return {name: float(score) for name, score in results.items()}

    def _retrieve_ranked_ids(
        self,
        retriever: BaseRetriever,
        query_id: str,
        query: str,
    ) -> tuple[str, dict[str, float]]:
        documents = retriever.invoke(query)
        retrieved_ids: list[str] = []
        for document in documents:
            doc_id = document.metadata.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(
                    f"Retriever result for query {query_id!r} is missing "
                    "a non-empty 'doc_id' in document metadata"
                )
            retrieved_ids.append(doc_id)
        return query_id, self._score_ranked_ids(retrieved_ids)

    def evaluate(
        self,
        retriever: BaseRetriever,
        ground_truth_dataset: Mapping[str, Any] | None = None,
        experiment_name: str | None = None,
    ) -> dict[str, float]:
        """Evaluate a retriever against a ``queries``/``qrels`` dataset."""
        dataset = (
            self.ground_truth_dataset
            if ground_truth_dataset is None
            else ground_truth_dataset
        )
        queries = dataset.get("queries")
        qrels = dataset.get("qrels")
        if not isinstance(queries, Mapping) or not isinstance(qrels, Mapping):
            raise ValueError(
                "ground_truth_dataset must contain 'queries' and 'qrels' mappings"
            )
        if not queries:
            raise ValueError("ground_truth_dataset must contain at least one query")
        if set(qrels) - set(queries):
            raise ValueError("qrels contains query IDs that are missing from queries")

        evaluated_qrels: dict[str, dict[str, int | float]] = {}
        validated_queries: list[tuple[str, str]] = []
        for query_id, query in queries.items():
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Query {query_id!r} must be a non-empty string")
            relevant_docs = qrels.get(query_id)
            if not isinstance(relevant_docs, Mapping):
                raise ValueError(f"Query {query_id!r} has no qrels mapping")
            positive_qrels = {
                doc_id: relevance
                for doc_id, relevance in relevant_docs.items()
                if isinstance(relevance, (int, float)) and relevance > 0
            }
            if not positive_qrels:
                raise ValueError(
                    f"Query {query_id!r} has no positively relevant documents"
                )

            evaluated_qrels[query_id] = positive_qrels
            validated_queries.append((query_id, query))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            ranked_results = executor.map(
                lambda item: self._retrieve_ranked_ids(
                    retriever,
                    item[0],
                    item[1],
                ),
                validated_queries,
            )
            run = dict(ranked_results)

        metrics = self._evaluate_rankings(evaluated_qrels, run)

        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")

        self._write_metrics_to_csv(metrics, experiment_name=experiment_name)
        return metrics
