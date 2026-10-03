import inspect
import json
import tempfile
import unittest
from pathlib import Path
from threading import Barrier

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config import GROUND_TRUTH_JSON_PATH
from evaluation.evaluate import Evaluator


class FakeRetriever(BaseRetriever):
    results: dict[str, list[Document]]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        return self.results[query]


class BarrierRetriever(FakeRetriever):
    barrier: Barrier

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        self.barrier.wait(timeout=3)
        return super()._get_relevant_documents(query, run_manager=run_manager)


class RetrievalMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth = {
            "queries": {"q1": "question"},
            "qrels": {"q1": {"doc-1": 1}},
        }
        self.ground_truth_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        json.dump(self.ground_truth, self.ground_truth_file)
        self.ground_truth_file.close()
        self.ground_truth_path = Path(self.ground_truth_file.name)
        self.evaluator = Evaluator(ground_truth_path=self.ground_truth_path)

    def tearDown(self) -> None:
        self.ground_truth_path.unlink(missing_ok=True)

    def test_retrieval_metrics_calculates_precision_recall_and_mrr(self) -> None:
        metrics = Evaluator(
            ground_truth_path=self.ground_truth_path,
            k_values=(1, 3),
        ).retrieval_metrics(
            ["wrong", "relevant", "another-relevant"],
            {"relevant", "another-relevant"},
        )

        self.assertEqual(metrics["precision@1"], 0.0)
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["mrr@1"], 0.0)
        self.assertAlmostEqual(metrics["precision@3"], 2 / 3)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr@3"], 0.5)

    def test_mean_retrieval_metrics_macro_averages_cases(self) -> None:
        metrics = Evaluator(
            ground_truth_path=self.ground_truth_path,
            k_values=(1,),
        ).mean_retrieval_metrics(
            [
                (["relevant"], {"relevant"}),
                (["wrong"], {"relevant"}),
            ]
        )

        self.assertEqual(metrics["precision@1"], 0.5)
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["mrr@1"], 0.5)

    def test_evaluate_retriever_uses_doc_ids_from_metadata(self) -> None:
        retriever = FakeRetriever(
            results={
                "question one": [
                    Document(page_content="irrelevant", metadata={"doc_id": "wrong"}),
                    Document(page_content="answer", metadata={"doc_id": "doc-1"}),
                ],
                "question two": [
                    Document(page_content="answer", metadata={"doc_id": "doc-2"}),
                ],
            }
        )
        ground_truth = {
            "queries": {"q1": "question one", "q2": "question two"},
            "qrels": {"q1": {"doc-1": 1}, "q2": {"doc-2": 1}},
        }

        metrics = Evaluator(
            ground_truth_path=self.ground_truth_path,
            k_values=(1, 2),
        ).evaluate(retriever, ground_truth)

        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["mrr@2"], 0.75)
        self.assertEqual(metrics["recall@1"], 0.5)

    def test_evaluate_retrieves_queries_concurrently(self) -> None:
        retriever = BarrierRetriever(
            results={
                "question one": [
                    Document(page_content="answer", metadata={"doc_id": "doc-1"}),
                ],
                "question two": [
                    Document(page_content="answer", metadata={"doc_id": "doc-2"}),
                ],
            },
            barrier=Barrier(2),
        )
        ground_truth = {
            "queries": {"q1": "question one", "q2": "question two"},
            "qrels": {"q1": {"doc-1": 1}, "q2": {"doc-2": 1}},
        }

        metrics = Evaluator(
            ground_truth_path=self.ground_truth_path,
            k_values=(1,),
            max_workers=2,
        ).evaluate(retriever, ground_truth)

        self.assertEqual(metrics["recall@1"], 1.0)

    def test_evaluate_retriever_rejects_missing_doc_id_metadata(self) -> None:
        retriever = FakeRetriever(
            results={
                "question": [Document(page_content="answer", metadata={})],
            }
        )

        with self.assertRaisesRegex(ValueError, "doc_id"):
            self.evaluator.evaluate(
                retriever,
            )

    def test_evaluator_rejects_invalid_k_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "k_values"):
            Evaluator(
                ground_truth_path=self.ground_truth_path,
                k_values=(1, 0),
            )

    def test_evaluator_rejects_invalid_max_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            Evaluator(
                ground_truth_path=self.ground_truth_path,
                max_workers=0,
            )

    def test_evaluate_loads_ground_truth_from_configured_path(self) -> None:
        retriever = FakeRetriever(
            results={
                "question": [
                    Document(page_content="answer", metadata={"doc_id": "doc-1"}),
                ],
            }
        )

        metrics = self.evaluator.evaluate(retriever)

        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(self.evaluator.ground_truth_dataset, self.ground_truth)

    def test_default_ground_truth_path_uses_project_configuration(self) -> None:
        default_path = inspect.signature(Evaluator).parameters[
            "ground_truth_path"
        ].default
        self.assertEqual(GROUND_TRUTH_JSON_PATH, default_path)

    def test_explicit_empty_dataset_is_validated_instead_of_using_loaded_data(self) -> None:
        retriever = FakeRetriever(results={})

        with self.assertRaisesRegex(ValueError, "queries.*qrels"):
            self.evaluator.evaluate(retriever, {})


if __name__ == "__main__":
    unittest.main()
