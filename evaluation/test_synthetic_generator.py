import json
import unittest

from evaluation.generation.synthetic_generator import SyntheticGenerator


class FakeGenerationChain:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.retry_calls = 0

    def batch(self, inputs: list[dict[str, str]]) -> list[str]:
        self.batch_sizes.append(len(inputs))
        return [
            "1"
            if item["text"] == "chunk-1"
            else self._valid_response(item["text"])
            for item in inputs
        ]

    def invoke(self, values: dict[str, str]) -> str:
        self.retry_calls += 1
        return self._valid_response(values["text"])

    @staticmethod
    def _valid_response(text: str) -> str:
        return json.dumps(
            {
                "queries": [
                    {"query": f"What is answered by {text}?"},
                    {"query": f"How does {text} work?"},
                ]
            }
        )


class SyntheticGeneratorBatchTests(unittest.TestCase):
    def test_generate_batches_prompts_and_retries_only_invalid_output(self) -> None:
        chain = FakeGenerationChain()
        generator = SyntheticGenerator({"batch_size": 2})
        generator._chain = chain
        chunks = [
            {"doc_id": f"doc-{index}", "text": f"chunk-{index}"}
            for index in range(5)
        ]

        generated = generator.generate(chunks)

        self.assertEqual(chain.batch_sizes, [2, 2, 1])
        self.assertEqual(chain.retry_calls, 1)
        self.assertEqual(len(generated["queries"]), 10)
        self.assertEqual(len(generated["qrels"]), 10)
        self.assertEqual(
            set().union(*generated["qrels"].values()),
            {f"doc-{index}" for index in range(5)},
        )

    def test_generate_rejects_non_positive_batch_size(self) -> None:
        generator = SyntheticGenerator({"batch_size": 0})
        generator._chain = FakeGenerationChain()

        with self.assertRaisesRegex(ValueError, "batch_size"):
            generator.generate([{"doc_id": "doc-1", "text": "chunk"}])


if __name__ == "__main__":
    unittest.main()
