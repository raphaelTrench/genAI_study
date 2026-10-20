import argparse
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config import (
    CHUNKS_JSON_PATH,
    DEFAULT_GENERATION_DEVICE,
    DEFAULT_TEXT_GENERATION_MODEL_ID,
    GENERATION_BATCH_SIZE,
    GENERATION_MAX_NEW_TOKENS,
    GROUND_TRUTH_JSON_PATH,
)
from genai_study.model_client import ModelClient
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel, Field, field_validator

MAX_PARSE_ATTEMPTS = 3


class GeneratedQueries(BaseModel):
    queries: list[str] = Field(
        min_length=2,
        max_length=3,
        description=(
            "2 to 3 distinct user queries that can be answered specifically "
            "by the chunk."
        ),
    )

    @field_validator("queries", mode="before")
    @classmethod
    def extract_query_text(cls, queries: Any) -> Any:
        if isinstance(queries, list):
            return [
                (
                    query["query"]
                    if isinstance(query, dict) and isinstance(query.get("query"), str)
                    else query
                )
                for query in queries
            ]
        return queries

    @field_validator("queries")
    @classmethod
    def queries_must_be_nonempty_and_distinct(cls, queries: list[str]) -> list[str]:
        normalized = [query.strip() for query in queries]
        if any(not query for query in normalized):
            raise ValueError("Generated queries must not be empty.")
        if len({query.casefold() for query in normalized}) != len(normalized):
            raise ValueError("Generated queries must be distinct.")
        return normalized


class SyntheticGenerator:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        model_client: ModelClient | None = None,
    ) -> None:
        self.config = config or {}
        self.model_client = model_client or ModelClient.from_config(self.config)
        self._chain: Any | None = None
        self._parser = PydanticOutputParser(pydantic_object=GeneratedQueries)
        self._tokenizer: Any | None = None

    def _parse_queries(self, completion: str) -> GeneratedQueries:
        try:
            payload = parse_json_markdown(completion)
        except json.JSONDecodeError:
            return self._parser.parse(completion)

        if isinstance(payload, list):
            payload = {"queries": payload}
        return self._parser.parse(json.dumps(payload))

    def _build_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You generate retrieval-evaluation data. Return exactly one "
                    "JSON object matching the required schema. Do not return a "
                    "number, markdown, explanation, or any text outside the "
                    "JSON object. Generate 2 to 3 distinct natural user queries "
                    "directly answered by the chunk; do not add facts.",
                ),
                (
                    "human",
                    "Text chunk:\n{text}\n\n{format_instructions}\n\n"
                    "{retry_instructions}",
                ),
            ]
        ).partial(format_instructions=self._parser.get_format_instructions())
        return self.model_client.build_chain(
            prompt,
            render_prompt=(
                getattr(self.model_client, "render_chat_prompt", None)
                if hasattr(self.model_client, "render_chat_prompt")
                else None
            ),
        )

    def _generate_queries(self, text: str, chain: Any) -> GeneratedQueries:
        retry_instructions = ""
        last_error: OutputParserException | None = None

        for attempt in range(MAX_PARSE_ATTEMPTS):
            completion = chain.invoke(
                {"text": text, "retry_instructions": retry_instructions}
            )
            try:
                return self._parse_queries(completion)
            except OutputParserException as error:
                last_error = error
                if attempt < MAX_PARSE_ATTEMPTS - 1:
                    retry_instructions = (
                        "Your previous response was invalid. Return a corrected "
                        "JSON object matching the schema, with no extra text. "
                        f"Previous response: {completion!r}"
                    )

        if last_error is None:
            raise RuntimeError("Query generation ended without a model response.")
        raise last_error

    def _generate_query_batch(
        self, chunks: Sequence[tuple[str, str]], chain: Any
    ) -> list[GeneratedQueries]:
        completions = chain.batch(
            [
                {"text": text, "retry_instructions": ""}
                for _, text in chunks
            ]
        )
        results: list[GeneratedQueries] = []
        for (doc_id, text), completion in zip(chunks, completions, strict=True):
            try:
                results.append(self._parse_queries(completion))
            except OutputParserException:
                try:
                    results.append(self._generate_queries(text, chain))
                except OutputParserException as error:
                    raise ValueError(
                        f"Failed to generate valid queries for document chunk "
                        f"{doc_id!r} after {MAX_PARSE_ATTEMPTS} attempts: {error}"
                    ) from error
        return results

    def generate(
        self, document_chunks: Sequence[Mapping[str, str]]
    ) -> dict[str, dict[str, Any]]:
        validated_chunks: list[tuple[str, str]] = []
        for chunk in document_chunks:
            doc_id = chunk.get("doc_id")
            text = chunk.get("text")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Each document chunk must have a non-empty doc_id.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Document chunk {doc_id!r} must have non-empty text.")
            validated_chunks.append((doc_id, text))

        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, int]] = {}

        if not validated_chunks:
            return {"queries": queries, "qrels": qrels}

        chain = self._chain
        if chain is None:
            chain = self._build_chain()
            self._chain = chain

        batch_size = self.config.get("batch_size", GENERATION_BATCH_SIZE)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        for start in range(0, len(validated_chunks), batch_size):
            chunk_batch = validated_chunks[start : start + batch_size]
            results = self._generate_query_batch(chunk_batch, chain)

            for (doc_id, _), result in zip(chunk_batch, results, strict=True):
                for query in result.queries:
                    query_id = f"q_{uuid.uuid4().hex[:6]}"
                    while query_id in queries:
                        query_id = f"q_{uuid.uuid4().hex[:6]}"
                    queries[query_id] = query
                    qrels[query_id] = {doc_id: 1}

        return {"queries": queries, "qrels": qrels}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic retrieval queries from document chunks."
    )
    parser.add_argument(
        "--input",
        default=Path(CHUNKS_JSON_PATH),
        type=Path,
        help="JSON file containing a list of {doc_id, text} chunks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(GROUND_TRUTH_JSON_PATH),
        help="Output JSON path (default: results/synthetic_ground_truth.json).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_TEXT_GENERATION_MODEL_ID,
        help=(
            "Hugging Face model ID or local model path "
            f"(default: {DEFAULT_TEXT_GENERATION_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_GENERATION_DEVICE,
        help="Transformers device index; use -1 for CPU or 0 for the first GPU.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=GENERATION_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=GENERATION_BATCH_SIZE,
        help=f"Prompts per GPU batch (default: {GENERATION_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the model to already be available in the local Hugging Face cache.",
    )
    args = parser.parse_args()

    try:
        input_data = json.loads(args.input.read_text(encoding="utf-8"))
    except OSError as error:
        parser.error(f"could not read input file {args.input}: {error}")
    except json.JSONDecodeError as error:
        parser.error(f"invalid JSON in {args.input}: {error}")

    chunks = input_data.get("chunks") if isinstance(input_data, dict) else input_data
    if not isinstance(chunks, list) or not chunks:
        parser.error("input must contain a non-empty list of document chunks.")

    generator = SyntheticGenerator(
        {
            "model_id": args.model_id,
            "device": args.device,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "local_files_only": args.local_files_only,
        }
    )
    try:
        generated_data = generator.generate(chunks)
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(generated_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(generated_data['queries'])} queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
