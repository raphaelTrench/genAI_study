import argparse
import json
from pathlib import Path

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_JSON_PATH,
    TEXT_SOURCE_DIR,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def generate_chunks(
    input_dir: Path,
    output_path: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    source_files = sorted(input_dir.rglob("*.txt"))
    if not source_files:
        raise ValueError(f"No .txt files found in {input_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[dict[str, str]] = []

    for source_path in source_files:
        text = source_path.read_text(encoding="utf-8")
        relative_source = source_path.relative_to(input_dir).as_posix()
        for index, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append(
                {
                    "doc_id": f"{relative_source}::chunk-{index:04d}",
                    "source": relative_source,
                    "text": chunk_text,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split source text files and save document chunks as JSON."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TEXT_SOURCE_DIR,
        help="Directory to recursively scan for .txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHUNKS_JSON_PATH,
        help="Path for the generated JSON chunk list.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    try:
        chunk_count = generate_chunks(
            args.input_dir,
            args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Saved {chunk_count} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
