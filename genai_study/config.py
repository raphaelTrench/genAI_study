"""Shared defaults for local Hugging Face models used by the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TEXT_GENERATION_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

TEXT_SOURCE_DIR = PROJECT_ROOT / "data/raw/mistborn"
CHUNKS_JSON_PATH = PROJECT_ROOT / "data/chunks/mistborn.json"
GROUND_TRUTH_JSON_PATH = PROJECT_ROOT / "data/ground_truth/mistborn.json"
EVALUATION_RESULTS_CSV_PATH = PROJECT_ROOT / "results" / "evaluation_metrics.csv"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_GENERATION_DEVICE = 0
GENERATION_BATCH_SIZE = 16
GENERATION_MAX_NEW_TOKENS = 128
BASE_QUESTION = "What is the length of Vin's hair?"
