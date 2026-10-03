.PHONY: help chunks generate

INPUT ?=
OUTPUT ?= results/synthetic_ground_truth.json
MODEL_ID ?=
DEVICE ?=
BATCH_SIZE ?=
ARGS ?=
CHUNK_SOURCE ?=
CHUNKS_OUTPUT ?=
CHUNK_SIZE ?=
CHUNK_OVERLAP ?=

help:
	@echo "make chunks [CHUNK_SOURCE=<override>] [CHUNKS_OUTPUT=<override>]"
	@echo "       [CHUNK_SIZE=<override>] [CHUNK_OVERLAP=<override>]"
	@echo "Default chunk settings are configured in config.py."
	@echo "make generate [INPUT=<override>] [OUTPUT=<override>]"
	@echo "       [MODEL_ID=<override>] [DEVICE=<override>] [BATCH_SIZE=<override>]"
	@echo "       [ARGS='--local-files-only']; defaults are configured in config.py."
	@echo "The default text-generation model is configured in config.py."

CHUNK_ARGS = $(if $(CHUNK_SOURCE),--input-dir "$(CHUNK_SOURCE)") \
	$(if $(CHUNKS_OUTPUT),--output "$(CHUNKS_OUTPUT)") \
	$(if $(CHUNK_SIZE),--chunk-size "$(CHUNK_SIZE)") \
	$(if $(CHUNK_OVERLAP),--chunk-overlap "$(CHUNK_OVERLAP)")

chunks:
	poetry run python -m evaluation.generation.chunk_generator $(CHUNK_ARGS)

ifneq ($(strip $(MODEL_ID)),)
MODEL_ARGS = --model-id "$(MODEL_ID)"
endif
ifneq ($(strip $(DEVICE)),)
DEVICE_ARGS = --device "$(DEVICE)"
endif
ifneq ($(strip $(BATCH_SIZE)),)
BATCH_ARGS = --batch-size "$(BATCH_SIZE)"
endif

generate:
	poetry run python -m evaluation.generation.synthetic_generator \
		$(if $(INPUT),--input "$(INPUT)") \
		$(MODEL_ARGS) \
		$(DEVICE_ARGS) \
		$(BATCH_ARGS) \
		$(ARGS)
