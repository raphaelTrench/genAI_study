"""Model-provider adapters used by the project's generation flows."""

from __future__ import annotations

import sysconfig
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_GENERATION_DEVICE,
    DEFAULT_TEXT_GENERATION_MODEL_ID,
    GENERATION_BATCH_SIZE,
    GENERATION_MAX_NEW_TOKENS,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline


def _disable_triton_bmm_when_python_headers_are_missing(device: int) -> None:
    if device < 0 or (Path(sysconfig.get_path("include")) / "Python.h").is_file():
        return

    try:
        from torch._native.registry import (
            deregister_op_overrides,
            get_dsl_operations,
        )
    except ImportError:
        return

    if "bmm" not in get_dsl_operations("triton"):
        return

    deregister_op_overrides(
        disable_dsl_names="triton",
        disable_op_symbols="bmm",
    )
    warnings.warn(
        "Python.h is unavailable, so PyTorch's optional Triton bmm override "
        "has been disabled. Generation will use PyTorch's standard CUDA "
        "matrix multiplication. Install the matching Python development "
        "headers to enable the Triton optimization.",
        RuntimeWarning,
        stacklevel=2,
    )


class ModelClient(ABC):
    """Shared interface for model-provider adapters."""

    provider_name = "abstract"

    def __init__(self, *, model_id: str, config: Mapping[str, Any] | None = None) -> None:
        self.model_id = model_id
        self.config = dict(config or {})

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        provider: str | None = None,
    ) -> "ModelClient":
        cfg = dict(config or {})
        provider_name = (provider or cfg.get("provider") or "huggingface").lower()
        model_id = cfg.get("model_id") or DEFAULT_TEXT_GENERATION_MODEL_ID

        if provider_name == "huggingface":
            return HuggingFaceModelClient(model_id=model_id, config=cfg)
        if provider_name in {"openai", "chatopenai", "langchain-openai"}:
            return OpenAIModelClient(model_id=model_id, config=cfg)
        if provider_name == "runnable":
            runnable = cfg.get("runnable")
            if runnable is None:
                raise ValueError("provider='runnable' requires a 'runnable' entry in config.")
            return RunnableModelClient(runnable=runnable, model_id=model_id, config=cfg)
        raise ValueError(f"Unsupported model provider: {provider_name!r}")

    @abstractmethod
    def as_runnable(self) -> Any:
        """Return the provider-specific LangChain runnable."""

    def build_chain(
        self,
        prompt: ChatPromptTemplate,
        *,
        render_prompt: Callable[[Any], str] | None = None,
    ) -> Runnable:
        chain: Runnable = prompt
        if render_prompt is not None:
            chain = chain | RunnableLambda(render_prompt)
        return chain | self.as_runnable() | StrOutputParser()

    def invoke(self, data: Any, **kwargs: Any) -> Any:
        return self.as_runnable().invoke(data, **kwargs)

    def batch(self, inputs: Sequence[Any], **kwargs: Any) -> list[Any]:
        return self.as_runnable().batch(inputs, **kwargs)


class RunnableModelClient(ModelClient):
    provider_name = "runnable"

    def __init__(
        self,
        *,
        runnable: Any,
        model_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id or "custom-runnable", config=config)
        self._runnable = runnable

    def as_runnable(self) -> Any:
        return self._runnable


class HuggingFaceModelClient(ModelClient):
    provider_name = "huggingface"

    def __init__(self, *, model_id: str, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(model_id=model_id, config=config)
        self._tokenizer: Any | None = None
        self._pipeline: Any | None = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            model_id = self.model_id
            device = self.config.get("device", DEFAULT_GENERATION_DEVICE)
            _disable_triton_bmm_when_python_headers_are_missing(device)
            pipeline_kwargs = {
                "max_new_tokens": self.config.get(
                    "max_new_tokens", GENERATION_MAX_NEW_TOKENS
                ),
                "do_sample": True,
                "temperature": self.config.get("temperature", 0.7),
                "return_full_text": False,
            }
            model_kwargs = {
                "local_files_only": self.config.get("local_files_only", False)
            }
            pipeline = HuggingFacePipeline.from_model_id(
                model_id=model_id,
                task="text-generation",
                device=device,
                batch_size=self.config.get("batch_size", GENERATION_BATCH_SIZE),
                pipeline_kwargs=pipeline_kwargs,
                model_kwargs=model_kwargs,
            )
            self._pipeline = pipeline
            self._tokenizer = pipeline.pipeline.tokenizer
        return self._pipeline

    def render_chat_prompt(self, prompt_value: Any) -> str:
        if self._tokenizer is None:
            self._ensure_pipeline()
        if self._tokenizer is None:
            raise RuntimeError("The Hugging Face tokenizer has not been initialized.")
        messages = prompt_value.to_messages()
        chat_messages = [
            {
                "role": {"human": "user", "ai": "assistant"}.get(
                    message.type, message.type
                ),
                "content": message.content,
            }
            for message in messages
        ]
        return self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def as_runnable(self) -> Any:
        return self._ensure_pipeline()

    def build_chain(
        self,
        prompt: ChatPromptTemplate,
        *,
        render_prompt: Callable[[Any], str] | None = None,
    ) -> Runnable:
        return super().build_chain(
            prompt,
            render_prompt=render_prompt or self.render_chat_prompt,
        )


class OpenAIModelClient(ModelClient):
    provider_name = "openai"

    def __init__(self, *, model_id: str, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(model_id=model_id, config=config)
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from langchain_openai import ChatOpenAI

            filtered_kwargs = {
                key: value
                for key, value in self.config.items()
                if key
                not in {
                    "provider",
                    "model_id",
                    "batch_size",
                    "device",
                    "local_files_only",
                    "max_new_tokens",
                    "temperature",
                    "max_tokens",
                }
            }
            self._model = ChatOpenAI(
                model=self.model_id,
                temperature=self.config.get("temperature", 0),
                max_tokens=self.config.get("max_tokens"),
                **filtered_kwargs,
            )
        return self._model

    def as_runnable(self) -> Any:
        return self._ensure_model()


ModelProvider = ModelClient
AbstractModelClient = ModelClient

__all__ = [
    "AbstractModelClient",
    "HuggingFaceModelClient",
    "ModelClient",
    "ModelProvider",
    "OpenAIModelClient",
    "RunnableModelClient",
]
