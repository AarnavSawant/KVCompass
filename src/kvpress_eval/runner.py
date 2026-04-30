from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import KVPressTextGenerationPipeline

from .compat import apply_kvpress_compat_patches

LOGGER = logging.getLogger(__name__)


def load_model_bundle(model_name: str, device: str = "auto", torch_dtype: str = "auto") -> tuple[Any, Any, Any]:
    apply_kvpress_compat_patches()

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    if device == "auto" and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    elif device not in {"auto", "cpu"}:
        kwargs["device_map"] = device

    LOGGER.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    return model, tokenizer, pipeline
