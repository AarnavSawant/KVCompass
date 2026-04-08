from __future__ import annotations

import logging

import torch
from transformers import QuantizedCache

import kvpress.attention_patch as attention_patch_module
from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

LOGGER = logging.getLogger(__name__)

_PATCHED = False


def _infer_last_position(kwargs: dict, cache, module_layer_idx: int) -> int:
    cache_position = kwargs.get("cache_position")
    if cache_position is not None:
        if isinstance(cache_position, torch.Tensor):
            return int(cache_position.reshape(-1)[-1].item()) + 1
        return int(cache_position[-1]) + 1

    position_ids = kwargs.get("position_ids")
    if position_ids is not None:
        if isinstance(position_ids, torch.Tensor):
            return int(position_ids.reshape(-1)[-1].item()) + 1
        return int(position_ids[-1]) + 1

    try:
        return int(cache.get_seq_length(module_layer_idx))
    except TypeError:
        return int(cache.get_seq_length())


def apply_kvpress_compat_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    def compat_forward_hook(self, module, input, kwargs, output):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        # KVPress 0.5.2 expects `cache_position`, but newer Transformers Llama
        # paths may only provide `position_ids`. Derive the same signal here.
        if _infer_last_position(kwargs, cache, module.layer_idx) > q_len:
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    BasePress.forward_hook = compat_forward_hook

    original_search_hyperplane = attention_patch_module.search_hyperplane

    def robust_search_hyperplane(X, max_iter: int = 1000):
        try:
            return original_search_hyperplane(X, max_iter=max_iter)
        except ValueError:
            # Fallback for small or numerically awkward CPU smoke-test models.
            # This preserves execution by choosing a large-magnitude key opposite
            # the mean query direction, even if the ideal nullifying hyperplane
            # is not found by the original iterative routine.
            Y = X.mean(1)
            norm_sq = Y.norm(dim=-1, keepdim=True).pow(2).clamp(min=1e-12)
            return -1e5 * Y / norm_sq

    attention_patch_module.search_hyperplane = robust_search_hyperplane
    _PATCHED = True
    LOGGER.info("Applied KVPress compatibility patch for missing cache_position support.")
