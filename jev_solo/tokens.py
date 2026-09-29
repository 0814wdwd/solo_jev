"""Token counting for Jev state serialization.

Jev's tokenizer is not public. Every count here is therefore a *proxy*, and the
whole module is built so the proxy can be swapped for empirically measured
billed-token counts once probe_tokenize.py has run against the real API.

Counters are pluggable via `get_counter(name)`. All of them cache per-string
counts, because a relational table serializes the same cell values over and
over -- that cache is what makes whole-table costing tractable.
"""
from __future__ import annotations

import functools
from typing import Callable, Dict, Sequence


class TokenCounter:
    """A cached string -> token-count function."""

    def __init__(self, name: str, fn: Callable[[str], int]):
        self.name = name
        self._fn = fn
        self._cache: Dict[str, int] = {}

    def __call__(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is None:
            cached = self._fn(text)
            self._cache[text] = cached
        return cached

    def count_many(self, texts: Sequence[str]) -> int:
        return sum(self(t) for t in texts)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


def _tiktoken_counter(encoding: str) -> Callable[[str], int]:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


def _hf_counter(model_id: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def _chars_over_four(text: str) -> int:
    # Crude fallback; only for environments with no tokenizer available.
    return max(1, (len(text) + 3) // 4)


@functools.lru_cache(maxsize=None)
def get_counter(name: str = "cl100k_base") -> TokenCounter:
    """Return a cached TokenCounter.

    Names:
      cl100k_base / o200k_base  -- tiktoken encodings (default proxy)
      hf:<model_id>             -- any HuggingFace tokenizer, e.g. hf:Qwen/Qwen2.5-7B-Instruct
      chars4                    -- len/4 fallback, no dependencies
    """
    if name == "chars4":
        return TokenCounter(name, _chars_over_four)
    if name.startswith("hf:"):
        return TokenCounter(name, _hf_counter(name[3:]))
    return TokenCounter(name, _tiktoken_counter(name))


def column_value_weights(
    rows: Sequence[Sequence[str]],
    counter: TokenCounter,
    sample: int = 4000,
    seed: int = 0,
) -> list[float]:
    """Mean token cost of one value per column: the w_c in the objective.

    Estimated on a row sample, since w_c only needs to rank columns.
    """
    import random

    if not rows:
        return []
    n_cols = len(rows[0])
    idx = range(len(rows))
    if len(rows) > sample:
        idx = random.Random(seed).sample(range(len(rows)), sample)
    totals = [0] * n_cols
    seen = 0
    for i in idx:
        row = rows[i]
        for c in range(n_cols):
            totals[c] += counter(str(row[c]))
        seen += 1
    return [t / max(1, seen) for t in totals]


def rendered_field_weights(
    rows: Sequence[Sequence[str]],
    header: Sequence[str],
    counter: TokenCounter,
    sample: int = 4000,
    seed: int = 0,
    template: str = " | {name}={value}",
) -> list[float]:
    """Mean token cost of a *rendered* field, e.g. " | Origin=BDL".

    Summing tokens of "Origin", "=" and "BDL" separately overcounts, because
    tokenizers merge across those boundaries. Measuring the rendered fragment
    removes that bias, which is worth ~16% of the total on flight.
    """
    import random

    if not rows:
        return []
    n_cols = len(rows[0])
    idx = list(range(len(rows)))
    if len(rows) > sample:
        idx = random.Random(seed).sample(idx, sample)
    totals = [0] * n_cols
    for i in idx:
        row = rows[i]
        for c in range(n_cols):
            totals[c] += counter(template.format(name=header[c], value=str(row[c])))
    return [t / max(1, len(idx)) for t in totals]
