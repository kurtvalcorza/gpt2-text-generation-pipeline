"""Intrinsic language-model metrics over a held-out text corpus and the unigram baseline.

The model's own per-token negative log-likelihoods (teacher-forced, natural log) are aggregated into
**perplexity** (`exp(mean NLL)`) and **bits per token** (`mean NLL / ln 2`); the mean is taken over every
predicted token of every record (token-weighted), so long records count more than short ones. Perplexity
needs no references, only text: it says how surprised the model is by the domain, not whether its
generations are good, fluent or true. The **unigram baseline** is the perplexity of an add-one-smoothed
unigram model fitted on the training tokens — the floor a model that ignores context reaches.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

METRIC_DEFINITIONS = {
    "perplexity": (
        "exp of the token-weighted mean negative log-likelihood (natural log) of every predicted token in "
        "the held-out records under teacher forcing; lower is better; a record's first token is not predicted"
    ),
    "bits_per_token": "the same mean negative log-likelihood divided by ln 2",
    "mean_nll": "the token-weighted mean negative log-likelihood in nats",
}


def sequence_metrics(losses: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Aggregate per-record lists of per-token NLLs (nats) into perplexity and bits per token."""
    if not losses:
        raise ValueError("no records to score")
    total = 0.0
    n_tokens = 0
    per_record = []
    for record_losses in losses:
        if not record_losses:
            raise ValueError("a record scored no tokens (it needs at least two)")
        s = float(sum(record_losses))
        total += s
        n_tokens += len(record_losses)
        per_record.append(math.exp(s / len(record_losses)))
    mean_nll = total / n_tokens
    return {
        "n_records": len(losses),
        "n_tokens": n_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "bits_per_token": mean_nll / math.log(2),
        "record_perplexity": {
            "min": min(per_record),
            "median": sorted(per_record)[len(per_record) // 2],
            "max": max(per_record),
        },
        "definitions": dict(METRIC_DEFINITIONS),
    }


def unigram_losses(
    train_ids: Sequence[Sequence[int]], test_ids: Sequence[Sequence[int]], vocab_size: int
) -> list[list[float]]:
    """Per-token NLLs of an add-one-smoothed unigram model fitted on `train_ids`, scored on `test_ids`
    (skipping each record's first token, like the model's teacher-forced scoring)."""
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    counts: Counter[int] = Counter()
    for ids in train_ids:
        counts.update(int(t) for t in ids)
    total = sum(counts.values()) + vocab_size
    log_denominator = math.log(total)
    out = []
    for ids in test_ids:
        out.append([log_denominator - math.log(counts.get(int(t), 0) + 1) for t in ids[1:]])
    return out


def unigram_baseline(
    train_ids: Sequence[Sequence[int]], test_ids: Sequence[Sequence[int]], vocab_size: int
) -> dict[str, Any]:
    """Perplexity of the context-free unigram model — the floor a language model must beat."""
    result = sequence_metrics(unigram_losses(train_ids, test_ids, vocab_size))
    result["baseline"] = "add-one-smoothed unigram model fitted on the training tokens (ignores context)"
    return result
