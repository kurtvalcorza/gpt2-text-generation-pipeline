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
from collections.abc import Mapping, Sequence
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


# The pre-registered re-selection grid of 2026-10-17 (docs/reselection-2026-10-17.md). Everything else is
# fixed: 2 epochs with the best epoch chosen on validation, batch 8, AdamW wd 0.01, clip 1.0, ADAPT_SEED 0,
# 512 training tokens.
RESELECTION_GRID: tuple[dict[str, Any], ...] = (
    {"name": "C0", "trainable_blocks": 4, "lr": 1e-4, "role": "incumbent default (kept unless beaten)"},
    {"name": "C1", "trainable_blocks": 2, "lr": 1e-4, "role": "smaller"},
    {"name": "C2", "trainable_blocks": 2, "lr": 5e-5, "role": "smaller and gentler"},
)
RESELECTION_MARGIN = 0.02  # a challenger must beat the incumbent by at least 2 % relative


def select_configuration(
    scores: Mapping[str, float],
    *,
    grid: Sequence[Mapping[str, Any]] = RESELECTION_GRID,
    margin: float = RESELECTION_MARGIN,
) -> dict[str, Any]:
    """Apply the pre-registered rule to best-epoch validation perplexities keyed by config name.

    The first grid entry is the incumbent. A challenger replaces it only if it is at least `margin` lower
    (relative); among challengers that clear the margin the lowest wins, unless two are within `margin` of
    each other, in which case the smaller configuration (fewer blocks, then lower learning rate) wins."""
    by_name = {g["name"]: dict(g) for g in grid}
    if set(scores) != set(by_name):
        raise ValueError(f"scores must cover exactly {sorted(by_name)}; got {sorted(scores)}")
    for name, value in scores.items():
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
            raise ValueError(f"{name}: a validation perplexity must be positive and finite, got {value!r}")
    incumbent = grid[0]["name"]
    base = scores[incumbent]
    relative = {name: (base - scores[name]) / base for name in by_name}
    cleared = [g["name"] for g in grid[1:] if relative[g["name"]] >= margin]

    def size(name: str) -> tuple[int, float]:
        return (by_name[name]["trainable_blocks"], by_name[name]["lr"])

    if not cleared:
        chosen = incumbent
        reason = f"no challenger beat {incumbent} by {margin:.0%} or more; {incumbent} stays"
    else:
        best = min(cleared, key=lambda n: (scores[n], size(n)))
        close = [n for n in cleared if (scores[n] - scores[best]) / scores[best] < margin]
        chosen = min(close, key=size)
        reason = f"{chosen} beat {incumbent} by {relative[chosen]:.1%}"
        if len(close) > 1:
            others = ", ".join(n for n in close if n != chosen)
            reason += f"; within {margin:.0%} of {others}, the smaller wins"
    return {
        "selected": chosen,
        "config": by_name[chosen],
        "reason": reason,
        "scores": dict(scores),
        "improvement_vs_incumbent": {k: round(v, 6) for k, v in relative.items()},
        "margin": margin,
        "rule": "lowest best-epoch validation perplexity; a challenger replaces the incumbent only if >= "
        "margin better (relative); challengers within margin of each other go to the smaller configuration",
    }
