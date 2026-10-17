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
import random
import statistics
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
            "median": statistics.median(per_record),
            "max": max(per_record),
        },
        "definitions": dict(METRIC_DEFINITIONS),
    }


def unigram_losses(
    train_ids: Sequence[Sequence[int]],
    test_ids: Sequence[Sequence[int]],
    vocab_size: int,
    alpha: float = 1.0,
) -> list[list[float]]:
    """Per-token NLLs of an add-`alpha`-smoothed unigram model fitted on `train_ids`, scored on `test_ids`
    (skipping each record's first token, like the model's teacher-forced scoring)."""
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if not alpha > 0:
        raise ValueError("alpha must be positive")
    counts: Counter[int] = Counter()
    for ids in train_ids:
        counts.update(int(t) for t in ids)
    total = sum(counts.values()) + alpha * vocab_size
    log_denominator = math.log(total)
    out = []
    for ids in test_ids:
        out.append([log_denominator - math.log(counts.get(int(t), 0) + alpha) for t in ids[1:]])
    return out


def unigram_baseline(
    train_ids: Sequence[Sequence[int]],
    test_ids: Sequence[Sequence[int]],
    vocab_size: int,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Perplexity of a context-free unigram reference. Add-one smoothing (the default) spreads 50,257 pseudo-
    counts over the vocabulary, so on a small training split it is a deliberately weak baseline; a smaller
    `alpha` gives a stronger one."""
    result = sequence_metrics(unigram_losses(train_ids, test_ids, vocab_size, alpha))
    label = "add-one" if alpha == 1.0 else f"add-{alpha:g}"
    result["baseline"] = f"{label}-smoothed unigram model fitted on the training tokens (ignores context)"
    result["alpha"] = alpha
    return result


def paired_delta(
    before: Mapping[str, Mapping[str, float]],
    after: Mapping[str, Mapping[str, float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired per-record comparison of two scorings of the same records (`evaluate(...)["records"]`).

    Per record: delta bits per token = (after mean NLL - before mean NLL) / ln 2; negative: easier.
    Reports the median and quartiles of those deltas, how many improved, the token-weighted delta and a
    paired bootstrap 95 % interval of it (records resampled with replacement, `seed` fixed)."""
    if set(before) != set(after) or not before:
        raise ValueError("paired_delta needs the same, non-empty set of record ids on both sides")
    ids = sorted(before)
    for rid in ids:
        if before[rid]["n_tokens"] != after[rid]["n_tokens"]:
            raise ValueError(f"record {rid!r} was scored on different tokens before and after")
    ln2 = math.log(2)
    delta = [(after[r]["mean_nll"] - before[r]["mean_nll"]) / ln2 for r in ids]
    weight = [before[r]["n_tokens"] for r in ids]

    def weighted(index: Sequence[int]) -> float:
        return sum(delta[i] * weight[i] for i in index) / sum(weight[i] for i in index)

    rng = random.Random(seed)
    n = len(ids)
    boots = sorted(weighted([rng.randrange(n) for _ in range(n)]) for _ in range(n_boot))
    q1, median, q3 = statistics.quantiles(delta, n=4, method="inclusive") if n > 1 else (delta[0],) * 3
    return {
        "n": n,
        "n_improved": sum(d < 0 for d in delta),
        "delta_bits_per_token": {"median": median, "q1": q1, "q3": q3, "min": min(delta), "max": max(delta)},
        "token_weighted_delta_bits": weighted(range(n)),
        "bootstrap_95ci": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
        "n_boot": n_boot,
        "seed": seed,
        "method": "paired per-record delta; bootstrap resamples records with replacement",
    }


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


# The score-once rule (2026-10-17): the pinned test split is read only under exactly these settings, the ones
# docs/reselection-2026-10-17.md fixes plus the selected configuration. Anything else is exploratory.
PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "sample",
    "split_seed": 42,
    "dataset_digests": {
        "train": "f6230198abc575863b6a3a116d8ff5b9df93a5f1843eace4156ffd8e81b5caa1",
        "validation": "479448c67fc6328e4039bd81ae3dd6b41aed66a1e40facd208f4bc8577453fbf",
        "test": "a20180b4219a3be5e75fc450406b854ba0b98758c1de92d78f306211aa7f8488",
    },
    "epochs": 2,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "trainable_blocks": 4,
    "adapt_seed": 0,
    "max_train_tokens": 512,
    "configuration": "C0",
}
TEST_LOOKS = ("pre-registered evaluation", "reproduction", "not scored (exploratory)")


def preregistration_diff(settings: Mapping[str, Any]) -> list[str]:
    """The names of the pre-registered settings that `settings` does not match (missing = different)."""
    return [name for name, value in PREREGISTERED_SETTINGS.items() if settings.get(name) != value]


def test_look(differs: Sequence[str], earlier_looks: int) -> str:
    """Label a run: exploratory if any setting differs, else the pre-registered evaluation or a reproduction
    of it when an identical run already scored the test split (in this session, or as recorded in
    docs/release-verification.md, where the first recorded pre-registered hosted run is the result)."""
    if differs:
        return TEST_LOOKS[2]
    return TEST_LOOKS[1] if earlier_looks else TEST_LOOKS[0]
