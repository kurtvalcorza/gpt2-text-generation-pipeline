"""Text-corpus dataset contract for domain adaptation of the language model: the pinned SciTLDR sample of
paper abstracts, validation, seeded splitting, BYOD loaders and CSV export.

The default corpus is **real** and far from GPT-2's WebText pre-training distribution: the abstracts of
computer-science papers shipped by SciTLDR (Cachola et al., EMNLP Findings 2020; Apache-2.0). The three
`SciTLDR-A` JSON-Lines files are fetched one by one from the project repository at a pinned commit and refused
on any byte-size or SHA-256 mismatch; only the abstract text is used here. The frozen model's held-out
perplexity on abstracts is the number to beat, and the fine-tuning question is whether a bounded adaptation
of the last transformer blocks lowers it on abstracts the model has not seen.

A record is ``{id, text}``: one document of the domain, no prompt and no reference — perplexity needs none.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_TEXT_CHARS, MODEL_ID

CORPUS_NAME = "SciTLDR-A (paper abstracts)"
CORPUS_RELEASE = "allenai/scitldr @ 5ccad9c00a60ad75c9e04abf7f27d0f53f983b20"
CORPUS_BASE_URL = "https://raw.githubusercontent.com/allenai/scitldr/5ccad9c00a60ad75c9e04abf7f27d0f53f983b20/SciTLDR-Data/SciTLDR-A/"
CORPUS_FILES = {
    "train": ("train.jsonl", 3_155_015, "b222771d387be585cfdf5ae957b36757138415a352e0a3e3b23f73f87c3b1119"),
    "dev": ("dev.jsonl", 1_124_865, "3191fa98ccc09521332b7a1cd63b1930be4e8df125a235ccd31e40329709525e"),
    "test": ("test.jsonl", 1_204_107, "fb42dd6cd4f4a1928ae8a01a189456fbfe994a07e938bd49f68653933f6503c9"),
}
CORPUS_LICENSE = "Apache-2.0 (Cachola et al. 2020; allenai/scitldr)"
CORPUS_PAPERS = {"train": 1_992, "dev": 619, "test": 618}
DEFAULT_CACHE_DIR = Path("weights") / "scitldr"
MAX_SAMPLE_TEXT_CHARS = 2_400  # longer abstracts are left out of the sample (the ceiling is 1,023 tokens)
MIN_SAMPLE_TEXT_CHARS = 200
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 300, "validation": 50, "test": 100}
MIN_RECORDS = 8
MAX_RECORDS = 20_000
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
# The never-scored test draw of the 2026-10-17 re-selection (docs/reselection-2026-10-17.md): 100 abstracts
# of the `test` member outside the SAMPLE_SEED test draw, which the 2026-09 sweep scored and which is kept
# only as the labelled "exposed development test". Texts that also occur in another member are excluded.
# Pinned by paper id and digest, so a changed corpus, filter or draw is refused instead of silently scored.
FRESH_TEST_SEED = 20261017
FRESH_TEST_SIZE = 100
FRESH_TEST_PAPER_IDS: tuple[str, ...] = (
    "HkzyX3CcFQ", "rJEjjoR9K7", "rk1FQA0pW", "H1xL_iR9Km", "BJzVUj0qtQ", "HyI6s40a-", "ByZmGjkA-",
    "S1lNELLKuN", "Hkes0iR9KX", "SkeUG30cFQ", "rkgwuiA9F7", "r1gKNs0qYX", "B1x-LjAcKX", "SJyVzQ-C-",
    "r1xQQhAqKX", "S1gUsoR9YX", "S1sqHMZCb", "BJJLHbb0-", "H1xAH2RqK7", "HygcvsAcFX", "r1gnQ20qYX",
    "Skvin0GWM", "ryEquiR9KX", "rkmu5b0a-", "HJGtFoC5Fm", "B1GAUs0cKQ", "H1BLjgZCb", "S1EwLkW0W",
    "SyMDXnCcF7", "rJgTciR9tm", "rkz1YD0vjm", "r1GaAjRcF7", "rJe7FW-Cb", "Hkfmn5n6W", "HkezXnA9YX",
    "HJDUjKeA-", "SJd0EAy0b", "BJzmzn0ctX", "Hkc-TeZ0W", "B1GHb2RqYX", "rylbWhC5Ym", "BJzbG20cFQ",
    "ryf7ioRqFX", "rkf5hnNDj7", "rJg_NjCqtX", "HyWrIgW0W", "BySRH6CpW", "HJSA_e1AW", "H1gR5iR5FX",
    "S1fNJhRqFX", "SJJQVZW0b", "Hk5elxbRW", "H1g0piA9tQ", "Byx83s09Km", "HJgeEh09KQ", "HJewuJWCZ",
    "r1kP7vlRb", "rJleN20qK7", "r1gGpjActQ", "rJHcpW-CW", "rJeXCo0cYX", "Bym0cU1CZ", "rkxkr8UKuN",
    "HkjL6MiTb", "BJ8vJebC-", "BkrSv0lA-", "HJxeGb5pTm", "SkloDjAqYm", "rkMk9j0qYm", "HkfPSh05K7",
    "Syeil309tX", "SkrHeXbCW", "SJVmjjR9FX", "Hy8hkYeRb", "rkLyJl-0-", "Hy7EPh10W", "BJe0Gn0cY7",
    "HkgSEnA5KQ", "H1GWAoRcKX", "rJ3fy0k0Z", "r1-4BLaQz", "S1lIMn05F7", "SkffVjUaW", "SJtChcgAW",
    "HyeGBj09Fm", "HyNmRiCqtm", "HkfwpiA9KX", "HJWpQCa7z", "ByOExmWAb", "BylBns0qtX", "SJgsCjCqt7",
    "HkxKH2AcFm", "SkHkeixAW", "rygDeZqap7", "HygBZnRctX", "SJLlmG-AZ", "H135uzZ0-", "ByJHuTgA-",
    "rkgpCoRctm", "BylE1205Fm",
)
FRESH_TEST_DIGEST = "a20180b4219a3be5e75fc450406b854ba0b98758c1de92d78f306211aa7f8488"


class DatasetRefused(ValueError):
    """An expected refusal of the learner's data (BYOD): the notebook shows the message, not a traceback."""

    dimer_refusal = True

    def _render_traceback_(self) -> list[str]:  # IPython shows this instead of a traceback
        return [str(self)]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return the three pinned SciTLDR-A files (bytes) from the cache or the project repository, verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, (name, size, digest) in CORPUS_FILES.items():
        local = cache / name
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = CORPUS_BASE_URL + name
            if fetcher is not None:
                data = fetcher(url)
            else:
                with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{name}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                    f"pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[split] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> dict[str, list[dict[str, Any]]]:
    """Parse the JSON-Lines members into abstract records keeping each SciTLDR `paper_id`."""
    out = {}
    for split in CORPUS_FILES:
        if split not in files:
            raise ValueError(f"corpus is missing the {split} file")
        records = []
        for line in files[split].decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                {
                    "id": f"{split}-{row['paper_id']}",
                    "text": " ".join(str(s).strip() for s in row["source"]),
                    "paper_id": str(row["paper_id"]),
                }
            )
        if len(records) != CORPUS_PAPERS[split]:
            raise ValueError(f"{split}: {len(records)} papers, expected {CORPUS_PAPERS[split]}")
        out[split] = records
    return out


def filter_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep records whose text is within the sample length window; drop repeated texts case-insensitively."""
    seen: set[str] = set()
    kept = []
    for record in records:
        text = str(record["text"])
        if not MIN_SAMPLE_TEXT_CHARS <= len(text) <= MAX_SAMPLE_TEXT_CHARS:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(dict(record))
    return kept


def build_sample_dataset(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draws from the three SciTLDR members: training from `train`, validation from `dev`, test from
    `test` — the release's own paper-disjoint partition, re-checked on texts by `check_split_disjoint`."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    source_of = {"train": "train", "validation": "dev", "test": "test"}
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, size in sizes.items():
        pool = filter_records(corpus[source_of[name]])
        if size > len(pool):
            raise ValueError(f"requested {size} {name} records but only {len(pool)} fit")
        rng.shuffle(pool)
        out[name] = [
            {"id": f"{name}-{i:04d}", "text": r["text"], "paper_id": r["paper_id"]}
            for i, r in enumerate(pool[:size])
        ]
    return out


def build_fresh_test(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = FRESH_TEST_SEED,
    size: int = FRESH_TEST_SIZE,
    verify: bool = True,
) -> list[dict[str, Any]]:
    """The never-scored test draw: `test`-member abstracts that are neither in the SAMPLE_SEED test draw (the
    exposed development test) nor share a text with the `train` or `dev` member, in file order, shuffled with
    `seed`; the first `size`. With `verify`, refused unless the paper ids and digest are the pinned ones."""
    exposed = {r["text"].lower() for r in build_sample_dataset(corpus, seed=SAMPLE_SEED)["test"]}
    elsewhere = {r["text"].lower() for name in ("train", "dev") for r in filter_records(corpus[name])}
    pool = [r for r in filter_records(corpus["test"]) if r["text"].lower() not in exposed | elsewhere]
    random.Random(seed).shuffle(pool)
    if size > len(pool):
        raise ValueError(f"requested {size} fresh test records but only {len(pool)} were never scored")
    out = [{"id": f"fresh-{i:04d}", "text": r["text"], "paper_id": r["paper_id"]} for i, r in enumerate(pool)]
    out = out[:size]
    ids = tuple(r["paper_id"] for r in out)
    if verify and (ids != FRESH_TEST_PAPER_IDS or dataset_digest(out) != FRESH_TEST_DIGEST):
        raise ValueError("the fresh test draw is not the pinned one (paper ids or digest differ)")
    return out


def check_near_duplicates(
    splits: Mapping[str, Sequence[Mapping[str, Any]]], *, n: int = 8, threshold: float = 0.5
) -> dict[str, Any]:
    """Refuse two records in different splits whose word `n`-gram containment (shared n-grams over the smaller
    record's n-grams) is at least `threshold`: an edited resubmission under a new id."""
    grams: list[tuple[str, int, set[tuple[str, ...]]]] = []
    for name, records in splits.items():
        for index, record in enumerate(records):
            words = str(record["text"]).lower().split()
            grams.append((name, index, {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}))
    owners: dict[tuple[str, ...], list[int]] = {}
    for k, (_name, _index, g) in enumerate(grams):
        for gram in g:
            owners.setdefault(gram, []).append(k)
    worst = 0.0
    for k, (name, index, g) in enumerate(grams):
        shared: dict[int, int] = {}
        for gram in g:
            for other in owners[gram]:
                if other > k and grams[other][0] != name:
                    shared[other] = shared.get(other, 0) + 1
        for other, count in shared.items():
            smaller = min(len(g), len(grams[other][2])) or 1
            containment = count / smaller
            worst = max(worst, containment)
            if containment >= threshold:
                o_name, o_index, _ = grams[other]
                raise ValueError(
                    f"{name}[{index}] and {o_name}[{o_index}] share {containment:.0%} of word {n}-grams "
                    "(a near-duplicate across splits)"
                )
    return {"n": n, "threshold": threshold, "max_containment": round(worst, 4)}


def build_tutorial_dataset(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]], *, seed: int = SAMPLE_SEED
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial's four sample splits: `train` and `validation` from the seeded draw, `test` = the pinned
    never-scored draw, and `exposed_test` = the SAMPLE_SEED test draw that the 2026-09 sweep scored (reported
    once, labelled, for continuity; never used for selection). Paper- and text-disjointness are asserted."""
    drawn = build_sample_dataset(corpus, seed=seed)
    exposed = drawn["test"] if seed == SAMPLE_SEED else build_sample_dataset(corpus, seed=SAMPLE_SEED)["test"]
    splits = {
        "train": drawn["train"],
        "validation": drawn["validation"],
        "test": build_fresh_test(corpus),
        "exposed_test": exposed,
    }
    check_split_disjoint(splits)
    check_paper_disjoint(splits)
    check_near_duplicates(splits)
    return splits


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/text")
    for key in ("id", "text"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid, text = record["id"], record["text"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    if not isinstance(text, str):
        raise ValueError(f"{label}: text must be a string")
    if not text.strip():
        raise ValueError(f"{label}: text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"{label}: text has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
    item = {"id": rid, "text": text.strip()}
    if "paper_id" in record:
        item["paper_id"] = str(record["paper_id"])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a text corpus; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, text} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    texts: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        texts.add(item["text"].lower())
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_texts": len(texts),
        "text_chars": {
            "min": min(len(r["text"]) for r in checked),
            "max": max(len(r["text"]) for r in checked),
        },
        "total_chars": sum(len(r["text"]) for r in checked),
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], r["text"]] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no lower-cased text appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record["text"]).lower()
            if key in seen and seen[key] != name:
                raise ValueError(f"a text ({record['text'][:60]!r}…) appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def check_paper_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    """Assert no `paper_id` appears in two splits (records without one, such as BYOD records, are skipped)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            paper = record.get("paper_id")
            if paper is None:
                continue
            if paper in seen and seen[paper] != name:
                raise ValueError(f"paper {paper!r} appears in both {seen[paper]} and {name}")
            seen[paper] = name
    return {name: sum(r.get("paper_id") is not None for r in records) for name, records in splits.items()}


def validate_splits(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Validate each split with the minimum the next stage applies: the training split needs MIN_RECORDS
    (what `adapt` requires); validation and test need one record each (what `evaluate` requires). A refusal
    names the split."""
    manifests = {}
    for name, part in splits.items():
        try:
            manifests[name] = validate_dataset(part, min_records=MIN_RECORDS if name == "train" else 1)
        except ValueError as exc:
            raise ValueError(f"{name} split: {exc}") from None
    return manifests


def _split_sizes(n: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    return n - n_test - n_val, n_val, n_test


def min_split_records(val_fraction: float = 0.15, test_fraction: float = 0.2) -> int:
    """The smallest number of unique records `split_dataset` accepts with these fractions: the training
    split must keep MIN_RECORDS (12 with the default fractions: 8 training + 2 validation + 2 test)."""
    n = MIN_RECORDS
    while _split_sizes(n, val_fraction, test_fraction)[0] < MIN_RECORDS:
        n += 1
    return n


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD corpus into train/validation/test after de-duplicating texts."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = record["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    _n_train, n_val, n_test = _split_sizes(len(unique), val_fraction, test_fraction)
    splits = {
        "test": unique[:n_test],
        "validation": unique[n_test : n_test + n_val],
        "train": unique[n_test + n_val :],
    }
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required. "
            f"Supply at least {min_split_records(val_fraction, test_fraction)} unique records with these "
            f"fractions (got {len(unique)} unique of {len(checked)})"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, text}` records from a CSV (columns id, text), a JSON array or JSONL of such objects, or a
    plain `.txt` file in which every non-empty line (or blank-line-separated paragraph) is one record."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in (".csv", ".json", ".jsonl", ".txt"):
        raise ValueError(
            f"unsupported BYOD file type {suffix or '(none)'!r}: use .csv, .json, .jsonl or .txt"
        )
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{file_path.name} is not UTF-8 text (byte {exc.start}); save it as UTF-8 (a spreadsheet's "
            "'CSV UTF-8') and try again"
        ) from None
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        missing = {"id", "text"} - set(rows[0].keys() if rows else set())
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)}")
        return [{"id": r["id"], "text": r["text"]} for r in rows]
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    if suffix == ".txt":
        paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
        return [{"id": f"doc-{i:05d}", "text": " ".join(p.split())} for i, p in enumerate(paragraphs)]
    raise ValueError("BYOD corpora must be .csv, .json, .jsonl or .txt")


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": record["id"], "text": record["text"]})
    return out
