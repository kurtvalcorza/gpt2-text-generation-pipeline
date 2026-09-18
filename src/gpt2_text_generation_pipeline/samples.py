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
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    splits = {
        "test": unique[:n_test],
        "validation": unique[n_test : n_test + n_val],
        "train": unique[n_test + n_val :],
    }
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, text}` records from a CSV (columns id, text), a JSON array or JSONL of such objects, or a
    plain `.txt` file in which every non-empty line (or blank-line-separated paragraph) is one record."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
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
