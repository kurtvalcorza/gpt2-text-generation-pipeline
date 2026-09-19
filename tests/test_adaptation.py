"""Offline tests for the text-corpus dataset contract, the pinned SciTLDR abstract reader, perplexity
metrics and the unigram baseline, BYOD loaders (CSV / JSON / JSONL / TXT), the injected-scorer evaluation
path, artifact-manifest rejections and adapt() argument validation. Nothing here imports torch or
transformers; the corpus is three crafted JSON-Lines files served through an injected fetcher."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from gpt2_text_generation_pipeline import (
    ARTIFACT_FORMAT,
    MAX_PROMPT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    TRANSFORMER_BLOCKS,
    VOCAB_SIZE,
    WEIGHT_SHA256,
    GPT2TextGenerationPipeline,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    fetch_sample_dataset,
    load_byod_dataset,
    sequence_metrics,
    split_dataset,
    unigram_baseline,
    validate_dataset,
    write_dataset_csv,
)
from gpt2_text_generation_pipeline import metrics as mt
from gpt2_text_generation_pipeline import pipeline as pl
from gpt2_text_generation_pipeline import samples as sm
from gpt2_text_generation_pipeline.samples import fetch_corpus, filter_records, read_corpus

ABSTRACTS = [
    (
        "p01",
        [
            "We study graph neural networks for molecule property prediction.",
            "Our model beats three baselines on two benchmarks.",
            "Code is released.",
        ],
    ),
    (
        "p02",
        [
            "Transformers are hard to train on small data.",
            "We propose a curriculum that orders examples by length.",
            "Accuracy improves by four points.",
        ],
    ),
    (
        "p03",
        [
            "Reinforcement learning agents forget old tasks.",
            "We add a replay buffer with prioritised sampling.",
            "Forgetting drops by half.",
        ],
    ),
    (
        "p04",
        [
            "Speech recognition degrades with accents.",
            "We fine-tune on accented data with adapters.",
            "Word error rate falls.",
        ],
    ),
    (
        "p05",
        [
            "Image captioning models hallucinate objects.",
            "We penalise captions naming absent objects.",
            "Hallucination rate halves.",
        ],
    ),
    (
        "p06",
        [
            "Sparse attention scales to long documents.",
            "We route tokens to experts by locality-sensitive hashing.",
            "Memory drops fourfold.",
        ],
    ),
    (
        "p07",
        [
            "Tabular data resists deep learning.",
            "We pretrain a transformer on synthetic tables.",
            "It matches gradient boosting.",
        ],
    ),
    (
        "p08",
        [
            "Machine translation for low-resource languages lacks data.",
            "We back-translate monolingual text.",
            "BLEU rises by six.",
        ],
    ),
    (
        "p09",
        [
            "Protein structure prediction is costly.",
            "We distil a large model into a small one.",
            "Speed improves ten times.",
        ],
    ),
    (
        "p10",
        [
            "Robots grasp unfamiliar objects poorly.",
            "We learn grasps from simulated point clouds.",
            "Success rate reaches ninety percent.",
        ],
    ),
    (
        "p11",
        [
            "Recommender systems amplify popularity bias.",
            "We reweight the loss by item frequency.",
            "Long-tail recall improves.",
        ],
    ),
    (
        "p12",
        [
            "Code models struggle with long files.",
            "We add retrieval over the repository.",
            "Completion accuracy improves.",
        ],
    ),
]


def _rows(prefix="", papers=ABSTRACTS):
    """Crafted SciTLDR rows; the prefix goes into the first sentence so the three members are text-disjoint
    like the real release. `target` and `title` are carried but unused by this row."""
    return [
        {
            "paper_id": f"{prefix}{pid}",
            "source": [f"{prefix.upper()}{src[0]}", *src[1:]] if prefix else src,
            "target": ["unused"],
            "title": f"Title {pid}",
        }
        for pid, src in papers
    ]


def _records(prefix="r"):
    return [{"id": f"{prefix}{i:03d}", "text": " ".join(src)} for i, (_pid, src) in enumerate(ABSTRACTS)]


def _files():
    def jsonl(rows):
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    return {"train": jsonl(_rows("tr-")), "dev": jsonl(_rows("dv-")), "test": jsonl(_rows("te-"))}


def _pin(monkeypatch, files):
    monkeypatch.setattr(
        sm,
        "CORPUS_FILES",
        {k: (f"{k}.jsonl", len(v), hashlib.sha256(v).hexdigest()) for k, v in files.items()},
    )
    monkeypatch.setattr(sm, "CORPUS_PAPERS", {k: 12 for k in files})
    monkeypatch.setattr(sm, "MIN_SAMPLE_TEXT_CHARS", 50)


def _tokenize(text: str) -> list[int]:
    """Word tokeniser over a stable hash into the GPT-2 vocabulary range."""
    return [int(hashlib.sha1(w.lower().encode()).hexdigest()[:6], 16) % VOCAB_SIZE for w in text.split()]


def _scorer(ids: list[int]) -> list[float]:
    """Fake per-token NLL: ln 2 for every predicted token, so perplexity is exactly 2."""
    return [math.log(2.0)] * (len(ids) - 1)


def _pipeline_without_model(scorer=_scorer):
    return GPT2TextGenerationPipeline(
        _tokenize, lambda ids, settings: [1, 2], lambda ids: " x", "cpu", "injected", _scorer=scorer
    )


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_BASE_URL.startswith("https://raw.githubusercontent.com/allenai/scitldr/5ccad9c0")
    assert {k: v[1] for k, v in sm.CORPUS_FILES.items()} == {
        "train": 3_155_015,
        "dev": 1_124_865,
        "test": 1_204_107,
    }
    assert all(len(v[2]) == 64 for v in sm.CORPUS_FILES.values())
    assert sum(SAMPLE_SPLIT.values()) == 450 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[url.rsplit("/", 1)[1].removesuffix(".jsonl")]

    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files
    assert len(calls) == 3 and all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_joins_abstract_sentences_and_checks_counts(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    assert len(corpus["train"]) == 12 and corpus["train"][0]["id"] == "train-tr-p01"
    assert corpus["train"][0]["text"].startswith("TR-We study graph neural networks")
    assert corpus["train"][0]["text"].endswith("Code is released.")
    assert set(corpus["dev"][0]) == {"id", "text", "paper_id"} and corpus["dev"][0]["paper_id"] == "dv-p01"
    with pytest.raises(ValueError, match="missing the test file"):
        read_corpus({"train": files["train"], "dev": files["dev"]})
    monkeypatch.setattr(sm, "CORPUS_PAPERS", {"train": 99, "dev": 12, "test": 12})
    with pytest.raises(ValueError, match="expected 99"):
        read_corpus(files)


def test_filter_and_sample_split_are_seeded_and_disjoint(monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    corpus = read_corpus(files)
    noisy = [
        *corpus["train"],
        {**corpus["train"][0], "id": "dup"},
        {**corpus["train"][1], "id": "short", "text": "Tiny."},
        {**corpus["train"][2], "id": "long", "text": "x" * (sm.MAX_SAMPLE_TEXT_CHARS + 1)},
    ]
    assert len(filter_records(noisy)) == 12
    sizes = {"train": 8, "validation": 3, "test": 4}
    splits = build_sample_dataset(corpus, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == sizes
    assert splits["train"][0]["id"] == "train-0000" and set(splits["test"][0]) == {"id", "text", "paper_id"}
    assert check_split_disjoint(splits) == sizes
    assert build_sample_dataset(corpus, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(corpus, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 100, "validation": 1, "test": 1})
    leaky = {"train": splits["train"], "test": [{**splits["train"][0], "id": "leak"}]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaky)


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _files()
    _pin(monkeypatch, files)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda url: files[url.rsplit("/", 1)[1].removesuffix(".jsonl")],
        sizes={"train": 8, "validation": 2, "test": 2},
    )
    assert validate_dataset(splits["train"])["n_records"] == 8


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(forbid_model_imports):
    report = validate_dataset(_records())
    assert report["n_records"] == 12 and report["unique_texts"] == 12
    assert report["text_chars"]["min"] > 50 and report["total_chars"] == sum(
        len(r["text"]) for r in _records()
    )
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    tagged = validate_dataset([{**_records()[0], "paper_id": 7}, *_records()[1:]])
    assert tagged["records"][0]["paper_id"] == "7" and "paper_id" not in tagged["records"][1]
    good = _records()
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "text": " "}, *good[1:]], "text is empty"),
        ([{**good[0], "text": 5}, *good[1:]], "text must be a string"),
        ([{**good[0], "text": "x" * 4_001}, *good[1:]], "MAX_TEXT_CHARS"),
        ([{"id": "a"}, *good[1:]], "missing 'text'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)


def test_split_dataset_deduplicates_and_is_seeded(forbid_model_imports):
    records = [*_records(), {**_records()[0], "id": "dup"}]
    splits = split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 12 and len(splits["test"]) == 2
    assert check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.1, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)


# --- metrics and baseline -----------------------------------------------------------------------------


def test_sequence_metrics_aggregate_token_weighted(forbid_model_imports):
    metrics = sequence_metrics([[math.log(2.0)] * 3, [math.log(8.0)]])
    assert metrics["n_records"] == 2 and metrics["n_tokens"] == 4
    assert metrics["perplexity"] == pytest.approx(math.exp((3 * math.log(2) + math.log(8)) / 4))
    assert metrics["bits_per_token"] == pytest.approx(1.5)  # (3 * 1 + 3) / 4 bits
    assert metrics["record_perplexity"] == {
        "min": pytest.approx(2.0),
        "median": pytest.approx(8.0),
        "max": pytest.approx(8.0),
    }
    assert "perplexity" in metrics["definitions"]
    with pytest.raises(ValueError, match="no records"):
        sequence_metrics([])
    with pytest.raises(ValueError, match="at least two"):
        sequence_metrics([[]])


def test_unigram_baseline_is_add_one_smoothed(forbid_model_imports):
    losses = mt.unigram_losses([[1, 1, 2]], [[9, 1, 3]], vocab_size=4)
    # counts: {1: 2, 2: 1}, total = 3 + 4 = 7; scored tokens are 1 and 3 (the first token is skipped)
    assert losses == [[pytest.approx(math.log(7 / 3)), pytest.approx(math.log(7 / 1))]]
    result = unigram_baseline([[1, 1, 2]], [[9, 1, 3]], vocab_size=4)
    assert result["n_tokens"] == 2 and "unigram" in result["baseline"]
    assert result["perplexity"] == pytest.approx(math.sqrt(7 / 3 * 7))
    with pytest.raises(ValueError, match="vocab_size"):
        mt.unigram_losses([[1]], [[1, 2]], vocab_size=0)
    pipe = _pipeline_without_model()
    baseline = pipe.unigram_baseline(_records()[:8], _records()[8:])
    assert baseline["n_records"] == 4 and 1.0 < baseline["perplexity"] <= VOCAB_SIZE


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_csv_json_jsonl_txt_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records()
    csv_path = write_dataset_csv(records, tmp_path / "data.csv")
    assert load_byod_dataset(csv_path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.jsonl") == records
    (tmp_path / "data.txt").write_text("\r\n\r\n".join(r["text"] for r in records) + "\n", encoding="utf-8")
    txt = load_byod_dataset(tmp_path / "data.txt")
    assert [r["text"] for r in txt] == [r["text"] for r in records] and txt[0]["id"] == "doc-00000"
    (tmp_path / "bad.csv").write_text("id,source\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv.bak").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="csv, .json, .jsonl or .txt"):
        load_byod_dataset(tmp_path / "data.csv.bak")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.csv")


# --- evaluation, adaptation and artifacts without a model ---------------------------------------------


def test_evaluate_uses_the_injected_scorer_and_refuses_oversized_records(forbid_model_imports):
    pipe = _pipeline_without_model()
    metrics = pipe.evaluate(_records())
    n_tokens = sum(len(r["text"].split()) - 1 for r in _records())
    assert metrics["n_records"] == 12 and metrics["n_tokens"] == n_tokens
    assert metrics["perplexity"] == pytest.approx(2.0) and metrics["bits_per_token"] == pytest.approx(1.0)
    assert metrics["verdict"] == "measured-small-sample" and metrics["adapted"] is False
    assert metrics["model_id"] == MODEL_ID and metrics["seconds"] >= 0.0
    big = {"id": "big", "text": "w " * (MAX_PROMPT_TOKENS + 1)}
    with pytest.raises(ValueError, match="ceiling is 1023"):
        pipe.evaluate([big])
    with pytest.raises(ValueError, match="fewer than two tokens"):
        pipe.evaluate([{"id": "one", "text": "single"}])
    with pytest.raises(ValueError, match="from_pretrained"):
        _pipeline_without_model(scorer=None).evaluate(_records())


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(_records(), batch_size=0)
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.adapt(_records(), trainable_blocks=TRANSFORMER_BLOCKS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["transformer.h.11.mlp.c_fc.weight"],
        "adapter": {"trainable_blocks": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_blocks": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
