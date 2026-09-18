"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): held-out
perplexity of the frozen model against the unigram floor, a one-epoch adaptation of the last transformer
block on a dozen abstracts, and the artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from gpt2_text_generation_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, GPT2TextGenerationPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

ABSTRACTS = [
    "We study graph neural networks for molecule property prediction. Our model beats three baselines.",
    "Transformers are hard to train on small data. We propose a curriculum that orders examples by length.",
    "Reinforcement learning agents forget old tasks. We add a replay buffer with prioritised sampling.",
    "Speech recognition degrades with accents. We fine-tune on accented data with adapters.",
    "Image captioning models hallucinate objects. We penalise captions naming absent objects.",
    "Sparse attention scales to long documents. We route tokens to experts by locality-sensitive hashing.",
    "Tabular data resists deep learning. We pretrain a transformer on synthetic tables.",
    "Machine translation for low-resource languages lacks data. We back-translate monolingual text.",
    "Protein structure prediction is costly. We distil a large model into a small one.",
    "Robots grasp unfamiliar objects poorly. We learn grasps from simulated point clouds.",
    "Recommender systems amplify popularity bias. We reweight the loss by item frequency.",
    "Code models struggle with long files. We add retrieval over the repository.",
]
RECORDS = [{"id": f"p{i:02d}", "text": t} for i, t in enumerate(ABSTRACTS)]


@pytest.fixture(scope="module")
def pipe():
    return GPT2TextGenerationPipeline.from_pretrained(device="cpu")


def test_frozen_perplexity_beats_the_unigram_floor(pipe):
    metrics = pipe.evaluate(RECORDS[8:])
    floor = pipe.unigram_baseline(RECORDS[:8], RECORDS[8:])
    assert (
        metrics["n_records"] == 4 and metrics["adapted"] is False and metrics["n_tokens"] == floor["n_tokens"]
    )
    assert 1.0 < metrics["perplexity"] < floor["perplexity"]


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    before = pipe.evaluate(RECORDS[:8])["perplexity"]
    result = pipe.adapt(RECORDS[:8], RECORDS[8:11], epochs=1, trainable_blocks=1, batch_size=4)
    assert result["n_trainable"] == 7_087_872 and result["history"][0]["note"] == "frozen model"
    assert all(name.startswith("transformer.h.11.") for name in result["trainable_names"])
    assert result["n_total"] == 124_439_808 and "perplexity" in result["history"][1]["val"]
    assert pipe.evaluate(RECORDS[:8])["perplexity"] < before  # the trained records got likelier
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = GPT2TextGenerationPipeline.from_artifact(artifact, device="cpu")
    assert reloaded.evaluate(RECORDS[:4])["perplexity"] == pytest.approx(
        pipe.evaluate(RECORDS[:4])["perplexity"]
    )
    a = pipe.generate(ABSTRACTS[0][:40], max_new_tokens=8)["text"]
    assert reloaded.generate(ABSTRACTS[0][:40], max_new_tokens=8)["text"] == a
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
