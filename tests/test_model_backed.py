"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): held-out
perplexity of the frozen model against the unigram floor, a one-epoch adaptation of the last transformer
block on a dozen abstracts, and the artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json

import pytest

from gpt2_text_generation_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, GPT2TextGenerationPipeline

pytest.importorskip("transformers")
torch = pytest.importorskip("torch")
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


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, tmp_path):
    """Without a validation split the recorded policy is "final epoch": the state after the last of three
    epochs is what stays in memory and what the artifact carries."""
    import torch

    result = pipe.adapt(RECORDS[:8], None, epochs=3, trainable_blocks=1, batch_size=4)
    assert result["best_epoch"] == 3 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 4
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = GPT2TextGenerationPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 3 and reloaded.adapter["trainable_blocks"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, tmp_path):
    """The loader derives the exact tensor set from the recorded layer count: a manifest listing fewer,
    more or other tensors — or one whose safetensors payload differs from its list — is refused."""
    import json as _json
    import shutil

    from safetensors.torch import load_file, save_file

    pipe.adapt(RECORDS[:8], None, epochs=1, trainable_blocks=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = _json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(_json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        GPT2TextGenerationPipeline.from_artifact(fewer, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    size = (extra / "adapter.safetensors").stat().st_size
    files = [{**manifest["files"][0], "bytes": size, "sha256": digest}]
    (extra / "manifest.json").write_text(_json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        GPT2TextGenerationPipeline.from_artifact(extra, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_blocks": 2}
    (other_layers / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        GPT2TextGenerationPipeline.from_artifact(other_layers, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    """A failure inside training leaves the base exactly as it was, frozen, with no adapter attached."""
    import torch

    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(RECORDS[:8], None, epochs=2, trainable_blocks=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())
