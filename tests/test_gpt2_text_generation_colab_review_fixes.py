"""Regression tests for the gpt2_text_generation_colab review fixes (GPT-M1..M3, GPT-m1..m5).

Most run within CI's install budget (pytest + numpy, no torch, no weights): the notebook's structure and text, the
split and BYOD contracts, and the notebook's own Section 4 cell executed with the carried package functions and a
stub pipeline. The rerun-validity tests (GPT-M3) need torch, transformers and safetensors and are skipped without
them; they wire a small randomly initialised 12-block GPT-2 (a stand-in, not the pinned weights) through
`GPT2TextGenerationPipeline.from_model`, the same closures `from_pretrained` uses. None of this is model-quality or
clean-runtime evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook source lines and refusal messages in full

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import gpt2_text_generation_pipeline as package
from gpt2_text_generation_pipeline import (
    TRANSFORMER_BLOCKS,
    GPT2TextGenerationPipeline,
    load_byod_dataset,
    min_split_records,
    split_dataset,
    validate_splits,
    write_dataset_csv,
)
from gpt2_text_generation_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "gpt2_text_generation_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, marker
    return found[0]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _learner_cells(nb: dict) -> list[str]:
    return [
        _src(c)
        for c in _code_cells(nb)
        if "# dimer: kernel cell" not in _src(c) and "embedded_module" not in c["metadata"].get("dimer", {})
    ]


def _docs(n: int, prefix: str = "doc") -> list[dict[str, str]]:
    return [{"id": f"{prefix}-{i:02d}", "text": f"Document {i} studies topic {i * 7} with method {i % 3}."} for i in range(n)]


# --- GPT-M1: isolated runtime, no in-kernel install, record corrected ------------------------------------------------


def test_exactly_two_kernel_cells_and_a_hash_locked_isolated_install(nb: dict) -> None:
    kernel = [_src(c) for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c)]
    assert len(kernel) == 2
    install = kernel[0]
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        assert needed in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in kernel[1]
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.0", "transformers==4.57.6", "tokenizers==0.22.2", "numpy==2.5.3", "safetensors==0.8.0", "huggingface-hub==0.36.2"):
        assert pin in lock
    assert "gpt2-text-generation-pipeline (pyproject.toml)" in lock and "ast-audio" not in lock and "esm2" not in lock


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """find_spec("google.colab") (accelerate does this) must not raise on the worker's stubs (fleet Colab failure)."""
    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    for name in ("google", "google.colab", "google.colab.files"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    shim_globals = {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None}
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), shim_globals)
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google.colab", "google.colab.files"):
            sys.modules.pop(name, None)


def test_release_record_no_longer_counts_the_restarted_run_as_a_pass() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "(1 restart after install cell)" not in text
    assert "an interpreter restart after the install is expected" not in text
    assert text.count("**Passed only after a manual restart** — not a one-pass Run all, not promotion evidence") == 2
    assert "`restarted: false`" in text and "**BYOD gate (REL12)**" in text
    assert "Current status: **Candidate" in (ROOT / "STATUS.md").read_text(encoding="utf-8")
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "MODEL_CARD.md"):
        body = (ROOT / name).read_text(encoding="utf-8")
        assert "**Release-grade** —" not in body and "(1 restart after install cell)" not in body, name


# --- GPT-M2: guided layer and infrastructure labels -------------------------------------------------------------------


def test_guided_layer_and_infrastructure_cells(nb: dict) -> None:
    md = _markdown(nb)
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 6),
        ("**What to notice:**", 6),
        ("<summary>Check your reasoning</summary>", 7),
        ("## 10. Your turn — change one thing", 1),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
    ):
        assert md.count(marker) >= least, marker
    titled = [c for c in _code_cells(nb) if _src(c).startswith("# @title Infrastructure:")]
    assert len(titled) == 7 and all(c["metadata"].get("cellView") == "form" for c in titled)
    # Sections 6 and 8 each ask for a prediction about the principal numbers before their code cell.
    cells = nb["cells"]
    for marker in ("frozen_test = pipe.evaluate(test_records)", "adapted_test = pipe.evaluate(test_records)"):
        index = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and marker in _src(c))
        assert "**Predict before running:**" in _src(cells[index - 1]), marker


# --- GPT-M3: every documented rerun measures and trains the pretrained model ------------------------------------------


def test_sections_4_to_7_reset_and_the_rerun_scope_is_named(nb: dict) -> None:
    for marker in ("USE_BYOD = False  # @param", "GREEDY_MAX_NEW_TOKENS = 32  # @param", "unigram = pipe.unigram_baseline(", "TRAINABLE_BLOCKS = 4  # @param"):
        cell = _cell(nb, marker)
        assert "pipe.reset_to_pretrained()" in cell, marker
    adapt = _cell(nb, "TRAINABLE_BLOCKS = 4  # @param")
    assert adapt.index("pipe.reset_to_pretrained()") < adapt.index("adapt_result = pipe.adapt(")
    assert "seed=ADAPT_SEED" in adapt and "start_check > 1e-4" in adapt
    md = _markdown(nb)
    assert "re-run from that cell" not in md and "watch the gain shrink" not in md
    experiments = md[md.index("**Next experiments**") : md.index("## Troubleshooting")]
    items = [line for line in experiments.splitlines() if line[:2] in ("1.", "2.", "3.", "4.")]
    assert len(items) == 4 and all("**Run after** from Section" in line for line in items)


def test_adapt_refuses_an_adapted_pipeline_before_importing_torch(forbid_model_imports) -> None:
    pipe = GPT2TextGenerationPipeline(lambda t: [1, 2, 3], lambda ids, s: [1], lambda ids: "x", "cpu", "injected")
    pipe.adapter = {"trainable_blocks": 4}
    with pytest.raises(ValueError, match="already holds an adaptation .* call reset_to_pretrained\\(\\) first"):
        pipe.adapt(_docs(8))
    assert pipe.reset_to_pretrained() == {"restored_tensors": 0, "adapted": False}
    assert pipe.adapter is None


class _ByteTokenizer:
    """One token per UTF-8 byte (ids 0..255), enough to drive the GPT-2 closures offline."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def _stand_in_pipeline(seed: int = 0) -> GPT2TextGenerationPipeline:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    torch.manual_seed(seed)
    config = transformers.GPT2Config(n_layer=TRANSFORMER_BLOCKS, n_head=2, n_embd=16, n_positions=1024, vocab_size=50257)
    model = transformers.GPT2LMHeadModel(config).eval()
    return GPT2TextGenerationPipeline.from_model(model, _ByteTokenizer(), device="cpu", source="stand-in")


def _state(pipe: GPT2TextGenerationPipeline) -> dict[str, Any]:
    return {k: v.detach().clone() for k, v in pipe._model.state_dict().items()}


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    import torch

    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


_TRAIN, _VAL = _docs(8, "train"), _docs(3, "val")


def test_reset_restores_the_pretrained_weights_and_a_second_run_starts_from_them() -> None:
    pipe = _stand_in_pipeline()
    pretrained = _state(pipe)
    first = pipe.adapt(_TRAIN, _VAL, epochs=1, lr=1e-3, trainable_blocks=4)
    assert not _same(_state(pipe), pretrained)
    with pytest.raises(ValueError, match="already holds an adaptation"):
        pipe.adapt(_TRAIN, _VAL, epochs=1, trainable_blocks=1)
    assert pipe.reset_to_pretrained() == {"restored_tensors": len(pipe._trainable_names(4)), "adapted": False}
    assert _same(_state(pipe), pretrained) and pipe.evaluate(_VAL)["adapted"] is False
    second = pipe.adapt(_TRAIN, _VAL, epochs=1, lr=1e-3, trainable_blocks=1)
    # Epoch 0 of the second run is the pretrained model, exactly as in the first run (the review's P4 saw the adapted one).
    assert second["history"][0]["val"]["perplexity"] == first["history"][0]["val"]["perplexity"]


def test_reload_parity_holds_after_switching_trainable_blocks(tmp_path: Path) -> None:
    pipe = _stand_in_pipeline()
    pipe.adapt(_TRAIN, _VAL, epochs=1, lr=1e-3, trainable_blocks=4)
    pipe.reset_to_pretrained()
    pipe.adapt(_TRAIN, _VAL, epochs=1, lr=1e-3, trainable_blocks=1)
    pipe.save_artifact(tmp_path / "adapter")
    manifest = json.loads((tmp_path / "adapter" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"]["trainable_blocks"] == 1 and all(t.startswith("transformer.h.11.") for t in manifest["tensors"])
    fresh = _stand_in_pipeline()  # the same seed builds the same "pretrained" base, as from_artifact reloads the snapshot
    fresh.load_artifact(tmp_path / "adapter")
    assert _same(_state(fresh), _state(pipe))
    assert fresh.evaluate(_VAL)["perplexity"] == pipe.evaluate(_VAL)["perplexity"]


def test_save_refuses_when_a_block_outside_the_adapter_differs_from_the_base(tmp_path: Path) -> None:
    pipe = _stand_in_pipeline()
    pipe.adapt(_TRAIN, _VAL, epochs=1, lr=1e-3, trainable_blocks=4)
    # Simulate the pre-fix state: the adapter names one block while four differ from the base.
    pipe.adapter["trainable_names"] = pipe._trainable_names(1)
    with pytest.raises(ValueError, match="tensors outside this adapter differ from the pretrained base"):
        pipe.save_artifact(tmp_path / "adapter")


def test_load_artifact_overlays_onto_the_base_not_onto_an_earlier_adaptation(tmp_path: Path) -> None:
    source = _stand_in_pipeline()
    source.adapt(_TRAIN, _VAL, epochs=1, lr=1e-3, trainable_blocks=1)
    source.save_artifact(tmp_path / "one")
    target = _stand_in_pipeline()
    target.adapt(_TRAIN, _VAL, epochs=1, lr=1e-3, trainable_blocks=4)
    target.load_artifact(tmp_path / "one")
    assert _same(_state(target), _state(source))
    target.reset_to_pretrained()
    assert _same(_state(target), _state(_stand_in_pipeline()))


# --- GPT-m1: BYOD minimum, path field, actionable refusals --------------------------------------------------------------


def test_the_stated_minimum_is_the_one_the_split_applies(forbid_model_imports) -> None:
    assert min_split_records() == 12
    with pytest.raises(ValueError, match="Supply at least 12 unique records"):
        split_dataset(_docs(11))
    splits = split_dataset(_docs(12))
    assert {k: len(v) for k, v in splits.items()} == {"test": 2, "validation": 2, "train": 8}
    manifests = validate_splits(splits)
    assert {k: m["n_records"] for k, m in manifests.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match="^train split: 7 records"):
        validate_splits({"train": _docs(7), "validation": _docs(1, "v"), "test": _docs(1, "t")})
    with pytest.raises(ValueError, match="^test split: 0 records"):
        validate_splits({"train": _docs(8), "validation": _docs(1, "v"), "test": []})


def test_byod_loader_refuses_non_utf8_and_unknown_types(tmp_path: Path, forbid_model_imports) -> None:
    latin = tmp_path / "docs.csv"
    latin.write_bytes("id,text\nx1,caf\xe9 au lait\n".encode("latin-1"))
    with pytest.raises(ValueError, match="is not UTF-8 text"):
        load_byod_dataset(latin)
    other = tmp_path / "docs.xlsx"
    other.write_bytes(b"PK")
    with pytest.raises(ValueError, match="unsupported BYOD file type '.xlsx'"):
        load_byod_dataset(other)


class _StubPipe:
    def __init__(self) -> None:
        self.resets = 0

    def reset_to_pretrained(self) -> dict[str, Any]:
        self.resets += 1
        return {"restored_tensors": 0, "adapted": False}


def _section4_namespace(tmp_path: Path) -> dict[str, Any]:
    ns: dict[str, Any] = {name: getattr(package, name) for name in package.__all__}
    ns.update({name: getattr(sm, name) for name in ("CORPUS_RELEASE", "read_corpus", "fetch_corpus", "filter_records")})
    ns.update({"pipe": _StubPipe(), "__name__": "__main__"})
    return ns


def _run_section4(nb: dict, ns: dict[str, Any], *, use_byod: bool, path: str) -> str:
    source = _cell(nb, "USE_BYOD = False  # @param")
    if use_byod:
        source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param")
    source = source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "section4", "exec"), ns)
    return out.getvalue()


def test_section4_byod_path_runs_outside_colab_at_the_stated_minimum(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)  # not Colab: the path field must not need it
    write_dataset_csv(_docs(12), tmp_path / "mine.csv")
    ns = _section4_namespace(tmp_path)
    printed = _run_section4(nb, ns, use_byod=True, path=str(tmp_path / "mine.csv"))
    assert ns["data_source"] == "BYOD (mine.csv)" and ns["pipe"].resets == 1
    assert {k: m["n_records"] for k, m in ns["dataset_manifests"].items()} == {"test": 2, "validation": 2, "train": 8}
    assert "'too small'" in printed and (tmp_path / "outputs" / "gpt2_text_generation_train.csv").is_file()


def test_section4_without_a_path_or_with_a_cancelled_upload_names_the_next_action(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _section4_namespace(tmp_path)
    with pytest.raises(RuntimeError, match="set BYOD_PATH"):
        _run_section4(nb, ns, use_byod=True, path="")
    google, colab, files = types.ModuleType("google"), types.ModuleType("google.colab"), types.ModuleType("google.colab.files")
    google.__path__, colab.__path__ = [], []
    files.upload = lambda: {}
    colab.files, google.colab = files, colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ValueError, match=r"Upload exactly one .* \(got 0; an empty or cancelled upload gives 0\)"):
        _run_section4(nb, ns, use_byod=True, path="")
    md = _markdown(nb)
    assert "at least **12 unique records**" in md and "8..20,000 records" not in md


# --- GPT-m2: recorded outcomes, no bare assertions ----------------------------------------------------------------------


def test_no_bare_assert_and_the_comparisons_are_recorded(nb: dict) -> None:
    for source in _learner_cells(nb):
        bare = [n.lineno for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assert) and n.msg is None]
        assert not bare, source[:80]
    floor = _cell(nb, "unigram = pipe.unigram_baseline(")
    assert "frozen_test['perplexity'] < unigram['perplexity']" in floor and "floor_outcome =" in floor
    evaluation = _cell(nb, "adapted_test = pipe.evaluate(test_records)")
    assert "adaptation_outcome = " in evaluation and "'adaptation_outcome': adaptation_outcome" in evaluation
    assert "assert adapted_test" not in evaluation


# --- GPT-m3, GPT-m4, GPT-m5: text ------------------------------------------------------------------------------------


def test_no_template_artefacts_and_the_prompt_is_printed_whole(nb: dict) -> None:
    md = _markdown(nb)
    assert "{{" not in md and "}}" not in md
    assert f"ids matching `{sm._ID_RE.pattern}`" in md
    export = _cell(nb, "pipe.save_artifact(artifact_dir")
    assert "head[-60:]" not in export and "print({'id': record['id'], 'prompt': head})" in export


def test_every_time_names_its_run(nb: dict) -> None:
    md = _markdown(nb)
    for stale in ("the build record", "The build record", "about 114 s", "about ten seconds on CPU", "about five minutes of model time"):
        assert stale not in md, stale
    assert "local Windows CPU pre-flight took 285.5 s" in md and "Kaggle Tesla T4 run took 251.2 s" in md


def test_pretraining_overlap_is_stated_where_the_pretrained_perplexity_is_first_read(nb: dict) -> None:
    cells = nb["cells"]
    index = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "frozen_test = pipe.evaluate(test_records)" in _src(c))
    section6 = _src(cells[index - 1])
    assert "**Pretraining overlap.**" in section6 and "cannot be ruled out" in section6
    assert "may contain some of these abstracts" in _markdown(nb)
