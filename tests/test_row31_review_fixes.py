"""Regressions for the row 31 review of the tutorial (2026-10-17, T31-1..T31-8).

T31-1 the pre-registered validation re-selection and the never-scored test draw; T31-2 run-neutral run-status
wording; T31-3 spec 2.5 declarations; T31-4 friendly BYOD refusals; T31-5 DIMER_ISOLATED_ENV; T31-6 the
provenance stamp names a commit holding the carried modules. NumPy-free and torch-free, like CI.
"""

# ruff: noqa: E501  -- long assertion strings are kept on one line, as in the other notebook tests
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import gpt2_text_generation_pipeline as package
from gpt2_text_generation_pipeline import pipeline as gp
from gpt2_text_generation_pipeline import samples as sm
from gpt2_text_generation_pipeline.metrics import RESELECTION_GRID, select_configuration

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "gpt2_text_generation_colab.ipynb"
sys.path.insert(0, str(ROOT / "tools"))
import build_notebook as build  # noqa: E402


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _index(nb: dict, marker: str, kind: str = "code") -> int:
    found = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == kind and marker in _src(c)]
    assert len(found) == 1, (marker, found)
    return found[0]


# --- T31-1: the selection rule ----------------------------------------------------------------------------------------


def test_the_incumbent_stays_unless_a_challenger_is_two_percent_better() -> None:
    assert select_configuration({"C0": 35.0, "C1": 34.4, "C2": 36.0})["selected"] == "C0"  # 1.7 %: not enough
    picked = select_configuration({"C0": 35.0, "C1": 34.3, "C2": 36.0})  # exactly 2 %
    assert picked["selected"] == "C1" and picked["config"]["trainable_blocks"] == 2


def test_two_challengers_within_the_margin_go_to_the_smaller_one() -> None:
    # Both clear 2 %; C2 is lower but within 2 % of C1 -> the smaller config (2 blocks, lower lr) = C2.
    assert select_configuration({"C0": 36.0, "C1": 34.0, "C2": 33.9})["selected"] == "C2"
    # C1 is lower and C2 within 2 % of it: the smaller configuration still wins.
    assert select_configuration({"C0": 36.0, "C1": 33.9, "C2": 34.0})["selected"] == "C2"
    # Far apart: the lower one wins.
    assert select_configuration({"C0": 36.0, "C1": 32.0, "C2": 34.5})["selected"] == "C1"


def test_the_rule_refuses_missing_or_invalid_scores() -> None:
    with pytest.raises(ValueError, match="exactly"):
        select_configuration({"C0": 35.0, "C1": 34.0})
    with pytest.raises(ValueError, match="positive and finite"):
        select_configuration({"C0": 35.0, "C1": float("nan"), "C2": 34.0})


def test_the_grid_is_the_preregistered_one() -> None:
    assert [(g["name"], g["trainable_blocks"], g["lr"]) for g in RESELECTION_GRID] == [
        ("C0", 4, 1e-4),
        ("C1", 2, 1e-4),
        ("C2", 2, 5e-5),
    ]
    prereg = (ROOT / "docs" / "reselection-2026-10-17.md").read_text(encoding="utf-8")
    for text in ("| C0 | 4 | 1e-4 |", "| C1 | 2 | 1e-4 |", "| C2 | 2 | 5e-5 |", "at least 2 %", sm.FRESH_TEST_DIGEST):
        assert text in prereg, text


class _GridPipe:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def reset_to_pretrained(self) -> dict:
        self.calls.append("reset")
        return {}

    def adapt(self, train, val, **kw):
        self.calls.append(("adapt", len(train), len(val), kw["trainable_blocks"], kw["lr"], kw["epochs"]))
        final = {4: 35.0, 2: 35.5}[kw["trainable_blocks"]] + (0.3 if kw["lr"] < 1e-4 else 0.0)
        history = [{"epoch": e, "val": {"perplexity": p}} for e, p in enumerate((41.0, 37.0, final))]
        self.last = final
        return {"best_epoch": 2, "history": history, "n_trainable": kw["trainable_blocks"]}

    def evaluate(self, records):
        return {"records": {f"v{i}": {"mean_nll": math.log(self.last) + i / 1000, "n_tokens": 10} for i in range(len(records))}}


def test_reselect_on_validation_trains_every_config_from_pretrained_on_validation_only() -> None:
    pipe = _GridPipe()
    out = gp.reselect_on_validation(pipe, [{}] * 8, [{}] * 3)
    adapts = [c for c in pipe.calls if c != "reset"]
    assert adapts == [("adapt", 8, 3, 4, 1e-4, 2), ("adapt", 8, 3, 2, 1e-4, 2), ("adapt", 8, 3, 2, 5e-5, 2)]
    assert pipe.calls[0] == "reset" and pipe.calls[-1] == "reset" and pipe.calls.count("reset") == 4
    assert out["selection"]["selected"] == "C0" and [r["best_val_perplexity"] for r in out["grid"]] == [35.0, 35.5, 35.8]


# --- T31-1: the never-scored test draw --------------------------------------------------------------------------------


def _corpus(n: int = 40) -> dict[str, list[dict[str, str]]]:
    def member(name: str, k: int) -> list[dict[str, str]]:
        return [{"paper_id": f"{name}{i}", "text": f"{name} abstract {i} " + " ".join(f"w{name}{i}x{j}" for j in range(40))} for i in range(k)]

    corpus = {"train": member("tr", n), "dev": member("dv", n), "test": member("te", n)}
    corpus["test"][5]["text"] = corpus["train"][3]["text"]  # a text shared with another member
    return corpus


def test_fresh_draw_excludes_the_exposed_draw_and_cross_member_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "SAMPLE_SPLIT", {"train": 10, "validation": 5, "test": 10})
    corpus = _corpus()
    exposed = sm.build_sample_dataset(corpus, seed=sm.SAMPLE_SEED)["test"]
    fresh = sm.build_fresh_test(corpus, size=20, verify=False)
    texts = {r["text"] for r in fresh}
    assert not texts & {r["text"] for r in exposed}
    assert corpus["train"][3]["text"] not in texts and len(fresh) == 20
    pool = [r for r in sm.filter_records(corpus["test"]) if r["text"] not in {x["text"] for x in exposed}]
    pool = [r for r in pool if r["paper_id"] != "te5"]
    random.Random(sm.FRESH_TEST_SEED).shuffle(pool)
    assert [r["paper_id"] for r in fresh] == [r["paper_id"] for r in pool[:20]]
    with pytest.raises(ValueError, match="not the pinned one"):
        sm.build_fresh_test(corpus, size=20)


def test_tutorial_dataset_is_paper_disjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "SAMPLE_SPLIT", {"train": 10, "validation": 5, "test": 10})
    corpus = _corpus(150)
    fresh = sm.build_fresh_test(corpus, verify=False)
    monkeypatch.setattr(sm, "FRESH_TEST_PAPER_IDS", tuple(r["paper_id"] for r in fresh))
    monkeypatch.setattr(sm, "FRESH_TEST_DIGEST", sm.dataset_digest(fresh))
    splits = sm.build_tutorial_dataset(corpus)
    assert {k: len(v) for k, v in splits.items()} == {"train": 10, "validation": 5, "test": 100, "exposed_test": 10}
    assert sm.check_paper_disjoint(splits)["test"] == 100
    with pytest.raises(ValueError, match="appears in both"):
        sm.check_paper_disjoint({"a": [{"paper_id": "x"}], "b": [{"paper_id": "x"}]})


def test_the_pin_is_100_distinct_papers() -> None:
    assert len(sm.FRESH_TEST_PAPER_IDS) == sm.FRESH_TEST_SIZE == 100 == len(set(sm.FRESH_TEST_PAPER_IDS))
    assert len(sm.FRESH_TEST_DIGEST) == 64


# --- T31-1: notebook order and test-once ------------------------------------------------------------------------------


def test_preregistration_precedes_every_score_and_independence_is_not_overclaimed(nb: dict) -> None:
    prereg = _index(nb, "**Pre-registered evaluation protocol (2026-10-17).**", "markdown")
    first_score = _index(nb, "frozen_test = pipe.evaluate(test_records)")
    assert prereg < first_score
    grid = _index(nb, "reselection = reselect_on_validation(")
    adapt = _index(nb, "adapt_result = pipe.adapt(")
    evaluate = _index(nb, "adapted_test = pipe.evaluate(test_records) if preregistered else None")
    assert first_score < grid < adapt < evaluate
    grid_src = _src(nb["cells"][grid])
    assert "test_records" not in grid_src and "RESELECT_ON_VALIDATION = True  # @param" in grid_src
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for stale in ("on an independent split", "test-split perplexities", "test split in view"):
        assert stale not in md, stale
    assert "parity_records = val_records[:10]" in _src(nb["cells"][_index(nb, "pipe.save_artifact(artifact_dir")])


def test_the_build_record_exists_and_names_its_selection() -> None:
    record = json.loads((ROOT / "docs/reselection/validation_selection_2026-10-17.json").read_text(encoding="utf-8"))
    assert record["test_records_scored"] == 0 and [r["name"] for r in record["grid"]] == ["C0", "C1", "C2"]
    assert record["selection"] == select_configuration({r["name"]: r["best_val_perplexity"] for r in record["grid"]})
    fresh = json.loads((ROOT / "docs/reselection/fresh_test_2026-10-17.json").read_text(encoding="utf-8"))
    assert fresh["selected"] == record["selection"]["selected"]
    assert fresh["dataset_digests"]["test"] == sm.FRESH_TEST_DIGEST
    template = (ROOT / "tools" / "notebook_template.py").read_text(encoding="utf-8")
    assert f"RECORDED_SELECTION = '{record['selection']['selected']}'" in template


# --- T31-4: friendly BYOD refusals ------------------------------------------------------------------------------------


class _StubPipe:
    def reset_to_pretrained(self) -> dict:
        return {}


def _run_section4(nb: dict, ns: dict, path: str) -> str:
    source = _src(nb["cells"][_index(nb, "USE_BYOD = False  # @param")])
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param")
    source = source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "section4", "exec"), ns)
    return out.getvalue()


def _ns() -> dict:
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update({name: getattr(sm, name) for name in ("CORPUS_RELEASE", "read_corpus", "fetch_corpus", "filter_records")})
    ns.update({"pipe": _StubPipe(), "__name__": "__main__", "adapt_result": {"stale": True}, "splits": {"stale": []}})
    return ns


def test_a_refused_file_is_a_message_that_keeps_outputs_and_ends_the_run(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "earlier.json").write_text("{}", encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_text("id,text\na,hello\n", encoding="utf-8")  # far below the 12-record minimum
    ns = _ns()
    with pytest.raises(sm.DatasetRefused, match=r"^BYOD refused: .*kept") as caught:
        _run_section4(nb, ns, str(bad))
    assert caught.value._render_traceback_() == [str(caught.value)]
    assert "adapt_result" not in ns and "splits" not in ns and (tmp_path / "outputs" / "earlier.json").is_file()
    with pytest.raises(sm.DatasetRefused, match="BYOD refused"):
        _run_section4(nb, _ns(), str(tmp_path / "missing.csv"))  # OSError path


def test_an_unexpected_error_keeps_its_traceback(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _ns()

    def broken(path):
        raise KeyError("a bug, not a refusal")

    ns["load_byod_dataset"] = broken
    with pytest.raises(KeyError):
        _run_section4(nb, ns, str(tmp_path / "x.csv"))


def test_the_isolated_runtime_forwards_a_refusal_without_a_traceback() -> None:
    source = Path(build.__file__).read_text(encoding="utf-8")
    assert 'getattr(exc, "dimer_refusal", False)' in source and '_send.send(("refusal", str(exc)))' in source
    assert "class IsolatedCellRefusal(RuntimeError)" in source and 'elif kind == "refusal"' in source


# --- T31-2, T31-3, T31-5 ----------------------------------------------------------------------------------------------


def test_declarations_use_notice_and_environment_override(nb: dict) -> None:
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    notice = _index(nb, "> **⚠️ Notice.** For research, training and evaluation only.", "markdown")
    assert notice < first_code
    head = _src(nb["cells"][0])
    assert "without warranty of any kind" in head and "no hosted run of this version is recorded yet" not in head
    assert "`docs/release-verification.md`" in head and "pending a hosted run (REL13)" in head
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "`DIMER_ISOLATED_ENV`" in md and "[1] A. Radford" in md and "within 1e-6" in md


# --- T31-6: the stamped revision holds the carried modules ------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)


def test_the_stamped_revision_contains_every_carried_module(nb: dict) -> None:
    try:
        shallow = _git("rev-parse", "--is-shallow-repository")
    except OSError:
        pytest.skip("git is not available")
    if shallow.returncode != 0:
        pytest.skip("not a git checkout")
    if shallow.stdout.strip() == b"true":
        pytest.skip("shallow clone: the stamped revision may be absent (CI fetches full history)")
    revision = nb["metadata"]["dimer"]["generated_from"]["revision"]
    carried = [c["metadata"]["dimer"] for c in nb["cells"] if "embedded_module" in c.get("metadata", {}).get("dimer", {})]
    assert len(carried) == 3
    for meta in carried:
        blob = _git("show", f"{revision}:{meta['embedded_module']}")
        assert blob.returncode == 0, (revision, meta["embedded_module"])
        assert hashlib.sha256(blob.stdout).hexdigest() == meta["module_sha256"], meta["embedded_module"]


def test_the_generator_refuses_a_revision_without_the_carried_bytes(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="does not hold the carried bytes"):
        build.check_revision_holds_modules(ROOT, "HEAD", {"src/gpt2_text_generation_pipeline/pipeline.py": "0" * 64})


# --- Round 2 (peer review G-M1, G-m4, G-m5, G-m7, G-s2, G-s3) ---------------------------------------------------------


def test_paired_delta_reads_each_record_against_itself() -> None:
    from gpt2_text_generation_pipeline.metrics import paired_delta

    before = {f"r{i}": {"mean_nll": 3.0 + i / 10, "n_tokens": 10 + i} for i in range(20)}
    after = {k: {"mean_nll": v["mean_nll"] - (0.1 if i % 4 else -0.05), "n_tokens": v["n_tokens"]} for i, (k, v) in enumerate(before.items())}
    out = paired_delta(before, after, n_boot=500, seed=1)
    assert out["n"] == 20 and out["n_improved"] == 15
    assert out["bootstrap_95ci"][0] <= out["token_weighted_delta_bits"] <= out["bootstrap_95ci"][1] < 0
    assert out == paired_delta(before, after, n_boot=500, seed=1)  # fixed seed: reproducible
    with pytest.raises(ValueError, match="same, non-empty set"):
        paired_delta(before, {"x": {"mean_nll": 1.0, "n_tokens": 3}})
    with pytest.raises(ValueError, match="different tokens"):
        paired_delta({"a": {"mean_nll": 1.0, "n_tokens": 3}}, {"a": {"mean_nll": 1.0, "n_tokens": 4}})


def test_the_record_median_is_the_true_median_and_alpha_smooths_the_unigram() -> None:
    from gpt2_text_generation_pipeline.metrics import sequence_metrics, unigram_baseline

    out = sequence_metrics([[1.0], [2.0], [3.0], [4.0]])
    assert out["record_perplexity"]["median"] == pytest.approx((math.exp(2.0) + math.exp(3.0)) / 2)
    train, test = [[1, 2, 2, 3]], [[0, 2, 9]]
    assert unigram_baseline(train, test, 50)["perplexity"] > unigram_baseline(train, test, 50, alpha=0.1)["perplexity"]


def test_near_duplicates_across_splits_are_refused() -> None:
    text = "we propose a new method for training deep networks with sparse gradients and show strong results on benchmarks"
    edited = text.replace("strong", "robust")
    with pytest.raises(ValueError, match="near-duplicate"):
        sm.check_near_duplicates({"train": [{"text": text}], "test": [{"text": edited}]})
    assert sm.check_near_duplicates({"train": [{"text": text}], "test": [{"text": "an unrelated abstract " * 5}]})["max_containment"] == 0


def test_selection_is_wired_to_training_and_the_prompt_is_not_a_test_record(nb: dict) -> None:
    grid = _src(nb["cells"][_index(nb, "reselection = reselect_on_validation(")])
    assert "replayed = reselection is not None and (USE_BYOD or SPLIT_SEED != 42)" in grid
    adapt = _src(nb["cells"][_index(nb, "adapt_result = pipe.adapt(")])
    assert adapt.index("TRAINABLE_BLOCKS, LEARNING_RATE = selected_config['trainable_blocks'], selected_config['lr']") < adapt.index("adapt_result = pipe.adapt(")
    evaluate = _src(nb["cells"][_index(nb, "adapted_test = pipe.evaluate(test_records) if preregistered else None")])
    assert "paired = paired_delta(before_m['records'], after_m['records'])" in evaluate
    generate = _src(nb["cells"][_index(nb, "GREEDY_MAX_NEW_TOKENS = 32  # @param")])
    assert "prompt = opening(val_records[0])" in generate and "test_records[0]" not in generate
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for stale in ("never-scored", "no earlier run scored", "spread across records you saw in Section 8?", "most abstracts got easier, not just the average"):
        assert stale not in md, stale
    for needed in ("**predeclared practical tolerance**", "at this budget only", "**Scoring convention.**", "paired_per_record"):
        assert needed in md, needed


# --- Score-once rule (SCORE_ONCE_RULE.md, 2026-10-17) -------------------------------------------------------------------


def _preregistered() -> dict:
    from gpt2_text_generation_pipeline.metrics import PREREGISTERED_SETTINGS

    return json.loads(json.dumps(PREREGISTERED_SETTINGS))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("epochs", 4),
        ("batch_size", 4),
        ("adapt_seed", 1),
        ("learning_rate", 5e-5),
        ("data_source", "BYOD"),
        ("split_seed", 7),
        ("trainable_blocks", 2),
        ("max_train_tokens", 256),
        ("configuration", "C1"),
    ],
)
def test_any_changed_setting_makes_the_run_exploratory(name: str, value: object) -> None:
    from gpt2_text_generation_pipeline.metrics import preregistration_diff, test_look

    settings = _preregistered()
    settings[name] = value
    assert preregistration_diff(settings) == [name]
    assert test_look(preregistration_diff(settings), 0) == "not scored (exploratory)"
    assert test_look(preregistration_diff(settings), 3) == "not scored (exploratory)"


def test_a_changed_dataset_digest_or_a_missing_setting_is_exploratory() -> None:
    from gpt2_text_generation_pipeline.metrics import preregistration_diff

    settings = _preregistered()
    settings["dataset_digests"]["test"] = "0" * 64
    assert preregistration_diff(settings) == ["dataset_digests"]
    settings = _preregistered()
    del settings["batch_size"]
    assert preregistration_diff(settings) == ["batch_size"]


def test_an_identical_rerun_is_a_reproduction() -> None:
    from gpt2_text_generation_pipeline.metrics import preregistration_diff, test_look

    differs = preregistration_diff(_preregistered())
    assert differs == []
    assert test_look(differs, 0) == "pre-registered evaluation"
    assert test_look(differs, 1) == "reproduction"
    assert test_look(differs, 5) == "reproduction"


def test_the_notebook_gates_the_test_split_on_every_preregistered_setting(nb: dict) -> None:
    train = _src(nb["cells"][_index(nb, "adapt_result = pipe.adapt(")])
    assert "differs = preregistration_diff(settings)" in train and "preregistered = not differs" in train
    for field in ("'epochs': EPOCHS", "'batch_size': BATCH_SIZE", "'learning_rate': LEARNING_RATE", "'adapt_seed': ADAPT_SEED",
                  "'split_seed': SPLIT_SEED", "'data_source': 'BYOD' if USE_BYOD else 'sample'", "'dataset_digests'"):
        assert field in train, field
    evaluate = _src(nb["cells"][_index(nb, "adapted_test = pipe.evaluate(test_records) if preregistered else None")])
    assert "look = test_look(differs," in evaluate
    assert "_exploratory_" in evaluate  # exploratory passes never overwrite the pre-registered report
    floor = _src(nb["cells"][_index(nb, "data_preregistered = not USE_BYOD and SPLIT_SEED == 42")])
    assert "frozen_test = pipe.evaluate(test_records) if data_preregistered else None" in floor
    every = "\n".join(_src(c) for c in nb["cells"])
    assert "is_selected_run" not in every
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "scored once" not in md and "**only under the exact pre-registered settings**" in md
