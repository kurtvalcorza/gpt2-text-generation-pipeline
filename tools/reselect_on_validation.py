"""Run the pre-registered re-selection of 2026-10-17 (docs/reselection-2026-10-17.md) and record it.

Phase 1 trains the three grid configurations on the validation split only and writes the selection
record (``--selection-out``) before any test record is scored. Phase 2 then trains the selected configuration
once and
scores it once on the pinned never-scored test draw, and once on the exposed development test (the 100
abstracts the 2026-09 sweep scored), writing ``--test-out``. Needs torch, transformers, safetensors and the
digest-verified GPT-2 snapshot; it is not part of CI.

    python tools/reselect_on_validation.py --weights-dir weights/gpt2 --corpus-cache weights/scitldr
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gpt2_text_generation_pipeline import pipeline as gp  # noqa: E402
from gpt2_text_generation_pipeline import samples  # noqa: E402

LABEL = "CPU build record (tools/reselect_on_validation.py), not a hosted run; pending the hosted run (REL13)"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"wrote {path}")


def _score(metrics: dict) -> dict:
    keys = ("perplexity", "bits_per_token", "mean_nll", "n_records", "n_tokens", "record_perplexity")
    keys += ("adapted",)
    return {k: metrics[k] for k in keys}


def _opening(record: dict, chars: int = 120) -> str:
    head = record["text"][:chars]
    return head.rsplit(" ", 1)[0] if " " in head else head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights-dir", type=Path, default=REPO / "weights" / "gpt2")
    parser.add_argument("--corpus-cache", type=Path, default=REPO / "weights" / "scitldr")
    parser.add_argument(
        "--selection-out", type=Path, default=REPO / "docs/reselection/validation_selection_2026-10-17.json"
    )
    parser.add_argument("--test-out", type=Path, default=REPO / "docs/reselection/fresh_test_2026-10-17.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    import torch
    import transformers

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
    }
    corpus = samples.read_corpus(samples.fetch_corpus(cache_dir=args.corpus_cache))
    splits = samples.build_tutorial_dataset(corpus, seed=samples.SAMPLE_SEED)
    digests = {name: samples.dataset_digest(part) for name, part in splits.items()}
    pipe = gp.GPT2TextGenerationPipeline.from_pretrained(device=args.device, weights_dir=args.weights_dir)
    train, val = splits["train"], splits["validation"]
    pretrained_val = pipe.evaluate(val)

    # Phase 1: validation only.
    def progress(name: str, entry: dict) -> None:
        val_ppl = None if entry["val"] is None else round(entry["val"]["perplexity"], 3)
        print({"config": name, "epoch": entry["epoch"], "val_perplexity": val_ppl}, flush=True)

    started = time.perf_counter()
    reselection = gp.reselect_on_validation(pipe, train, val, progress=progress)
    record = {
        "label": LABEL,
        "date": time.strftime("%Y-%m-%d"),
        "preregistration": "docs/reselection-2026-10-17.md section 1",
        "runtime": runtime,
        "dataset_digests": {k: digests[k] for k in ("train", "validation")},
        "pretrained_validation_perplexity": pretrained_val["perplexity"],
        **reselection,
        "grid_seconds": round(time.perf_counter() - started, 1),
        "test_records_scored": 0,
    }
    _write(args.selection_out, record)
    chosen = reselection["selection"]["config"]
    print({"selected": reselection["selection"]["selected"], "reason": reselection["selection"]["reason"]})

    # Phase 2: the selected configuration, scored once on the fresh test and once on the exposed test.
    test, exposed = splits["test"], splits["exposed_test"]
    unigram = pipe.unigram_baseline(train, test)
    pretrained_test = pipe.evaluate(test)
    pretrained_exposed = pipe.evaluate(exposed)
    used = {r["text"].lower() for part in splits.values() for r in part}
    unseen = [r for r in samples.filter_records(corpus["dev"]) if r["text"].lower() not in used][:3]
    before = [pipe.generate(_opening(r), max_new_tokens=32)["completion"] for r in unseen]
    t0 = time.perf_counter()
    adapt = pipe.adapt(
        train,
        val,
        epochs=2,
        lr=chosen["lr"],
        batch_size=8,
        trainable_blocks=chosen["trainable_blocks"],
        seed=0,
    )
    adapt_seconds = round(time.perf_counter() - t0, 1)
    grid_row = next(r for r in reselection["grid"] if r["name"] == chosen["name"])
    curve = [None if h["val"] is None else h["val"]["perplexity"] for h in adapt["history"]]
    adapted_test = pipe.evaluate(test)
    adapted_exposed = pipe.evaluate(exposed)
    after = [pipe.generate(_opening(r), max_new_tokens=32)["completion"] for r in unseen]
    tmp = Path(tempfile.mkdtemp(prefix="gpt2-reselect-"))
    try:
        art = pipe.save_artifact(tmp / "adapter", metadata={"tool": "reselect_on_validation"})
        manifest = json.loads((Path(art) / "manifest.json").read_text(encoding="utf-8"))
        reloaded = gp.GPT2TextGenerationPipeline.from_artifact(
            Path(art), weights_dir=args.weights_dir, device=args.device
        )
        parity_pair = (pipe.evaluate(val[:10])["perplexity"], reloaded.evaluate(val[:10])["perplexity"])
        parity_gen = sum(
            reloaded.generate(_opening(r), max_new_tokens=32)["completion"] == a
            for r, a in zip(unseen, after, strict=True)
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _write(
        args.test_out,
        {
            "label": LABEL,
            "date": time.strftime("%Y-%m-%d"),
            "runtime": runtime,
            "selected": reselection["selection"]["selected"],
            "config": chosen,
            "dataset_digests": digests,
            "adaptation": {
                "best_epoch": adapt["best_epoch"],
                "val_perplexity_by_epoch": curve,
                "matches_grid_run": curve == grid_row["val_perplexity_by_epoch"],
                "n_trainable": adapt["n_trainable"],
                "n_train_tokens": adapt["n_train_tokens"],
                "seconds": adapt_seconds,
            },
            "unigram_floor_fresh_test": {k: unigram[k] for k in ("perplexity", "bits_per_token", "n_tokens")},
            "fresh_test": {"pretrained": _score(pretrained_test), "adapted": _score(adapted_test)},
            "exposed_development_test": {
                "pretrained": _score(pretrained_exposed),
                "adapted": _score(adapted_exposed),
            },
            "completions_changed": sum(a != b for a, b in zip(before, after, strict=True)),
            "completions_of": len(unseen),
            "artifact": {"tensors": len(manifest["tensors"]), "bytes": manifest["files"][0]["bytes"]},
            "reload_parity_validation10": {
                "in_memory": parity_pair[0],
                "reloaded": parity_pair[1],
                "identical_completions": parity_gen,
                "of": len(unseen),
            },
            "pretrained_validation_perplexity": pretrained_val["perplexity"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
