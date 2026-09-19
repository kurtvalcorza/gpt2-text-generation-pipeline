"""Causal text generation over the pinned ``openai-community/gpt2`` (124M) checkpoint.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``generate``: greedy decoding by default
(deterministic), nucleus sampling only when asked for and seeded. One prompt per call.

The adaptation contract (``evaluate``, ``unigram_baseline``, ``adapt``, ``save_artifact``, ``from_artifact``)
scores a validated ``{id, text}`` corpus by teacher-forced perplexity, fine-tunes the last transformer blocks
on it with validation-perplexity epoch selection, and exports the trained tensors as a safetensors adapter
bound to the pinned base weights. The inference contract above is unchanged by it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MODEL_LICENSE = "mit"
MODEL_KEY = "gpt2"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Ceilings. CONTEXT_LENGTH is n_positions / n_ctx in the pinned config.json; prompt tokens plus new tokens
# must fit in it, so a prompt is rejected (never truncated) above MAX_PROMPT_TOKENS and the combined length
# is checked before the model runs. MAX_NEW_TOKENS bounds one call's cost on CPU.
CONTEXT_LENGTH = 1024
MAX_PROMPT_TOKENS = CONTEXT_LENGTH - 1  # leaves room for at least one generated token
MAX_NEW_TOKENS = 256
DEFAULT_MAX_NEW_TOKENS = 32
MAX_TEXT_CHARS = 4_000  # pre-tokenisation guard; ~4 chars per byte-level BPE token on English text
VOCAB_SIZE = 50257  # config.json vocab_size
EOS_TOKEN_ID = 50256  # config.json eos_token_id == bos_token_id; GPT-2 has no pad token, so pad = eos
PAD_TOKEN_ID = EOS_TOKEN_ID
DECODING_DEFAULT = "greedy"  # do_sample=False -> argmax over the next-token distribution at every step
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 124_439_808
TRANSFORMER_BLOCKS = 12  # config.json n_layer
DEFAULT_TRAINABLE_BLOCKS = 4  # the last four transformer blocks (28,351,488 parameters)
MAX_TRAIN_TOKENS = 512  # text truncation ceiling during adaptation (scoring never truncates — it rejects)
MAX_EVAL_RECORDS = 2_000
MAX_RECORDS_FIT = 20_000  # the unigram baseline may be fitted on a whole training split
MIN_SCORED_RECORDS = 50  # below this a scored corpus is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.gpt2.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def validate_settings(
    max_new_tokens: Any, do_sample: Any, temperature: Any, top_p: Any, seed: Any
) -> dict[str, Any]:
    """Check decoding settings before the model runs; sampling must be explicit and seeded."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    if not isinstance(do_sample, bool):
        raise TypeError("do_sample must be a bool")
    if isinstance(temperature, bool) or not isinstance(temperature, int | float) or not temperature > 0:
        raise ValueError("temperature must be a number > 0")
    if isinstance(top_p, bool) or not isinstance(top_p, int | float) or not 0 < top_p <= 1:
        raise ValueError("top_p must be a number in (0, 1]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise TypeError("seed must be a non-negative int or None")
    if do_sample and seed is None:
        raise ValueError("seed is required when do_sample=True so that sampled output is reproducible")
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
        "seed": seed if do_sample else None,
        "decoding": "nucleus-sampling" if do_sample else DECODING_DEFAULT,
    }


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one non-empty str prompt; no special tokens are added and the prompt is continued verbatim",
    "prompt_chars": [1, MAX_TEXT_CHARS],
    "prompt_tokens": [1, MAX_PROMPT_TOKENS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "context_length": CONTEXT_LENGTH,
    "temperature": "number > 0 (sampling only)",
    "top_p": "number in (0, 1] (sampling only)",
    "seed": "non-negative int, required when do_sample=True",
    "vocab_size": VOCAB_SIZE,
    "eos_token_id": EOS_TOKEN_ID,
    "pad_token_id": PAD_TOKEN_ID,
    "preprocessing": (
        "byte-level BPE with no special tokens; a prompt over MAX_PROMPT_TOKENS, or a prompt whose "
        "tokens plus max_new_tokens exceed CONTEXT_LENGTH, is rejected rather than truncated"
    ),
}


def _check_prompt(prompt: Any) -> str:
    """Raise TypeError/ValueError naming the first violated prompt ceiling; return the prompt."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a str, got {type(prompt).__name__}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty or whitespace only")
    if len(prompt) > MAX_TEXT_CHARS:
        raise ValueError(f"prompt has {len(prompt)} chars > MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
    return prompt


def validate_inputs(
    prompt: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``generate`` would: both route the prompt through
    ``_check_prompt`` and the decoding settings through the public ``validate_settings``. The two
    token ceilings (``MAX_PROMPT_TOKENS`` and ``prompt + max_new_tokens <= CONTEXT_LENGTH``) need
    the loaded tokenizer and are therefore enforced inside ``generate``, not here.
    """
    checked = _check_prompt(prompt)
    settings = validate_settings(max_new_tokens, do_sample, temperature, top_p, seed)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry: generate takes one prompt per call")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "prompt-0", "chars": len(checked)}],
        "settings": settings,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], references: Sequence[str] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though no metric exists here.

    A free-text continuation has no ground truth and the repository ships no metric helper, so the
    verdict is always ``not-measurable`` (EVAL9). ``references`` exists for interface parity with
    the fleet's other pipelines and is recorded in ``reason`` rather than scored: perplexity needs a
    held-out corpus scored by the model, not a reference string compared to one completion, and any
    quality judgement needs human raters or a labelled downstream task.
    """
    settings = result.get("settings", {})
    supplied = references is not None
    return {
        "task": "causal text generation (continuing one prompt)",
        "score_semantics": (
            "the completion carries no score, probability or confidence; `finished_by` says whether "
            "the end-of-text token or the token budget stopped it, and the echoed `settings` say how "
            f"it was decoded ({settings.get('decoding', DECODING_DEFAULT)})"
        ),
        "sample_kind": sample_kind,
        "n_new_tokens": int(result.get("new_tokens", 0)),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "a continuation has no ground truth and the repository ships no metric helper"
            + (
                "; references were supplied but no metric helper exists to score them here, and a "
                "reference string is not a corpus"
                if supplied
                else "; the evaluated sample has no reference corpus"
            )
        ),
        "needs": (
            "a held-out reference corpus from the deployment domain, scored for perplexity with the "
            "caller's own code, for an intrinsic number; or human raters, or a labelled downstream "
            "task, for any quality or factuality claim — none of which this repository ships"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class GPT2TextGenerationPipeline:
    """``_tokenize`` maps text to token ids, ``_runner`` maps (prompt ids, settings) to new token ids, and
    ``_decode`` maps ids back to text; all three are injectable so tests run offline."""

    _tokenize: Callable[[str], list[int]]
    _runner: Callable[[list[int], dict[str, Any]], list[int]]
    _decode: Callable[[list[int]], str]
    device: str = "cpu"
    source: str = "injected"
    _scorer: Callable[[list[int]], list[float]] | None = field(default=None, repr=False)
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GPT2TextGenerationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2TokenizerFast.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = GPT2LMHeadModel.from_pretrained(
            source, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(prompt_ids: list[int], settings: dict[str, Any]) -> list[int]:
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=resolved_device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": settings["max_new_tokens"],
                "do_sample": settings["do_sample"],
                "pad_token_id": PAD_TOKEN_ID,
                "eos_token_id": EOS_TOKEN_ID,
            }
            if settings["do_sample"]:
                gen_kwargs.update(temperature=settings["temperature"], top_p=settings["top_p"])
                torch.manual_seed(settings["seed"])
            with torch.inference_mode():
                output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs)
            return output[0, len(prompt_ids) :].tolist()

        def tokenize(text: str) -> list[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        def scorer(ids: list[int]) -> list[float]:
            """Per-token NLLs (nats) of ids[1:] given the preceding tokens, under teacher forcing."""
            input_ids = torch.tensor([ids], dtype=torch.long, device=resolved_device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
            log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            return (-log_probs.gather(1, input_ids[0, 1:, None])[:, 0]).tolist()

        source_kind = "local-snapshot" if kwargs else "hf-hub"
        return cls(
            tokenize,
            runner,
            tokenizer.decode,
            resolved_device,
            source_kind,
            _scorer=scorer,
            _model=model,
            _tokenizer=tokenizer,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Continue one prompt. Greedy (deterministic) unless ``do_sample=True`` with a ``seed``."""
        prompt = _check_prompt(prompt)
        settings = validate_settings(max_new_tokens, do_sample, temperature, top_p, seed)
        prompt_ids = list(self._tokenize(prompt))
        if not 1 <= len(prompt_ids) <= MAX_PROMPT_TOKENS:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens, outside 1..MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS}"
            )
        if len(prompt_ids) + settings["max_new_tokens"] > CONTEXT_LENGTH:
            raise ValueError(
                f"prompt tokens {len(prompt_ids)} + max_new_tokens {settings['max_new_tokens']} "
                f"> CONTEXT_LENGTH={CONTEXT_LENGTH}"
            )
        new_ids = [int(t) for t in self._runner(prompt_ids, settings)]
        if len(new_ids) > settings["max_new_tokens"] or any(not 0 <= t < VOCAB_SIZE for t in new_ids):
            raise RuntimeError("runner returned more than max_new_tokens tokens, or an id outside the vocab")
        finished_by = "eos" if EOS_TOKEN_ID in new_ids else "max_new_tokens"
        kept = new_ids[: new_ids.index(EOS_TOKEN_ID)] if finished_by == "eos" else new_ids
        completion = self._decode(kept) if kept else ""
        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion,
            "prompt_tokens": len(prompt_ids),
            "new_tokens": len(kept),
            "finished_by": finished_by,
            "settings": settings,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def _record_ids(self, records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
        """Tokenise validated records; a record is refused (never truncated) above MAX_PROMPT_TOKENS or
        below two tokens (one token predicts nothing)."""
        out = []
        for record in records:
            ids = list(self._tokenize(record["text"]))
            if len(ids) > MAX_PROMPT_TOKENS:
                raise ValueError(
                    f"record {record['id']} has {len(ids)} tokens; ceiling is {MAX_PROMPT_TOKENS}"
                )
            if len(ids) < 2:
                raise ValueError(f"record {record['id']} tokenises to fewer than two tokens")
            out.append(ids)
        return out

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Score every record by teacher-forced perplexity (natural-log NLL per predicted token)."""
        from .metrics import sequence_metrics
        from .samples import validate_dataset

        if self._scorer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        losses = [self._scorer(ids) for ids in self._record_ids(checked)]
        metrics = sequence_metrics(losses)
        metrics.update(
            {
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def unigram_baseline(
        self, train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Perplexity of an add-one unigram model over the GPT-2 vocabulary, fitted on `train`, on `test`."""
        from .metrics import unigram_baseline
        from .samples import validate_dataset

        train_checked = validate_dataset(train, min_records=1, max_records=MAX_RECORDS_FIT)["records"]
        test_checked = validate_dataset(test, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        return unigram_baseline(self._record_ids(train_checked), self._record_ids(test_checked), VOCAB_SIZE)

    def _trainable_names(self, trainable_blocks: int) -> list[str]:
        if not isinstance(trainable_blocks, int) or not 1 <= trainable_blocks <= TRANSFORMER_BLOCKS:
            raise ValueError(f"trainable_blocks must be an int in 1..{TRANSFORMER_BLOCKS}")
        model, _ = self._require_model()
        first = TRANSFORMER_BLOCKS - trainable_blocks
        prefixes = tuple(f"transformer.h.{k}." for k in range(first, TRANSFORMER_BLOCKS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        lr: float = 1e-4,
        batch_size: int = 8,
        trainable_blocks: int = DEFAULT_TRAINABLE_BLOCKS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded causal-LM fine-tuning on a validated text corpus.

        Only the last `trainable_blocks` transformer blocks train (4 by default; the token and position
        embeddings, the tied output projection, the final layer norm and the earlier blocks stay frozen).
        Next-token cross-entropy on every token of every record (the end-of-text token is appended so the
        model also learns where a document ends), AdamW at a fixed learning rate with gradient clipping at
        1.0, records truncated to MAX_TRAIN_TOKENS **during training only**. Epoch 0 records the frozen
        model's validation perplexity; the epoch with the lowest validation perplexity is kept."""
        from .samples import validate_dataset

        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        names = self._trainable_names(trainable_blocks)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        train_ids = [ids[:MAX_TRAIN_TOKENS] + [EOS_TOKEN_ID] for ids in self._record_ids(train_checked)]
        import torch

        torch.manual_seed(seed)
        model, _ = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked).items()
                if k in ("perplexity", "bits_per_token", "n_tokens")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_ppl = entry["val"]["perplexity"] if entry["val"] else math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_ids), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_ids[i] for i in order[start : start + batch_size]]
                    width = max(len(ids) for ids in batch)
                    input_ids = torch.full((len(batch), width), PAD_TOKEN_ID, dtype=torch.long)
                    attention = torch.zeros((len(batch), width), dtype=torch.long)
                    labels = torch.full((len(batch), width), -100, dtype=torch.long)
                    for row, ids in enumerate(batch):
                        input_ids[row, : len(ids)] = torch.tensor(ids)
                        attention[row, : len(ids)] = 1
                        labels[row, : len(ids)] = torch.tensor(ids)
                    out = model(
                        input_ids=input_ids.to(device),
                        attention_mask=attention.to(device),
                        labels=labels.to(device),
                    )
                    optimiser.zero_grad(set_to_none=True)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(out.loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["perplexity"] if entry["val"] else -math.inf
                if current < best_ppl or not entry["val"]:
                    best_ppl = current
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the base
            # exactly as it was, with every parameter frozen again.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_blocks": trainable_blocks,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "lowest validation perplexity"
            if val_checked
            else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "max_train_tokens": MAX_TRAIN_TOKENS,
            "n_train": len(train_checked),
            "n_train_tokens": sum(len(ids) for ids in train_ids),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted transformer-block tensors as safetensors with a manifest naming the base."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        """Refuse an artifact whose manifest is not exactly the one this pipeline writes: the supported
        format and version, the pinned base (id, revision, weight file, digest), exactly one file entry
        named `adapter.safetensors` that resolves inside the artifact directory, and a recorded
        `trainable_blocks` in range. Nothing is deserialised here. The digest check that follows
        detects corruption or drift of the weights relative to the adjacent manifest; it is not
        authenticity against an actor who can replace both files."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not the supported "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", WEIGHT_FILE) != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_blocks") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int):
            raise ValueError("artifact manifest does not record an integer trainable_blocks")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        # The exact tensor set the recorded configuration implies — no subset, no extra, no other layer.
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_blocks"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("transformer.h."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable transformer-block tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key}: shape {tuple(value.shape)} != {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GPT2TextGenerationPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
