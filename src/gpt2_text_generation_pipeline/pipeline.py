"""Causal text generation over the pinned ``openai-community/gpt2`` (124M) checkpoint.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``generate``: greedy decoding by default
(deterministic), nucleus sampling only when asked for and seeded. One prompt per call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

        source_kind = "local-snapshot" if kwargs else "hf-hub"
        return cls(tokenize, runner, tokenizer.decode, resolved_device, source_kind)

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
