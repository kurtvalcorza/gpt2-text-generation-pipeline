"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from gpt2_text_generation_pipeline import (
    CONTEXT_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    EOS_TOKEN_ID,
    INPUT_SCHEMA,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    evaluation_report,
    validate_inputs,
)

PROMPT = "The weather in the mountains is usually"


def _result(decoding: str = "greedy", new_tokens: int = 32) -> dict:
    return {
        "prompt": PROMPT,
        "completion": " good, but the snow is not.",
        "text": PROMPT + " good, but the snow is not.",
        "prompt_tokens": 7,
        "new_tokens": new_tokens,
        "finished_by": "max_new_tokens",
        "settings": {"max_new_tokens": new_tokens, "do_sample": decoding != "greedy", "decoding": decoding},
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(PROMPT, max_new_tokens=16, names=["synthetic_weather_prompt"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["prompt_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["prompt_tokens"] == [1, MAX_PROMPT_TOKENS]
    assert manifest["schema"]["max_new_tokens"] == [1, MAX_NEW_TOKENS]
    assert manifest["schema"]["context_length"] == CONTEXT_LENGTH
    assert manifest["schema"]["vocab_size"] == VOCAB_SIZE
    assert manifest["schema"]["eos_token_id"] == EOS_TOKEN_ID == PAD_TOKEN_ID
    assert manifest["inputs"] == [{"id": "synthetic_weather_prompt", "chars": len(PROMPT)}]
    assert manifest["settings"]["decoding"] == "greedy"
    assert manifest["settings"]["max_new_tokens"] == 16
    assert manifest["settings"]["seed"] is None
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_sampling_settings() -> None:
    manifest = validate_inputs(PROMPT, do_sample=True, temperature=0.8, top_p=0.9, seed=7)
    assert [entry["id"] for entry in manifest["inputs"]] == ["prompt-0"]
    assert manifest["settings"]["decoding"] == "nucleus-sampling"
    assert manifest["settings"]["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS
    assert (manifest["settings"]["temperature"], manifest["settings"]["top_p"]) == (0.8, 0.9)
    assert manifest["settings"]["seed"] == 7


def test_validate_inputs_rejects_like_generate() -> None:
    with pytest.raises(TypeError, match="prompt must be a str"):
        validate_inputs(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        validate_inputs("   ")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs("x" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(PROMPT, max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(ValueError, match="temperature must be a number > 0"):
        validate_inputs(PROMPT, do_sample=True, temperature=0.0, seed=1)
    with pytest.raises(ValueError, match=r"top_p must be a number in \(0, 1\]"):
        validate_inputs(PROMPT, do_sample=True, top_p=1.5, seed=1)
    with pytest.raises(ValueError, match="seed is required when do_sample=True"):
        validate_inputs(PROMPT, do_sample=True)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(PROMPT, names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_new_tokens"] == 32
    assert report["sample_kind"] == "synthetic"
    assert "no reference corpus" in report["reason"]
    assert "perplexity" in report["needs"]
    assert "greedy" in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_stays_not_measurable_when_references_are_supplied() -> None:
    report = evaluation_report(_result("nucleus-sampling", 16), ["a reference"], sample_kind="BYOD upload")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD upload"
    assert report["n_new_tokens"] == 16
    assert "a reference string is not a corpus" in report["reason"]
    assert "nucleus-sampling" in report["score_semantics"]
