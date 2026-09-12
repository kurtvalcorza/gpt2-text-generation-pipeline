import hashlib
import json
import re
from pathlib import Path

import pytest

from gpt2_text_generation_pipeline import (
    CONTEXT_LENGTH,
    DEFAULT_WEIGHTS_DIR,
    EOS_TOKEN_ID,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    GPT2TextGenerationPipeline,
    stage_missing_files,
    validate_settings,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "openai-community/gpt2"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert PAD_TOKEN_ID == EOS_TOKEN_ID == 50256 and VOCAB_SIZE == 50257
    assert MAX_PROMPT_TOKENS < CONTEXT_LENGTH == 1024
    manifest = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        assert data["modelKey"] == MODEL_KEY


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> Path:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "gpt2"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "gpt2"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    manifest_path = _write_snapshot(tmp_path, b"abc")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        GPT2TextGenerationPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


# --- offline pipeline with injected tokenizer / runner / decoder -------------------------------------------

VOCAB = ["<unused>", "the", "cat", "sat", "on", "mat", "dog", "a"]  # toy ids 1..7, all < VOCAB_SIZE


def _tokenize(text: str) -> list[int]:
    return [VOCAB.index(word) for word in text.split()]


def _decode(ids: list[int]) -> str:
    return " " + " ".join(VOCAB[i] if 0 <= i < len(VOCAB) else f"<{i}>" for i in ids)


def _pipeline(calls: list | None = None, new_ids: list[int] | None = None) -> GPT2TextGenerationPipeline:
    def runner(prompt_ids, settings):
        if calls is not None:
            calls.append((list(prompt_ids), dict(settings)))
        return list(new_ids) if new_ids is not None else [5, 4, 6][: settings["max_new_tokens"]]

    return GPT2TextGenerationPipeline(_tokenize, runner, _decode, "cpu", "injected")


def test_generate_greedy_output_fields():
    calls: list = []
    result = _pipeline(calls).generate("the cat sat", max_new_tokens=3)
    assert result["prompt"] == "the cat sat" and result["completion"] == " mat on dog"
    assert result["text"] == "the cat sat mat on dog"
    assert result["prompt_tokens"] == 3 and result["new_tokens"] == 3
    assert result["finished_by"] == "max_new_tokens"
    assert result["settings"] == {
        "max_new_tokens": 3,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "seed": None,
        "decoding": "greedy",
    }
    assert result["device"] == "cpu" and result["source"] == "injected"
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert calls == [([1, 2, 3], result["settings"])]


def test_generate_stops_at_eos_and_strips_it():
    result = _pipeline(new_ids=[5, EOS_TOKEN_ID, 6]).generate("the cat", max_new_tokens=8)
    assert result["finished_by"] == "eos" and result["new_tokens"] == 1 and result["completion"] == " mat"
    result = _pipeline(new_ids=[EOS_TOKEN_ID]).generate("the cat", max_new_tokens=8)
    assert result["finished_by"] == "eos" and result["new_tokens"] == 0 and result["completion"] == ""


def test_generate_sampling_requires_seed_and_echoes_settings():
    calls: list = []
    pipe = _pipeline(calls)
    with pytest.raises(ValueError, match="seed is required"):
        pipe.generate("the cat", do_sample=True)
    result = pipe.generate("the cat", max_new_tokens=2, do_sample=True, temperature=0.7, top_p=0.9, seed=42)
    assert result["settings"] == {
        "max_new_tokens": 2,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "decoding": "nucleus-sampling",
    }
    assert calls[-1][1]["seed"] == 42


def test_generate_rejects_bad_prompts():
    pipe = _pipeline()
    with pytest.raises(TypeError):
        pipe.generate(b"the cat")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.generate("   ")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.generate("a" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_PROMPT_TOKENS"):
        pipe.generate("a " * (MAX_PROMPT_TOKENS + 1))
    with pytest.raises(ValueError, match="CONTEXT_LENGTH"):
        pipe.generate("a " * (CONTEXT_LENGTH - 10), max_new_tokens=11)
    fits = pipe.generate("a " * (CONTEXT_LENGTH - 10), max_new_tokens=10)
    assert fits["prompt_tokens"] == CONTEXT_LENGTH - 10


def test_validate_settings_rejects_bad_values():
    with pytest.raises(TypeError):
        validate_settings(2.5, False, 1.0, 1.0, None)
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_settings(0, False, 1.0, 1.0, None)
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_settings(MAX_NEW_TOKENS + 1, False, 1.0, 1.0, None)
    with pytest.raises(TypeError, match="do_sample"):
        validate_settings(4, "yes", 1.0, 1.0, None)
    with pytest.raises(ValueError, match="temperature"):
        validate_settings(4, True, 0.0, 1.0, 1)
    with pytest.raises(ValueError, match="top_p"):
        validate_settings(4, True, 1.0, 1.5, 1)
    with pytest.raises(TypeError, match="seed"):
        validate_settings(4, True, 1.0, 1.0, -1)
    assert validate_settings(4, False, 1.0, 1.0, None)["decoding"] == "greedy"


def test_generate_rejects_backend_overrun_and_foreign_ids():
    with pytest.raises(RuntimeError):
        _pipeline(new_ids=[1, 2, 3, 4]).generate("the cat", max_new_tokens=3)
    with pytest.raises(RuntimeError):
        _pipeline(new_ids=[VOCAB_SIZE]).generate("the cat", max_new_tokens=3)
