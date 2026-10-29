"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there; the router relays it as the same line. A genuine failure keeps its
traceback. Section 4 is executed from the committed notebook with a stub pipeline."""
# ruff: noqa: E501

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import gpt2_text_generation_pipeline as package
from gpt2_text_generation_pipeline import samples as sm
from test_gpt2_text_generation_colab_review_fixes import (  # noqa: F401  (fixture)
    _run_section4,
    _section4_namespace,
    nb,
)

REFUSAL_CELL = "gpt2_text_generation-17"


def _line(exc: BaseException) -> str:
    assert type(exc).__name__ == "DatasetRefused" and exc.dimer_refusal is True
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    assert exc.__suppress_context__
    return rendered[0]


def test_the_scripted_refusal_input_is_one_line_naming_the_file(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    ns = _section4_namespace(tmp_path)
    ns["adapt_result"] = {"stale": True}
    with pytest.raises(sm.DatasetRefused) as caught:
        _run_section4(nb, ns, use_byod=True, path=str(hostname))
    line = _line(caught.value)
    assert line.startswith("Refused: hostname: ") and "BYOD refused" not in line and "run this cell again" in line
    assert "adapt_result" not in ns  # the in-cell reset of the earlier run is kept


def test_a_directory_and_a_missing_file_are_refused_by_name(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "etc"
    folder.mkdir()
    for path, name in ((folder, "etc"), (tmp_path / "missing.csv", "missing.csv")):
        with pytest.raises(sm.DatasetRefused) as caught:
            _run_section4(nb, _section4_namespace(tmp_path), use_byod=True, path=str(path))
        assert _line(caught.value).startswith(f"Refused: {name}: ")


def test_no_upload_dialog_is_refused_as_no_file(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(sm.DatasetRefused) as caught:
        _run_section4(nb, _section4_namespace(tmp_path), use_byod=True, path="")
    assert _line(caught.value).startswith("Refused: (no file): USE_BYOD = True needs a file")


def test_a_genuine_failure_keeps_its_type(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine.csv").write_text("id,text\na,hello\n", encoding="utf-8")
    ns = _section4_namespace(tmp_path)

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug"):
        _run_section4(nb, ns, use_byod=True, path=str(tmp_path / "mine.csv"))


def test_the_refusal_cell_id_and_the_router_line(nb: dict) -> None:  # noqa: F811
    cell = next(c for c in nb["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert cell["cell_type"] == "code" and "BYOD_PATH = ''" in source and "raise DatasetRefused(f'{byod_label}: " in source
    router = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "class IsolatedCellRefusal(RuntimeError):" in "".join(c["source"]))
    assert 'return [f"Refused: {self}"]' in router and 'getattr(exc, "dimer_refusal", False)' in router
    assert package.DatasetRefused("x.csv")._render_traceback_() == ["Refused: x.csv"]
