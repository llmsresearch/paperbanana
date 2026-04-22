"""Tests for paperbanana.core.batch — manifest loading and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.core.batch import (
    REPORT_FILENAME,
    generate_batch_report_html,
    generate_batch_report_md,
    load_batch_manifest,
    load_batch_report,
    load_plot_batch_manifest,
    validate_manifest,
    write_batch_report,
)

# ---------------------------------------------------------------------------
# load_batch_manifest
# ---------------------------------------------------------------------------


def test_load_batch_manifest_pdf_pages(tmp_path: Path) -> None:
    m = tmp_path / "m.yaml"
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m.write_text(
        f"""items:
  - input: {txt.name}
    caption: "Fig 1"
    pdf_pages: "1-3"
""",
        encoding="utf-8",
    )
    items = load_batch_manifest(m)
    assert len(items) == 1
    assert items[0]["pdf_pages"] == "1-3"


def test_load_batch_manifest_pdf_pages_must_be_string(tmp_path: Path) -> None:
    m = tmp_path / "m.json"
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m.write_text(
        json.dumps(
            {
                "items": [
                    {"input": txt.name, "caption": "c", "pdf_pages": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pdf_pages"):
        load_batch_manifest(m)


# ---------------------------------------------------------------------------
# load_plot_batch_manifest
# ---------------------------------------------------------------------------


def test_load_plot_batch_manifest_resolves_paths(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n", encoding="utf-8")
    m = tmp_path / "plots.yaml"
    m.write_text(
        f"""items:
  - data: {csv.name}
    intent: "Line chart of y vs x"
    id: p1
""",
        encoding="utf-8",
    )
    items = load_plot_batch_manifest(m)
    assert len(items) == 1
    assert items[0]["id"] == "p1"
    assert items[0]["intent"].startswith("Line chart")
    assert Path(items[0]["data"]) == csv.resolve()


def test_load_plot_batch_manifest_requires_data_and_intent(tmp_path: Path) -> None:
    m = tmp_path / "bad.yaml"
    m.write_text('items:\n  - data: "x.csv"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="intent"):
        load_plot_batch_manifest(m)


def test_load_plot_batch_manifest_empty_items(tmp_path: Path) -> None:
    m = tmp_path / "empty.yaml"
    m.write_text("items: []\n", encoding="utf-8")
    assert load_plot_batch_manifest(m) == []


def test_load_plot_batch_manifest_rejects_non_tabular_suffix(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - data: {txt.name}
    intent: "test"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="csv or .json"):
        load_plot_batch_manifest(m)


# ---------------------------------------------------------------------------
# load_batch_report
# ---------------------------------------------------------------------------


def test_load_batch_report_success(tmp_path: Path):
    report_data = {
        "batch_id": "batch_20250109_120000_abc",
        "manifest": "/path/to/manifest.yaml",
        "items": [
            {"id": "fig1", "caption": "Overview", "output_path": "/out/fig1.png", "iterations": 3},
            {"id": "fig2", "caption": "Pipeline", "error": "API error"},
        ],
        "total_seconds": 42.5,
    }
    (tmp_path / REPORT_FILENAME).write_text(json.dumps(report_data), encoding="utf-8")
    loaded = load_batch_report(tmp_path)
    assert loaded["batch_id"] == "batch_20250109_120000_abc"
    assert len(loaded["items"]) == 2
    assert loaded["total_seconds"] == 42.5


def test_load_batch_report_dir_not_found():
    with pytest.raises(FileNotFoundError, match="Batch directory not found"):
        load_batch_report(Path("/nonexistent/batch_dir"))


def test_load_batch_report_json_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No batch_report.json"):
        load_batch_report(tmp_path)


def test_load_batch_report_invalid_json(tmp_path: Path):
    (tmp_path / REPORT_FILENAME).write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_batch_report(tmp_path)


def test_load_batch_report_missing_items_key(tmp_path: Path):
    (tmp_path / REPORT_FILENAME).write_text('{"batch_id": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid report"):
        load_batch_report(tmp_path)


# ---------------------------------------------------------------------------
# generate_batch_report_md
# ---------------------------------------------------------------------------


def test_generate_batch_report_md_contains_summary(tmp_path: Path):
    report = {
        "batch_id": "batch_123",
        "manifest": "manifest.yaml",
        "items": [
            {"id": "a", "caption": "Cap A", "output_path": "/out/a.png", "iterations": 2},
            {"id": "b", "caption": "Cap B", "error": "Failed"},
        ],
        "total_seconds": 10.0,
    }
    md = generate_batch_report_md(report, tmp_path)
    assert "# Batch Report: batch_123" in md
    assert "1/2 succeeded" in md
    assert "10.0s" in md
    assert "| a |" in md
    assert "| b |" in md
    assert "Success" in md
    assert "Failed" in md


def test_generate_batch_report_md_includes_batch_kind(tmp_path: Path) -> None:
    report = {
        "batch_id": "batch_plot",
        "manifest": "p.yaml",
        "batch_kind": "statistical_plot",
        "items": [],
        "total_seconds": 0.0,
    }
    md = generate_batch_report_md(report, tmp_path)
    assert "statistical plots" in md


def test_generate_batch_report_html_includes_batch_kind(tmp_path: Path) -> None:
    report = {
        "batch_id": "batch_m",
        "manifest": "m.yaml",
        "batch_kind": "methodology",
        "items": [],
        "total_seconds": 0.0,
    }
    html = generate_batch_report_html(report, tmp_path)
    assert "methodology diagrams" in html


# ---------------------------------------------------------------------------
# generate_batch_report_html
# ---------------------------------------------------------------------------


def test_generate_batch_report_html_contains_summary(tmp_path: Path):
    report = {
        "batch_id": "batch_456",
        "manifest": "m.yaml",
        "items": [
            {"id": "x", "caption": "X", "output_path": str(tmp_path / "x.png"), "iterations": 1},
        ],
        "total_seconds": 5.0,
    }
    html = generate_batch_report_html(report, tmp_path)
    assert "Batch Report: batch_456" in html
    assert "1/1" in html
    assert "5.0s" in html
    assert "<table>" in html
    assert "Success" in html
    assert "x.png" in html


# ---------------------------------------------------------------------------
# write_batch_report
# ---------------------------------------------------------------------------


def test_write_batch_report_markdown(tmp_path: Path):
    report_data = {
        "batch_id": "batch_write",
        "manifest": "manifest.yaml",
        "items": [
            {
                "id": "i1",
                "caption": "C",
                "output_path": str(tmp_path / "out.png"),
                "iterations": 1,
            }
        ],
        "total_seconds": 1.0,
    }
    (tmp_path / REPORT_FILENAME).write_text(json.dumps(report_data), encoding="utf-8")
    out_path = tmp_path / "report.md"
    written = write_batch_report(tmp_path, output_path=out_path, format="markdown")
    assert written == out_path
    assert out_path.exists()
    assert "Batch Report: batch_write" in out_path.read_text(encoding="utf-8")


def test_write_batch_report_html_default_path(tmp_path: Path):
    report_data = {
        "batch_id": "b",
        "manifest": "m",
        "items": [],
        "total_seconds": 0,
    }
    (tmp_path / REPORT_FILENAME).write_text(json.dumps(report_data), encoding="utf-8")
    written = write_batch_report(tmp_path, format="html")
    assert written == tmp_path / "batch_report.html"
    assert written.exists()


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------


def test_validate_manifest_valid_batch(tmp_path: Path) -> None:
    txt = tmp_path / "method.txt"
    txt.write_text("methodology", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - input: {txt.name}
    caption: "Fig 1"
    id: fig1
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m)
    assert errors == []


def test_validate_manifest_valid_plot(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - data: {csv.name}
    intent: "Bar chart"
    id: p1
    aspect_ratio: "16:9"
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="plot")
    assert errors == []


def test_validate_manifest_missing_required_fields(tmp_path: Path) -> None:
    m = tmp_path / "m.yaml"
    m.write_text(
        """items:
  - caption: "missing input"
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="batch")
    assert any("'input'" in e for e in errors)


def test_validate_manifest_duplicate_ids(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - input: {txt.name}
    caption: "a"
    id: dup
  - input: {txt.name}
    caption: "b"
    id: dup
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="batch")
    assert any("duplicate id" in e for e in errors)


def test_validate_manifest_missing_file_path(tmp_path: Path) -> None:
    m = tmp_path / "m.yaml"
    m.write_text(
        """items:
  - input: does_not_exist.txt
    caption: "Fig"
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="batch")
    assert any("does not exist" in e for e in errors)


def test_validate_manifest_unrecognised_keys(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - input: {txt.name}
    caption: "c"
    bogus_field: 42
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="batch")
    assert any("unrecognised" in e for e in errors)


def test_validate_manifest_invalid_aspect_ratio(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - data: {csv.name}
    intent: "Chart"
    aspect_ratio: "99:1"
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="plot")
    assert any("unsupported aspect_ratio" in e for e in errors)


def test_validate_manifest_invalid_pdf_pages(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - input: {txt.name}
    caption: "c"
    pdf_pages: "abc"
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="batch")
    assert any("invalid pdf_pages format" in e for e in errors)


def test_validate_manifest_collects_all_errors(tmp_path: Path) -> None:
    """Ensure multiple violations are reported, not just the first."""
    m = tmp_path / "m.yaml"
    m.write_text(
        """items:
  - input: missing.txt
    caption: "a"
    id: dup
    extra_key: true
  - input: also_missing.txt
    caption: "b"
    id: dup
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m, manifest_type="batch")
    assert len(errors) >= 3  # missing paths + duplicate id + unrecognised key


def test_validate_manifest_nonexistent_file(tmp_path: Path) -> None:
    errors = validate_manifest(tmp_path / "nope.yaml")
    assert len(errors) == 1
    assert "not found" in errors[0].lower()


def test_validate_manifest_auto_detect_plot(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n", encoding="utf-8")
    m = tmp_path / "m.yaml"
    m.write_text(
        f"""items:
  - data: {csv.name}
    intent: "Chart"
""",
        encoding="utf-8",
    )
    errors = validate_manifest(m)
    assert errors == []
