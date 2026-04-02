"""Tests for paperbanana.core.plot_data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperbanana.core.plot_data import load_statistical_plot_payload


def test_load_statistical_plot_payload_csv(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    ctx, payload = load_statistical_plot_payload(p)
    assert "a" in ctx and "b" in ctx
    assert len(payload) == 2
    assert set(payload[0].keys()) == {"a", "b"}


def test_load_statistical_plot_payload_json(tmp_path: Path) -> None:
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"x": 1}, {"x": 2}]), encoding="utf-8")
    ctx, payload = load_statistical_plot_payload(p)
    assert "JSON data" in ctx
    assert payload == [{"x": 1}, {"x": 2}]


def test_load_statistical_plot_payload_json_unwraps_data_envelope(tmp_path: Path) -> None:
    """{"data": [...]} must not double-wrap when passed as raw_data={"data": payload}."""
    p = tmp_path / "env.json"
    inner = [{"x": 1}]
    p.write_text(json.dumps({"data": inner, "meta": "ignored"}), encoding="utf-8")
    ctx, payload = load_statistical_plot_payload(p)
    assert payload == inner
    assert "ignored" not in ctx


def test_load_statistical_plot_payload_json_object_without_data_key(tmp_path: Path) -> None:
    p = tmp_path / "obj.json"
    obj = {"models": ["a"], "scores": [0.9]}
    p.write_text(json.dumps(obj), encoding="utf-8")
    ctx, payload = load_statistical_plot_payload(p)
    assert payload == obj
    assert "models" in ctx


def test_load_statistical_plot_payload_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_statistical_plot_payload(tmp_path / "nope.csv")


def test_load_statistical_plot_payload_bad_suffix(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="csv or .json"):
        load_statistical_plot_payload(p)
