"""Tests for CriticAgent._parse_response robustness."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from paperbanana.agents.critic import CriticAgent
from paperbanana.core.utils import extract_json


class TestExtractJsonNoneSafe:
    """extract_json should handle None and empty strings gracefully."""

    def test_none_returns_none(self):
        assert extract_json(None) is None

    def test_empty_string_returns_none(self):
        assert extract_json("") is None

    def test_valid_json_object(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        result = extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_truncated_json_returns_none(self):
        assert extract_json('{"key": "val') is None


class TestCriticParseResponse:
    """CriticAgent._parse_response should not crash on malformed input."""

    @pytest.fixture()
    def critic(self):
        return CriticAgent(vlm_provider=MagicMock(), prompt_dir="prompts")

    def test_none_response_returns_empty_critique(self, critic):
        result = critic._parse_response(None)
        assert result.needs_revision is False
        assert result.critic_suggestions == []

    def test_empty_string_returns_empty_critique(self, critic):
        result = critic._parse_response("")
        assert result.needs_revision is False

    def test_valid_json_parsed(self, critic):
        data = {
            "critic_suggestions": ["fix spacing"],
            "revised_description": "improved version",
        }
        result = critic._parse_response(json.dumps(data))
        assert result.needs_revision is True
        assert result.critic_suggestions == ["fix spacing"]
        assert result.revised_description == "improved version"

    def test_no_suggestions_means_no_revision(self, critic):
        data = {"critic_suggestions": [], "revised_description": None}
        result = critic._parse_response(json.dumps(data))
        assert result.needs_revision is False

    def test_truncated_json_returns_empty_critique(self, critic):
        result = critic._parse_response('{"critic_suggestions": ["fix')
        assert result.needs_revision is False
