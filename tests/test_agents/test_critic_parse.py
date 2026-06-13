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


class TestExtractJsonRobustness:
    """extract_json should recover JSON wrapped in fences or prose."""

    def test_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_inline_fence_without_newline(self):
        text = '```json{"key": "value"}```'
        assert extract_json(text) == {"key": "value"}

    def test_leading_prose(self):
        text = 'Here is my evaluation of the image:\n{"key": "value"}'
        assert extract_json(text) == {"key": "value"}

    def test_trailing_commentary(self):
        text = '{"key": "value"}\nLet me know if you need anything else.'
        assert extract_json(text) == {"key": "value"}

    def test_prose_around_fenced_json(self):
        text = 'Sure! Here it is:\n```json\n{"key": "value"}\n```\nHope that helps.'
        assert extract_json(text) == {"key": "value"}

    def test_nested_braces_in_strings(self):
        text = 'Result: {"summary": "use {curly} braces", "n": 1} done'
        assert extract_json(text) == {"summary": "use {curly} braces", "n": 1}

    def test_genuinely_invalid_returns_none(self):
        assert extract_json("I could not evaluate the image, sorry.") is None

    def test_array_with_trailing_prose_stays_an_array(self):
        # Seen live: an array of detections followed by commentary was
        # collapsed to its first element because the object scan ran first.
        text = '[{"kind": "figure"}, {"kind": "table"}]\nThese are all the regions.'
        assert extract_json(text) == [{"kind": "figure"}, {"kind": "table"}]

    def test_prose_then_array(self):
        text = 'Here are the regions:\n[{"kind": "figure"}, {"kind": "table"}]'
        assert extract_json(text) == [{"kind": "figure"}, {"kind": "table"}]

    def test_object_containing_array_with_prose_stays_an_object(self):
        text = 'Result: {"items": [1, 2], "ok": true} as requested.'
        assert extract_json(text) == {"items": [1, 2], "ok": True}

    def test_truncated_array_falls_back_to_first_complete_object(self):
        # A mid-array truncation has no parseable array; surfacing the first
        # complete object lets callers detect the wrong shape and retry.
        text = '[{"kind": "figure", "bbox": [1, 2, 3, 4]}, {"kind": "table", "bbox": [5,'
        assert extract_json(text) == {"kind": "figure", "bbox": [1, 2, 3, 4]}


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

    def test_fenced_json_parsed(self, critic):
        data = {"critic_suggestions": ["fix arrow"], "revised_description": "v2"}
        result = critic._parse_response(f"```json\n{json.dumps(data)}\n```")
        assert result.needs_revision is True
        assert result.critic_suggestions == ["fix arrow"]
        assert result.revised_description == "v2"

    def test_json_with_leading_prose_parsed(self, critic):
        data = {"critic_suggestions": ["align labels"], "revised_description": None}
        result = critic._parse_response(f"Here is the critique:\n{json.dumps(data)}")
        assert result.critic_suggestions == ["align labels"]

    def test_json_with_trailing_commentary_parsed(self, critic):
        data = {"critic_suggestions": [], "revised_description": None}
        result = critic._parse_response(f"{json.dumps(data)}\nOverall the figure looks good.")
        assert result.needs_revision is False

    def test_invalid_response_returns_empty_critique_without_raising(self, critic):
        result = critic._parse_response("The model refused to answer in JSON format.")
        assert result.needs_revision is False
        assert result.critic_suggestions == []
        assert result.revised_description is None
