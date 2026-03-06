"""Tests for Gemini thinking-model token budget handling."""

from __future__ import annotations

import pytest

from paperbanana.providers.vlm.gemini import (
    _DEFAULT_THINKING_BUDGET,
    GeminiVLM,
)

# ---------------------------------------------------------------------------
# _is_thinking_model detection
# ---------------------------------------------------------------------------


class TestIsThinkingModel:
    """Verify that thinking-model detection matches Gemini 2.5+ naming."""

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-preview-05-20",
            "gemini-3.0-pro",
            "gemini-10.0-ultra",
        ],
    )
    def test_thinking_models(self, model: str):
        vlm = GeminiVLM(api_key="fake", model=model)
        assert vlm._is_thinking_model() is True

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite",
        ],
    )
    def test_non_thinking_models(self, model: str):
        vlm = GeminiVLM(api_key="fake", model=model)
        assert vlm._is_thinking_model() is False


# ---------------------------------------------------------------------------
# Token budget adjustment
# ---------------------------------------------------------------------------


class TestThinkingBudgetAdjustment:
    """Verify that generate() adjusts config for thinking models."""

    @pytest.fixture()
    def _mock_genai(self, monkeypatch):
        """Mock google.genai so generate() doesn't need a real API key."""
        import types as builtin_types
        from unittest.mock import MagicMock

        # Build a mock types module matching google.genai.types interface.
        mock_types = builtin_types.ModuleType("types")

        class _FakeConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class _FakeThinkingConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        mock_types.GenerateContentConfig = _FakeConfig
        mock_types.ThinkingConfig = _FakeThinkingConfig
        mock_types.Part = MagicMock()

        # Capture the config passed to generate_content.
        captured = {}

        class _FakeClient:
            class Models:  # noqa: N801
                @staticmethod
                def generate_content(model, contents, config):
                    captured["config"] = config
                    resp = MagicMock()
                    resp.text = "{}"
                    resp.usage_metadata = None
                    return resp

            models = Models()

        # Patch the import inside generate().
        import paperbanana.providers.vlm.gemini as gemini_mod

        monkeypatch.setattr(gemini_mod, "__name__", gemini_mod.__name__)  # no-op, just to anchor

        # We need to patch the dynamic import of google.genai.types.
        # The generate() method does `from google.genai import types`,
        # so we mock it at the module level.
        import sys

        mock_google = builtin_types.ModuleType("google")
        mock_genai = builtin_types.ModuleType("google.genai")
        mock_genai.types = mock_types
        mock_google.genai = mock_genai

        monkeypatch.setitem(sys.modules, "google", mock_google)
        monkeypatch.setitem(sys.modules, "google.genai", mock_genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", mock_types)

        return captured, _FakeClient

    @pytest.mark.asyncio
    async def test_thinking_model_scales_tokens(self, _mock_genai):
        captured, fake_client_cls = _mock_genai
        vlm = GeminiVLM(api_key="fake", model="gemini-2.5-flash")
        vlm._client = fake_client_cls()

        await vlm.generate("hello", max_tokens=4096)

        config = captured["config"]
        assert config.max_output_tokens == 4096 + _DEFAULT_THINKING_BUDGET
        assert hasattr(config, "thinking_config")
        assert config.thinking_config.thinking_budget == _DEFAULT_THINKING_BUDGET

    @pytest.mark.asyncio
    async def test_non_thinking_model_unchanged(self, _mock_genai):
        captured, fake_client_cls = _mock_genai
        vlm = GeminiVLM(api_key="fake", model="gemini-2.0-flash")
        vlm._client = fake_client_cls()

        await vlm.generate("hello", max_tokens=4096)

        config = captured["config"]
        assert config.max_output_tokens == 4096
        assert not hasattr(config, "thinking_config")
