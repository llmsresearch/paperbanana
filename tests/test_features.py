"""Tests for auto-refine, continue-run, and critic user feedback features."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from paperbanana.core.config import Settings
from paperbanana.core.resume import load_resume_state
from paperbanana.core.types import CritiqueResult, DiagramType

# ── Settings tests ───────────────────────────────────────────────


def test_auto_refine_defaults():
    """auto_refine defaults to False, max_iterations to 30."""
    settings = Settings()
    assert settings.auto_refine is False
    assert settings.max_iterations == 30


def test_auto_refine_override():
    """auto_refine and max_iterations can be overridden."""
    settings = Settings(auto_refine=True, max_iterations=5)
    assert settings.auto_refine is True
    assert settings.max_iterations == 5


def test_auto_refine_from_yaml():
    """auto_refine loads from YAML config."""
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"pipeline": {"auto_refine": True, "max_iterations": 15}}, f)
        path = f.name
    try:
        settings = Settings.from_yaml(path)
        assert settings.auto_refine is True
        assert settings.max_iterations == 15
    finally:
        Path(path).unlink(missing_ok=True)


# ── CritiqueResult tests ────────────────────────────────────────


def test_critique_needs_revision_with_suggestions():
    """needs_revision is True when suggestions exist."""
    cr = CritiqueResult(
        critic_suggestions=["Fix arrow direction"],
        revised_description="Updated desc",
    )
    assert cr.needs_revision is True


def test_critique_no_revision_when_empty():
    """needs_revision is False when no suggestions."""
    cr = CritiqueResult(critic_suggestions=[], revised_description=None)
    assert cr.needs_revision is False


# ── Resume state tests ──────────────────────────────────────────


def test_load_resume_state_with_iterations():
    """load_resume_state correctly finds the last iteration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_test_123"
        run_dir.mkdir()

        # Write run_input.json
        run_input = {
            "source_context": "Our encoder-decoder framework...",
            "communicative_intent": "Overview of our framework",
            "diagram_type": "methodology",
            "raw_data": None,
        }
        (run_dir / "run_input.json").write_text(json.dumps(run_input))

        # Write iter_1
        iter1 = run_dir / "iter_1"
        iter1.mkdir()
        (iter1 / "details.json").write_text(
            json.dumps(
                {
                    "description": "Initial description",
                    "critique": {
                        "critic_suggestions": ["Fix colors"],
                        "revised_description": "Revised desc v1",
                    },
                }
            )
        )

        # Write iter_2
        iter2 = run_dir / "iter_2"
        iter2.mkdir()
        (iter2 / "details.json").write_text(
            json.dumps(
                {
                    "description": "Revised desc v1",
                    "critique": {
                        "critic_suggestions": [],
                        "revised_description": None,
                    },
                }
            )
        )

        state = load_resume_state(tmpdir, "run_test_123")
        assert state.run_id == "run_test_123"
        assert state.last_iteration == 2
        assert state.source_context == "Our encoder-decoder framework..."
        assert state.communicative_intent == "Overview of our framework"
        assert state.diagram_type == DiagramType.METHODOLOGY
        # Last iteration had no revised_description, falls back to description
        assert state.last_description == "Revised desc v1"


def test_load_resume_state_no_iterations():
    """load_resume_state falls back to planning.json when no iterations exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_test_456"
        run_dir.mkdir()

        run_input = {
            "source_context": "Method text",
            "communicative_intent": "Caption",
            "diagram_type": "methodology",
        }
        (run_dir / "run_input.json").write_text(json.dumps(run_input))

        planning = {
            "retrieved_examples": [],
            "initial_description": "Raw desc",
            "optimized_description": "Optimized desc",
        }
        (run_dir / "planning.json").write_text(json.dumps(planning))

        state = load_resume_state(tmpdir, "run_test_456")
        assert state.last_iteration == 0
        assert state.last_description == "Optimized desc"


def test_load_resume_state_missing_run_input():
    """load_resume_state raises FileNotFoundError for old runs without run_input.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run_old"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="run_input.json not found"):
            load_resume_state(tmpdir, "run_old")


def test_load_resume_state_missing_dir():
    """load_resume_state raises FileNotFoundError for non-existent run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            load_resume_state(tmpdir, "run_nonexistent")


# ── Critic user feedback tests ───────────────────────────────────


def test_critic_agent_accepts_user_feedback():
    """CriticAgent.run() accepts user_feedback parameter."""
    # Verify the parameter exists in the signature
    import inspect

    from paperbanana.agents.critic import CriticAgent

    sig = inspect.signature(CriticAgent.run)
    assert "user_feedback" in sig.parameters
    param = sig.parameters["user_feedback"]
    assert param.default is None


# ── Input optimizer tests ────────────────────────────────────────


def test_optimize_inputs_default_false():
    """optimize_inputs defaults to False."""
    settings = Settings()
    assert settings.optimize_inputs is False


def test_optimize_inputs_override():
    """optimize_inputs can be enabled."""
    settings = Settings(optimize_inputs=True)
    assert settings.optimize_inputs is True


def test_optimizer_agent_signature():
    """InputOptimizerAgent.run() accepts expected parameters."""
    import inspect

    from paperbanana.agents.optimizer import InputOptimizerAgent

    sig = inspect.signature(InputOptimizerAgent.run)
    params = list(sig.parameters.keys())
    assert "source_context" in params
    assert "caption" in params
    assert "diagram_type" in params


def test_optimizer_prompts_exist():
    """Optimizer prompt templates exist for diagram type."""
    from pathlib import Path

    prompts_dir = Path(__file__).parent.parent / "prompts" / "diagram"
    assert (prompts_dir / "context_enricher.txt").exists()
    assert (prompts_dir / "caption_sharpener.txt").exists()


def test_optimizer_from_yaml():
    """optimize_inputs loads from YAML config."""
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"pipeline": {"optimize_inputs": True}}, f)
        path = f.name
    try:
        settings = Settings.from_yaml(path)
        assert settings.optimize_inputs is True
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_input_json_structure():
    """run_input.json has the expected structure."""
    data = {
        "source_context": "text",
        "communicative_intent": "caption",
        "diagram_type": "methodology",
        "raw_data": None,
    }
    # Verify it round-trips through JSON
    parsed = json.loads(json.dumps(data))
    assert parsed["source_context"] == "text"
    assert parsed["diagram_type"] == "methodology"
    assert DiagramType(parsed["diagram_type"]) == DiagramType.METHODOLOGY
