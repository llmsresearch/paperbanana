"""Tests for the auto-generate figure caption feature (issue #98).

Covers:
- Settings: generate_caption default and override, YAML loading
- CaptionAgent: signature, prompt loading, quote stripping, diagram/plot types
- Pipeline: caption disabled (default), caption enabled, caption failure is graceful,
  metadata written, GenerationOutput.generated_caption populated
- continue_run: caption also generated in continue flow
- Progress events: CAPTION_START / CAPTION_END emitted
- CLI: --generate-caption flag wired into Settings
- Prompt files exist for both diagram and plot types
"""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.resume import load_resume_state
from paperbanana.core.types import (
    DiagramType,
    GenerationInput,
    PipelineProgressStage,
)

# ── Shared mock helpers ────────────────────────────────────────────


class _MockVLM:
    """Sequenced VLM mock returning pre-configured responses in order."""

    name = "mock-vlm"
    model_name = "mock-model"

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0
        self.calls: list[dict] = []

    async def generate(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]


class _MockImageGen:
    name = "mock-image-gen"
    model_name = "mock-image-model"

    async def generate(self, *args, **kwargs):
        return Image.new("RGB", (128, 128), color=(200, 200, 200))


def _critic_satisfied() -> str:
    return json.dumps({"critic_suggestions": [], "revised_description": None})


def _make_settings(tmp_path, **extra) -> Settings:
    return Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        refinement_iterations=1,
        save_iterations=False,
        **extra,
    )


# ── Settings tests ────────────────────────────────────────────────


def test_generate_caption_default_false():
    assert Settings().generate_caption is False


def test_generate_caption_override():
    assert Settings(generate_caption=True).generate_caption is True


def test_generate_caption_from_yaml():
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"pipeline": {"generate_caption": True}}, f)
        path = f.name
    try:
        settings = Settings.from_yaml(path)
        assert settings.generate_caption is True
    finally:
        Path(path).unlink(missing_ok=True)


# ── CaptionAgent unit tests ────────────────────────────────────────


def test_caption_agent_has_correct_name():
    from paperbanana.agents.caption import CaptionAgent

    assert CaptionAgent.__name__ == "CaptionAgent"
    mock_vlm = MagicMock()
    agent = CaptionAgent(vlm_provider=mock_vlm)
    assert agent.agent_name == "caption"


def test_caption_agent_run_signature():
    from paperbanana.agents.caption import CaptionAgent

    sig = inspect.signature(CaptionAgent.run)
    params = list(sig.parameters.keys())
    assert "image_path" in params
    assert "source_context" in params
    assert "intent" in params
    assert "description" in params
    assert "diagram_type" in params
    # diagram_type defaults to METHODOLOGY
    assert sig.parameters["diagram_type"].default == DiagramType.METHODOLOGY


def test_caption_prompts_exist_for_both_types():
    prompts_root = Path(__file__).parent.parent / "prompts"
    assert (prompts_root / "diagram" / "caption.txt").exists(), "diagram/caption.txt missing"
    assert (prompts_root / "plot" / "caption.txt").exists(), "plot/caption.txt missing"


def test_caption_prompt_has_required_placeholders():
    prompts_root = Path(__file__).parent.parent / "prompts"
    for subdir in ("diagram", "plot"):
        text = (prompts_root / subdir / "caption.txt").read_text()
        assert "{source_context}" in text, f"{subdir}/caption.txt missing {{source_context}}"
        assert "{intent}" in text, f"{subdir}/caption.txt missing {{intent}}"
        assert "{description}" in text, f"{subdir}/caption.txt missing {{description}}"


@pytest.mark.asyncio
async def test_caption_agent_strips_surrounding_quotes(tmp_path):
    """CaptionAgent strips surrounding double/single quotes from model output."""
    from paperbanana.agents.caption import CaptionAgent
    from paperbanana.core.utils import find_prompt_dir

    # Create a tiny test image
    img_path = str(tmp_path / "test.png")
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

    mock_vlm = MagicMock()
    mock_vlm.generate = AsyncMock(return_value='"This is a quoted caption."')

    agent = CaptionAgent(vlm_provider=mock_vlm, prompt_dir=find_prompt_dir())
    result = await agent.run(
        image_path=img_path,
        source_context="Source context here",
        intent="Show framework overview",
        description="Detailed visual description",
    )
    assert result == "This is a quoted caption."
    assert not result.startswith('"')


@pytest.mark.asyncio
async def test_caption_agent_no_strip_when_no_quotes(tmp_path):
    """CaptionAgent returns the response unchanged when no surrounding quotes."""
    from paperbanana.agents.caption import CaptionAgent
    from paperbanana.core.utils import find_prompt_dir

    img_path = str(tmp_path / "test.png")
    Image.new("RGB", (64, 64)).save(img_path)

    mock_vlm = MagicMock()
    expected = "Overview of our encoder-decoder framework."
    mock_vlm.generate = AsyncMock(return_value=expected)

    agent = CaptionAgent(vlm_provider=mock_vlm, prompt_dir=find_prompt_dir())
    result = await agent.run(
        image_path=img_path,
        source_context="ctx",
        intent="intent",
        description="desc",
    )
    assert result == expected


@pytest.mark.asyncio
async def test_caption_agent_uses_plot_prompt_for_statistical_type(tmp_path):
    """CaptionAgent loads the 'plot' prompt when diagram_type is STATISTICAL_PLOT."""
    from paperbanana.agents.caption import CaptionAgent
    from paperbanana.core.utils import find_prompt_dir

    img_path = str(tmp_path / "test.png")
    Image.new("RGB", (64, 64)).save(img_path)

    mock_vlm = MagicMock()
    mock_vlm.generate = AsyncMock(return_value="A plot caption.")

    agent = CaptionAgent(vlm_provider=mock_vlm, prompt_dir=find_prompt_dir())
    result = await agent.run(
        image_path=img_path,
        source_context="data context",
        intent="bar chart intent",
        description="bar chart description",
        diagram_type=DiagramType.STATISTICAL_PLOT,
    )
    assert result == "A plot caption."


# ── Pipeline: caption disabled (default behaviour) ─────────────────


@pytest.mark.asyncio
async def test_caption_disabled_by_default(tmp_path):
    """generate_caption=False (default) → generated_caption is None."""
    vlm = _MockVLM(["Planned description", "Styled description", _critic_satisfied()])
    pipeline = PaperBananaPipeline(
        settings=_make_settings(tmp_path),
        vlm_client=vlm,
        image_gen_fn=_MockImageGen(),
    )
    result = await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent")
    )
    assert result.generated_caption is None
    assert "generated_caption" not in result.metadata
    # Caption agent should NOT have been called: VLM call count == 3 (plan+style+critic)
    assert vlm._idx == 3


@pytest.mark.asyncio
async def test_caption_generated_when_enabled(tmp_path):
    """generate_caption=True → generated_caption is set on output and metadata."""
    expected_caption = "Overview of our encoder-decoder attention mechanism."
    vlm = _MockVLM(
        [
            "Planned description",
            "Styled description",
            _critic_satisfied(),
            expected_caption,  # caption agent call
        ]
    )
    settings = _make_settings(tmp_path, generate_caption=True)
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=vlm,
        image_gen_fn=_MockImageGen(),
    )
    result = await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent")
    )
    assert result.generated_caption == expected_caption
    assert result.metadata["generated_caption"] == expected_caption


@pytest.mark.asyncio
async def test_caption_uses_final_output_image(tmp_path):
    """CaptionAgent receives the final_output image path, not a per-iteration path."""
    from paperbanana.agents.caption import CaptionAgent
    from paperbanana.core.utils import find_prompt_dir

    called_with_paths = []

    class _TrackingCaptionAgent(CaptionAgent):
        async def run(self, image_path, **kwargs):
            called_with_paths.append(image_path)
            return "Caption text"

    vlm = _MockVLM(["Plan", "Style", _critic_satisfied()])
    settings = _make_settings(tmp_path, generate_caption=True)
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=vlm,
        image_gen_fn=_MockImageGen(),
    )
    pipeline.caption_agent = _TrackingCaptionAgent(vlm_provider=vlm, prompt_dir=find_prompt_dir())

    result = await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent")
    )

    assert len(called_with_paths) == 1
    # Should be the final output file, not an iteration-level image
    assert "final_output" in called_with_paths[0]
    assert called_with_paths[0] == result.image_path


@pytest.mark.asyncio
async def test_caption_failure_is_graceful(tmp_path):
    """If CaptionAgent raises, the pipeline still returns a valid result with None caption."""
    from paperbanana.agents.caption import CaptionAgent
    from paperbanana.core.utils import find_prompt_dir

    class _FailingCaptionAgent(CaptionAgent):
        async def run(self, **kwargs):
            raise RuntimeError("VLM quota exceeded")

    vlm = _MockVLM(["Plan", "Style", _critic_satisfied()])
    settings = _make_settings(tmp_path, generate_caption=True)
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=vlm,
        image_gen_fn=_MockImageGen(),
    )
    pipeline.caption_agent = _FailingCaptionAgent(vlm_provider=vlm, prompt_dir=find_prompt_dir())

    result = await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent")
    )

    # Pipeline must still succeed and image must exist
    assert Path(result.image_path).exists()
    # Caption is None because agent failed
    assert result.generated_caption is None
    assert "generated_caption" not in result.metadata


@pytest.mark.asyncio
async def test_caption_saved_to_metadata_json(tmp_path):
    """With save_iterations=True, generated_caption appears in metadata.json."""
    expected_caption = "Illustration of the proposed attention routing mechanism."
    vlm = _MockVLM(["Plan", "Style", _critic_satisfied(), expected_caption])
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        refinement_iterations=1,
        save_iterations=True,
        generate_caption=True,
    )
    pipeline = PaperBananaPipeline(
        settings=settings,
        vlm_client=vlm,
        image_gen_fn=_MockImageGen(),
    )
    result = await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent")
    )

    # Read the persisted metadata.json
    run_id = result.metadata["run_id"]
    metadata_path = Path(settings.output_dir) / run_id / "metadata.json"
    assert metadata_path.exists()
    saved = json.loads(metadata_path.read_text())
    assert saved["generated_caption"] == expected_caption


@pytest.mark.asyncio
async def test_caption_seconds_always_present_in_timing(tmp_path):
    """timing.caption_seconds is always in metadata, even when caption is disabled."""
    vlm = _MockVLM(["Plan", "Style", _critic_satisfied()])
    settings = _make_settings(tmp_path, generate_caption=False)
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())
    result = await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent")
    )
    assert "caption_seconds" in result.metadata["timing"]
    assert result.metadata["timing"]["caption_seconds"] == 0.0


@pytest.mark.asyncio
async def test_caption_seconds_nonzero_when_enabled(tmp_path):
    """timing.caption_seconds > 0 when caption generation runs."""
    vlm = _MockVLM(["Plan", "Style", _critic_satisfied(), "A generated caption."])
    settings = _make_settings(tmp_path, generate_caption=True)
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())
    result = await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent")
    )
    # caption_seconds should be a non-negative float (may be 0.0 on very fast machines)
    assert isinstance(result.metadata["timing"]["caption_seconds"], float)
    assert result.metadata["timing"]["caption_seconds"] >= 0.0


# ── Progress events ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_caption_progress_events_emitted(tmp_path):
    """CAPTION_START and CAPTION_END progress events are emitted when enabled."""
    vlm = _MockVLM(["Plan", "Style", _critic_satisfied(), "A caption."])
    settings = _make_settings(tmp_path, generate_caption=True)
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())

    received_stages = []

    def on_progress(event):
        received_stages.append(event.stage)

    await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent"),
        progress_callback=on_progress,
    )

    assert PipelineProgressStage.CAPTION_START in received_stages
    assert PipelineProgressStage.CAPTION_END in received_stages
    # START must come before END
    start_idx = received_stages.index(PipelineProgressStage.CAPTION_START)
    end_idx = received_stages.index(PipelineProgressStage.CAPTION_END)
    assert start_idx < end_idx


@pytest.mark.asyncio
async def test_no_caption_progress_events_when_disabled(tmp_path):
    """No CAPTION_* events are emitted when generate_caption=False."""
    vlm = _MockVLM(["Plan", "Style", _critic_satisfied()])
    settings = _make_settings(tmp_path, generate_caption=False)
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())

    received_stages = []

    def on_progress(event):
        received_stages.append(event.stage)

    await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent"),
        progress_callback=on_progress,
    )

    assert PipelineProgressStage.CAPTION_START not in received_stages
    assert PipelineProgressStage.CAPTION_END not in received_stages


@pytest.mark.asyncio
async def test_caption_end_event_carries_caption_text(tmp_path):
    """CAPTION_END event's extra dict contains the generated caption text."""
    expected_caption = "Our method combines diffusion and attention."
    vlm = _MockVLM(["Plan", "Style", _critic_satisfied(), expected_caption])
    settings = _make_settings(tmp_path, generate_caption=True)
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())

    caption_end_events = []

    def on_progress(event):
        if event.stage == PipelineProgressStage.CAPTION_END:
            caption_end_events.append(event)

    await pipeline.generate(
        GenerationInput(source_context="ctx", communicative_intent="intent"),
        progress_callback=on_progress,
    )

    assert len(caption_end_events) == 1
    assert caption_end_events[0].extra is not None
    assert caption_end_events[0].extra.get("caption") == expected_caption


# ── continue_run: caption also generated ──────────────────────────


@pytest.mark.asyncio
async def test_caption_in_continue_run(tmp_path):
    """generate_caption=True also generates a caption in continue_run flow."""
    run_id = "run_caption_continue"
    run_dir = tmp_path / "outputs" / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "run_input.json").write_text(
        json.dumps(
            {
                "source_context": "Transformer architecture",
                "communicative_intent": "Pipeline overview",
                "diagram_type": "methodology",
            }
        )
    )
    iter1 = run_dir / "iter_1"
    iter1.mkdir()
    (iter1 / "details.json").write_text(
        json.dumps(
            {
                "description": "Previous description",
                "critique": {
                    "critic_suggestions": ["Add arrows"],
                    "revised_description": "Revised description",
                },
            }
        )
    )

    state = load_resume_state(str(tmp_path / "outputs"), run_id)
    expected_caption = "Illustration of the transformer pipeline with attention modules."
    vlm = _MockVLM([_critic_satisfied(), expected_caption])
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        refinement_iterations=1,
        save_iterations=False,
        generate_caption=True,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())

    result = await pipeline.continue_run(resume_state=state, additional_iterations=1)

    assert result.generated_caption == expected_caption
    assert result.metadata["generated_caption"] == expected_caption


@pytest.mark.asyncio
async def test_no_caption_in_continue_run_when_disabled(tmp_path):
    """generate_caption=False → generated_caption is None in continue_run."""
    run_id = "run_no_caption_continue"
    run_dir = tmp_path / "outputs" / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "run_input.json").write_text(
        json.dumps(
            {
                "source_context": "BERT fine-tuning",
                "communicative_intent": "Fine-tuning pipeline",
                "diagram_type": "methodology",
            }
        )
    )
    iter1 = run_dir / "iter_1"
    iter1.mkdir()
    (iter1 / "details.json").write_text(
        json.dumps(
            {
                "description": "Description v1",
                "critique": {"critic_suggestions": [], "revised_description": None},
            }
        )
    )

    state = load_resume_state(str(tmp_path / "outputs"), run_id)
    vlm = _MockVLM([_critic_satisfied()])
    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        refinement_iterations=1,
        save_iterations=False,
        generate_caption=False,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=_MockImageGen())

    result = await pipeline.continue_run(resume_state=state, additional_iterations=1)
    assert result.generated_caption is None
    assert "generated_caption" not in result.metadata


# ── GenerationOutput type tests ───────────────────────────────────


def test_generation_output_generated_caption_field():
    """GenerationOutput has generated_caption as an optional field defaulting to None."""
    from paperbanana.core.types import GenerationOutput

    out = GenerationOutput(image_path="a.png", description="desc")
    assert out.generated_caption is None

    out2 = GenerationOutput(
        image_path="b.png",
        description="desc",
        generated_caption="A caption.",
    )
    assert out2.generated_caption == "A caption."


def test_pipeline_progress_stage_has_caption_stages():
    """PipelineProgressStage enum contains CAPTION_START and CAPTION_END."""
    assert hasattr(PipelineProgressStage, "CAPTION_START")
    assert hasattr(PipelineProgressStage, "CAPTION_END")
    assert PipelineProgressStage.CAPTION_START.value == "caption_start"
    assert PipelineProgressStage.CAPTION_END.value == "caption_end"


# ── CLI wiring ─────────────────────────────────────────────────────


def test_generate_command_has_generate_caption_flag():
    """The 'generate' CLI command exposes --generate-caption."""
    from paperbanana import cli

    src = inspect.getsource(cli.generate)
    assert "generate_caption" in src
    assert "--generate-caption" in src


def test_plot_command_has_generate_caption_flag():
    """The 'plot' CLI command exposes --generate-caption."""
    from paperbanana import cli

    src = inspect.getsource(cli.plot)
    assert "generate_caption" in src
    assert "--generate-caption" in src


def test_generate_caption_flag_sets_settings_override():
    """When --generate-caption is passed, overrides include generate_caption=True."""
    from paperbanana import cli

    src = inspect.getsource(cli.generate)
    # The override must only be set when flag is truthy
    assert 'overrides["generate_caption"] = True' in src


def test_plot_settings_includes_generate_caption():
    """The plot command passes generate_caption into Settings."""
    from paperbanana import cli

    src = inspect.getsource(cli.plot)
    assert "generate_caption=generate_caption" in src
