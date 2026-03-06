"""Tests for auto-refine, continue-run, and critic user feedback features."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.pipeline import PaperBananaPipeline
from paperbanana.core.resume import load_resume_state
from paperbanana.core.types import CritiqueResult, DiagramType, GenerationInput

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


# ── Mock helpers for behavioral tests ─────────────────────────────


class _MockVLM:
    """Sequenced VLM mock: returns pre-configured responses in order."""

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
    """Minimal image gen stub returning a solid PIL Image."""

    name = "mock-image-gen"
    model_name = "mock-image-model"

    async def generate(self, *args, **kwargs):
        return Image.new("RGB", (128, 128), color=(255, 255, 255))


def _critic_json(suggestions: list[str], revised: str | None = None) -> str:
    """Build a JSON string matching the critic's expected output format."""
    return json.dumps({"critic_suggestions": suggestions, "revised_description": revised})


# ── Behavioral: --continue resume flow ────────────────────────────


@pytest.mark.asyncio
async def test_continue_run_resumes_from_last_iteration(tmp_path):
    """continue_run() starts iteration numbering after the resumed iteration."""
    # Set up a previous run with 2 completed iterations
    run_id = "run_resume_test"
    run_dir = tmp_path / "outputs" / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "run_input.json").write_text(
        json.dumps(
            {
                "source_context": "Encoder-decoder with attention",
                "communicative_intent": "Architecture overview",
                "diagram_type": "methodology",
            }
        )
    )
    iter2 = run_dir / "iter_2"
    iter2.mkdir()
    (iter2 / "details.json").write_text(
        json.dumps(
            {
                "description": "Description from iter 2",
                "critique": {
                    "critic_suggestions": ["Improve label clarity"],
                    "revised_description": "Revised after iter 2",
                },
            }
        )
    )

    # Load the resume state
    state = load_resume_state(str(tmp_path / "outputs"), run_id)
    assert state.last_iteration == 2
    assert state.last_description == "Revised after iter 2"

    # Mock VLM: critic returns "satisfied" immediately (empty suggestions)
    vlm = _MockVLM([_critic_json([], None)])
    image_gen = _MockImageGen()

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        refinement_iterations=2,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    result = await pipeline.continue_run(resume_state=state, additional_iterations=2)

    # Iteration numbering should start from 3 (last_iteration=2, so 2+1=3)
    assert result.iterations[0].iteration == 3
    # Critic was satisfied → only 1 iteration executed
    assert len(result.iterations) == 1
    assert result.image_path.endswith(".png")
    assert Path(result.image_path).exists()


@pytest.mark.asyncio
async def test_continue_run_passes_user_feedback_to_critic(tmp_path):
    """continue_run() forwards user_feedback into the critic's prompt."""
    run_id = "run_feedback_test"
    run_dir = tmp_path / "outputs" / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "run_input.json").write_text(
        json.dumps(
            {
                "source_context": "Transformer pipeline",
                "communicative_intent": "Pipeline diagram",
                "diagram_type": "methodology",
            }
        )
    )
    iter1 = run_dir / "iter_1"
    iter1.mkdir()
    (iter1 / "details.json").write_text(
        json.dumps(
            {
                "description": "Initial desc",
                "critique": {
                    "critic_suggestions": ["Arrows too thin"],
                    "revised_description": "Desc with thicker arrows",
                },
            }
        )
    )

    state = load_resume_state(str(tmp_path / "outputs"), run_id)

    vlm = _MockVLM([_critic_json([], None)])
    image_gen = _MockImageGen()

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        refinement_iterations=1,
        save_iterations=True,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    feedback_text = "Make arrows thicker and colors more distinct"
    result = await pipeline.continue_run(
        resume_state=state, additional_iterations=1, user_feedback=feedback_text
    )

    # The critic VLM call should contain the user feedback in the prompt
    assert len(vlm.calls) >= 1
    critic_call = vlm.calls[0]
    # Prompt is passed as first positional arg or as 'prompt' kwarg
    prompt_text = critic_call["kwargs"].get(
        "prompt", critic_call["args"][0] if critic_call["args"] else ""
    )
    assert feedback_text in prompt_text

    # Metadata should record the user feedback
    assert result.metadata.get("user_feedback") == feedback_text


@pytest.mark.asyncio
async def test_continue_run_multiple_iterations_then_stops(tmp_path):
    """continue_run() runs multiple iterations and stops when critic is satisfied."""
    run_id = "run_multi_iter"
    run_dir = tmp_path / "outputs" / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "run_input.json").write_text(
        json.dumps(
            {
                "source_context": "GAN architecture",
                "communicative_intent": "GAN overview",
                "diagram_type": "methodology",
            }
        )
    )
    iter1 = run_dir / "iter_1"
    iter1.mkdir()
    (iter1 / "details.json").write_text(
        json.dumps(
            {
                "description": "Initial GAN desc",
                "critique": {
                    "critic_suggestions": ["Fix discriminator"],
                    "revised_description": "Revised GAN desc",
                },
            }
        )
    )

    state = load_resume_state(str(tmp_path / "outputs"), run_id)

    # Critic: iteration 2 → needs revision; iteration 3 → satisfied
    vlm = _MockVLM(
        [
            _critic_json(["Add loss labels"], "Desc with loss labels"),
            _critic_json([], None),
        ]
    )
    image_gen = _MockImageGen()

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        refinement_iterations=5,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    result = await pipeline.continue_run(resume_state=state, additional_iterations=5)

    # Should have run 2 iterations (iter 2 and 3), then stopped
    assert len(result.iterations) == 2
    assert result.iterations[0].iteration == 2
    assert result.iterations[1].iteration == 3
    # Final description should be the revised one from the first continuation iter
    assert result.description == "Desc with loss labels"


# ── Behavioral: --auto termination ────────────────────────────────


@pytest.mark.asyncio
async def test_auto_refine_stops_when_critic_satisfied(tmp_path):
    """With auto_refine, the loop terminates once critic returns no suggestions."""
    # VLM responses: planner → stylist → critic (iter 1: revise) → critic (iter 2: satisfied)
    vlm = _MockVLM(
        [
            "Planned description for the diagram",
            "Styled description with improved aesthetics",
            _critic_json(["Fix color contrast"], "Better contrast description"),
            _critic_json([], None),
        ]
    )
    image_gen = _MockImageGen()

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        auto_refine=True,
        max_iterations=30,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    result = await pipeline.generate(
        GenerationInput(
            source_context="Multi-head attention mechanism",
            communicative_intent="Attention overview",
        )
    )

    # auto_refine was on with max_iterations=30, but should stop at 2
    assert len(result.iterations) == 2
    assert not result.iterations[-1].critique.needs_revision


@pytest.mark.asyncio
async def test_auto_refine_respects_max_iterations_cap(tmp_path):
    """auto_refine obeys max_iterations even if critic never stops suggesting."""
    max_cap = 3

    # Critic always suggests revisions — never satisfied
    critic_always_revise = _critic_json(["Needs more detail"], "More detailed desc")
    vlm = _MockVLM(
        [
            "Planned description",
            "Styled description",
            # One critic response per iteration, always revising
            critic_always_revise,
            critic_always_revise,
            critic_always_revise,
        ]
    )
    image_gen = _MockImageGen()

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        auto_refine=True,
        max_iterations=max_cap,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    result = await pipeline.generate(
        GenerationInput(
            source_context="Diffusion model pipeline",
            communicative_intent="Diffusion overview",
        )
    )

    # Should run exactly max_iterations, not more
    assert len(result.iterations) == max_cap
    # All iterations had revisions since critic was never satisfied
    assert all(it.critique.needs_revision for it in result.iterations)


@pytest.mark.asyncio
async def test_non_auto_fixed_iterations(tmp_path):
    """Without auto_refine, pipeline runs exactly refinement_iterations."""
    # Critic always suggests revisions
    critic_always_revise = _critic_json(["Improve spacing"], "Improved spacing desc")
    vlm = _MockVLM(
        [
            "Planned description",
            "Styled description",
            critic_always_revise,
            critic_always_revise,
        ]
    )
    image_gen = _MockImageGen()

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        auto_refine=False,
        refinement_iterations=2,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    result = await pipeline.generate(
        GenerationInput(
            source_context="VAE architecture",
            communicative_intent="VAE diagram",
        )
    )

    assert len(result.iterations) == 2


@pytest.mark.asyncio
async def test_auto_refine_early_stop_on_first_iteration(tmp_path):
    """auto_refine stops after 1 iteration if critic is immediately satisfied."""
    vlm = _MockVLM(
        [
            "Planned description",
            "Styled description",
            _critic_json([], None),
        ]
    )
    image_gen = _MockImageGen()

    settings = Settings(
        output_dir=str(tmp_path / "outputs"),
        reference_set_path=str(tmp_path / "refs"),
        auto_refine=True,
        max_iterations=30,
        save_iterations=False,
    )
    pipeline = PaperBananaPipeline(settings=settings, vlm_client=vlm, image_gen_fn=image_gen)

    result = await pipeline.generate(
        GenerationInput(
            source_context="Simple encoder",
            communicative_intent="Encoder diagram",
        )
    )

    assert len(result.iterations) == 1
    assert not result.iterations[0].critique.needs_revision


# ── Behavioral: OpenAI provider execution path ────────────────────


def test_openai_vlm_provider_creation():
    """Registry creates OpenAIVLM with correct model and base_url from settings."""
    from paperbanana.providers.registry import ProviderRegistry
    from paperbanana.providers.vlm.openai import OpenAIVLM

    settings = Settings(
        vlm_provider="openai",
        openai_api_key="test-openai-key",
        openai_vlm_model="gpt-4o",
        openai_base_url="https://custom.endpoint.com/v1",
    )
    vlm = ProviderRegistry.create_vlm(settings)

    assert isinstance(vlm, OpenAIVLM)
    assert vlm.name == "openai"
    assert vlm.model_name == "gpt-4o"
    assert vlm._base_url == "https://custom.endpoint.com/v1"
    assert vlm.is_available() is True


def test_openai_vlm_falls_back_to_vlm_model():
    """When openai_vlm_model is not set, OpenAI VLM uses the generic vlm_model."""
    from paperbanana.providers.registry import ProviderRegistry
    from paperbanana.providers.vlm.openai import OpenAIVLM

    settings = Settings(
        vlm_provider="openai",
        vlm_model="gpt-5.2",
        openai_api_key="test-key",
        openai_vlm_model=None,
    )
    vlm = ProviderRegistry.create_vlm(settings)

    assert isinstance(vlm, OpenAIVLM)
    assert vlm.model_name == "gpt-5.2"


def test_openai_imagen_provider_creation():
    """Registry creates OpenAIImageGen with correct model and base_url."""
    from paperbanana.providers.image_gen.openai_imagen import OpenAIImageGen
    from paperbanana.providers.registry import ProviderRegistry

    settings = Settings(
        image_provider="openai_imagen",
        openai_api_key="test-openai-key",
        openai_image_model="gpt-image-1.5",
        openai_base_url="https://custom.endpoint.com/v1",
    )
    gen = ProviderRegistry.create_image_gen(settings)

    assert isinstance(gen, OpenAIImageGen)
    assert gen.name == "openai_imagen"
    assert gen.model_name == "gpt-image-1.5"
    assert gen._base_url == "https://custom.endpoint.com/v1"
    assert gen.is_available() is True


def test_openai_missing_api_key_raises_helpful_error():
    """Missing OPENAI_API_KEY raises ValueError with setup instructions."""
    from paperbanana.providers.registry import ProviderRegistry

    settings = Settings(vlm_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY not found") as exc_info:
        ProviderRegistry.create_vlm(settings)
    error_msg = str(exc_info.value)
    assert "platform.openai.com" in error_msg
    assert "export OPENAI_API_KEY" in error_msg


def test_openai_vlm_not_available_without_key():
    """OpenAIVLM.is_available() returns False when no API key is provided."""
    from paperbanana.providers.vlm.openai import OpenAIVLM

    vlm = OpenAIVLM(api_key=None)
    assert vlm.is_available() is False


@pytest.mark.asyncio
async def test_openai_vlm_generate_builds_correct_messages():
    """OpenAIVLM.generate() builds the correct message payload for the SDK."""
    from paperbanana.providers.vlm.openai import OpenAIVLM

    # Create a mock async client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Generated analysis"
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    vlm = OpenAIVLM(api_key="test-key", model="gpt-4o")
    vlm._client = mock_client  # Inject mock to skip real SDK init

    result = await vlm.generate(
        prompt="Analyze this diagram",
        system_prompt="You are an expert",
        temperature=0.5,
        response_format="json",
    )

    assert result == "Generated analysis"

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["response_format"] == {"type": "json_object"}

    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are an expert"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_openai_vlm_generate_encodes_images_as_base64():
    """OpenAIVLM.generate() encodes images as base64 data URIs in the payload."""
    from paperbanana.providers.vlm.openai import OpenAIVLM

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Image analysis"
    mock_response.usage = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    vlm = OpenAIVLM(api_key="test-key", model="gpt-4o")
    vlm._client = mock_client

    test_image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    result = await vlm.generate(prompt="Describe this image", images=[test_image])

    assert result == "Image analysis"

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    user_content = call_kwargs["messages"][-1]["content"]

    # Should have image_url entry and text entry
    assert len(user_content) == 2
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_content[1]["type"] == "text"
    assert user_content[1]["text"] == "Describe this image"


@pytest.mark.asyncio
async def test_openai_imagen_generate_returns_pil_image():
    """OpenAIImageGen.generate() decodes base64 response into a PIL Image."""
    import base64
    from io import BytesIO

    from paperbanana.providers.image_gen.openai_imagen import OpenAIImageGen

    # Create a real base64-encoded image for the mock to return
    img = Image.new("RGB", (64, 64), color=(0, 128, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode()

    mock_result = MagicMock()
    mock_result.data = [MagicMock()]
    mock_result.data[0].b64_json = b64_data

    mock_client = AsyncMock()
    mock_client.images.generate = AsyncMock(return_value=mock_result)

    gen = OpenAIImageGen(api_key="test-key", model="gpt-image-1.5")
    gen._client = mock_client

    result = await gen.generate(prompt="A methodology diagram", width=1792, height=1024)

    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)

    call_kwargs = mock_client.images.generate.call_args[1]
    assert call_kwargs["model"] == "gpt-image-1.5"
    assert call_kwargs["n"] == 1
    # 1792x1024 is landscape, should map to "1536x1024"
    assert call_kwargs["size"] == "1536x1024"


def test_openai_imagen_size_mapping():
    """OpenAIImageGen maps pixel dimensions to the correct OpenAI size strings."""
    from paperbanana.providers.image_gen.openai_imagen import OpenAIImageGen

    gen = OpenAIImageGen(api_key="test-key")

    # Landscape
    assert gen._size_string(1792, 1024) == "1536x1024"
    # Portrait
    assert gen._size_string(1024, 1792) == "1024x1536"
    # Square
    assert gen._size_string(1024, 1024) == "1024x1024"
    # Near-square (within threshold)
    assert gen._size_string(1100, 1024) == "1024x1024"


@pytest.mark.asyncio
async def test_openai_imagen_appends_negative_prompt():
    """OpenAIImageGen.generate() appends negative prompt to the main prompt."""
    import base64
    from io import BytesIO

    from paperbanana.providers.image_gen.openai_imagen import OpenAIImageGen

    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode()

    mock_result = MagicMock()
    mock_result.data = [MagicMock()]
    mock_result.data[0].b64_json = b64_data

    mock_client = AsyncMock()
    mock_client.images.generate = AsyncMock(return_value=mock_result)

    gen = OpenAIImageGen(api_key="test-key")
    gen._client = mock_client

    await gen.generate(
        prompt="A clean architecture diagram",
        negative_prompt="blurry, low quality",
    )

    call_kwargs = mock_client.images.generate.call_args[1]
    sent_prompt = call_kwargs["prompt"]
    assert "A clean architecture diagram" in sent_prompt
    assert "Avoid: blurry, low quality" in sent_prompt
