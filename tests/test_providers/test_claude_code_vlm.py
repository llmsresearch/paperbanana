"""Tests for the Claude Code CLI VLM provider."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from tenacity import stop_after_attempt, wait_none

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry
from paperbanana.providers.vlm.claude_code import ClaudeCodeVLM


@pytest.fixture(autouse=True)
def _no_retry_delay():
    """Disable retry backoff/attempts in tests for speed."""
    original = ClaudeCodeVLM.generate
    ClaudeCodeVLM.generate = original.retry_with(
        stop=stop_after_attempt(1),
        wait=wait_none(),
        reraise=True,
    )
    yield
    ClaudeCodeVLM.generate = original

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_create_claude_code_vlm_when_cli_available():
    """Registry returns ClaudeCodeVLM when `claude` is on PATH."""
    settings = Settings(vlm_provider="claude_code", vlm_model="sonnet")
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        vlm = ProviderRegistry.create_vlm(settings)
    assert isinstance(vlm, ClaudeCodeVLM)
    assert vlm.name == "claude_code"
    assert vlm.model_name == "claude-code (sonnet)"


def test_create_claude_code_vlm_raises_when_cli_missing():
    """Registry raises a helpful error when `claude` is not installed."""
    settings = Settings(vlm_provider="claude_code", vlm_model="sonnet")
    with patch("shutil.which", return_value=None):
        with pytest.raises(ValueError, match="claude CLI not found"):
            ProviderRegistry.create_vlm(settings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc_mock(
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
) -> AsyncMock:
    """Build a mock for ``asyncio.create_subprocess_exec``."""
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode

    factory = AsyncMock(return_value=proc)
    return factory


# ---------------------------------------------------------------------------
# generate – happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_basic_text() -> None:
    """Basic text prompt → parses JSON, returns result, captures session."""
    payload = json.dumps({
        "result": "hello world",
        "session_id": "sess-abc",
    }).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        result = await vlm.generate("say hi")

    assert result == "hello world"
    assert vlm._session_id == "sess-abc"

    # Verify command shape
    cmd_args = factory.call_args[0]
    assert cmd_args[0] == "claude"
    assert "-p" in cmd_args
    assert "--output-format" in cmd_args
    assert "say hi" in cmd_args[-1]


@pytest.mark.asyncio
async def test_generate_resumes_session() -> None:
    """Second call includes ``--resume`` with the captured session_id."""
    payload = json.dumps({
        "result": "ok",
        "session_id": "sess-xyz",
    }).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        vlm._session_id = "sess-prev"
        await vlm.generate("continue")

    cmd_args = factory.call_args[0]
    assert "--resume" in cmd_args
    resume_idx = cmd_args.index("--resume")
    assert cmd_args[resume_idx + 1] == "sess-prev"


@pytest.mark.asyncio
async def test_generate_non_json_fallback() -> None:
    """Falls back to raw text when the CLI outputs non-JSON."""
    factory = _make_proc_mock(stdout=b"plain text answer")

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        result = await vlm.generate("anything")

    assert result == "plain text answer"


# ---------------------------------------------------------------------------
# generate – images
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_with_images_correct_order() -> None:
    """Image preambles appear in order (Image 1, Image 2, …) before the task prompt."""
    payload = json.dumps({"result": "described"}).encode()
    factory = _make_proc_mock(stdout=payload)

    img1 = Image.new("RGB", (4, 4), color="red")
    img2 = Image.new("RGB", (4, 4), color="blue")

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        result = await vlm.generate(
            "describe both",
            images=[img1, img2],
        )

    assert result == "described"

    full_prompt: str = factory.call_args[0][-1]
    idx1 = full_prompt.index("[Image 1:")
    idx2 = full_prompt.index("[Image 2:")
    idx_task = full_prompt.index("describe both")
    assert idx1 < idx2 < idx_task


@pytest.mark.asyncio
async def test_temp_files_cleaned_up_on_success() -> None:
    """Temp image files are removed after a successful call."""
    payload = json.dumps({"result": "ok"}).encode()
    factory = _make_proc_mock(stdout=payload)

    img = Image.new("RGB", (4, 4))
    created_paths: list[str] = []

    original_mkstemp = __import__("tempfile").mkstemp

    def _tracking_mkstemp(**kwargs: Any) -> tuple[int, str]:
        fd, path = original_mkstemp(**kwargs)
        created_paths.append(path)
        return fd, path

    with (
        patch("asyncio.create_subprocess_exec", factory),
        patch("tempfile.mkstemp", side_effect=_tracking_mkstemp),
    ):
        vlm = ClaudeCodeVLM(model="sonnet")
        await vlm.generate("go", images=[img])

    import pathlib
    for p in created_paths:
        assert not pathlib.Path(p).exists(), f"temp file leaked: {p}"


@pytest.mark.asyncio
async def test_temp_files_cleaned_up_on_subprocess_failure() -> None:
    """Temp image files are removed even when the subprocess fails."""
    factory = _make_proc_mock(
        stdout=b"", stderr=b"boom", returncode=1,
    )

    img = Image.new("RGB", (4, 4))
    created_paths: list[str] = []

    original_mkstemp = __import__("tempfile").mkstemp

    def _tracking_mkstemp(**kwargs: Any) -> tuple[int, str]:
        fd, path = original_mkstemp(**kwargs)
        created_paths.append(path)
        return fd, path

    with (
        patch("asyncio.create_subprocess_exec", factory),
        patch("tempfile.mkstemp", side_effect=_tracking_mkstemp),
    ):
        vlm = ClaudeCodeVLM(model="sonnet")
        with pytest.raises(RuntimeError, match="claude CLI exited"):
            await vlm.generate("go", images=[img])

    import pathlib
    for p in created_paths:
        assert not pathlib.Path(p).exists(), f"temp file leaked: {p}"


# ---------------------------------------------------------------------------
# generate – prompt construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_passed_via_cli_flag() -> None:
    """System prompt is passed via --system-prompt flag, not in prompt body."""
    payload = json.dumps({"result": "ok"}).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        await vlm.generate(
            "do stuff",
            system_prompt="be concise",
            response_format="json",
        )

    cmd_args = list(factory.call_args[0])
    assert "--system-prompt" in cmd_args
    sp_idx = cmd_args.index("--system-prompt")
    assert cmd_args[sp_idx + 1] == "be concise"

    full_prompt: str = cmd_args[-1]
    assert "[System instructions]" not in full_prompt
    json_idx = full_prompt.index("[Output format:")
    task_idx = full_prompt.index("do stuff")
    assert json_idx < task_idx


@pytest.mark.asyncio
async def test_full_prompt_with_system_images_and_json() -> None:
    """System prompt via flag; images, json header, task in prompt body."""
    payload = json.dumps({"result": "ok"}).encode()
    factory = _make_proc_mock(stdout=payload)
    img = Image.new("RGB", (4, 4))

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        await vlm.generate(
            "analyze",
            images=[img],
            system_prompt="expert mode",
            response_format="json",
        )

    cmd_args = list(factory.call_args[0])
    sp_idx = cmd_args.index("--system-prompt")
    assert cmd_args[sp_idx + 1] == "expert mode"

    full_prompt: str = cmd_args[-1]
    assert "[System instructions]" not in full_prompt
    img_idx = full_prompt.index("[Image 1:")
    json_idx = full_prompt.index("[Output format:")
    task_idx = full_prompt.index("analyze")
    assert img_idx < json_idx < task_idx


@pytest.mark.asyncio
async def test_empty_images_list_treated_as_no_images() -> None:
    """An empty images list behaves identically to None."""
    payload = json.dumps({"result": "ok"}).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        await vlm.generate("hi", images=[])

    full_prompt: str = factory.call_args[0][-1]
    assert "[Image" not in full_prompt


@pytest.mark.asyncio
async def test_model_passed_through_to_cli() -> None:
    """The configured model appears in the CLI command."""
    payload = json.dumps({"result": "ok"}).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="opus")
        await vlm.generate("hi")

    cmd_args = factory.call_args[0]
    model_idx = list(cmd_args).index("--model")
    assert cmd_args[model_idx + 1] == "opus"


@pytest.mark.asyncio
async def test_no_resume_flag_on_first_call() -> None:
    """First call (no session) must not include --resume."""
    payload = json.dumps({"result": "ok", "session_id": "s1"}).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        await vlm.generate("first")

    cmd_args = list(factory.call_args[0])
    assert "--resume" not in cmd_args


# ---------------------------------------------------------------------------
# generate – session management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_chains_across_sequential_calls() -> None:
    """Session ID from call 1 is passed via --resume in call 2."""
    resp1 = json.dumps({"result": "r1", "session_id": "s-1"}).encode()
    resp2 = json.dumps({"result": "r2", "session_id": "s-2"}).encode()

    call_count = 0

    async def _fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        proc = AsyncMock()
        proc.communicate.return_value = (
            resp1 if call_count == 1 else resp2,
            b"",
        )
        proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
        vlm = ClaudeCodeVLM(model="sonnet")
        r1 = await vlm.generate("first")
        r2 = await vlm.generate("second")

    assert r1 == "r1"
    assert r2 == "r2"
    assert vlm._session_id == "s-2"


@pytest.mark.asyncio
async def test_session_preserved_when_response_has_no_session_id() -> None:
    """If the JSON response lacks session_id, the previous one is kept."""
    payload = json.dumps({"result": "ok"}).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        vlm._session_id = "keep-me"
        await vlm.generate("go")

    assert vlm._session_id == "keep-me"


@pytest.mark.asyncio
async def test_json_without_result_key_falls_back_to_raw() -> None:
    """Valid JSON missing the 'result' key returns the raw JSON string."""
    payload = json.dumps({"session_id": "s1", "other": "data"}).encode()
    factory = _make_proc_mock(stdout=payload)

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        result = await vlm.generate("go")

    # data.get("result", raw.strip()) → raw.strip()
    assert result == payload.decode().strip()


@pytest.mark.asyncio
async def test_concurrent_calls_chain_session_ids() -> None:
    """After concurrent generate() calls, session_id reflects the last."""
    call_count = 0

    async def _fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        n = call_count
        await asyncio.sleep(0.02)
        proc = AsyncMock()
        proc.communicate.return_value = (
            json.dumps({
                "result": f"r{n}",
                "session_id": f"s-{n}",
            }).encode(),
            b"",
        )
        proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
        vlm = ClaudeCodeVLM(model="sonnet")
        results = await asyncio.gather(
            vlm.generate("a"),
            vlm.generate("b"),
        )

    # Both completed; session_id should be from the second (serialised)
    assert len(results) == 2
    assert vlm._session_id == "s-2"


# ---------------------------------------------------------------------------
# generate – error edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_prefers_stderr_over_stdout() -> None:
    """RuntimeError message uses stderr when both stderr and stdout exist."""
    factory = _make_proc_mock(
        stdout=b"stdout noise",
        stderr=b"real error",
        returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        with pytest.raises(RuntimeError, match="real error"):
            await vlm.generate("go")


@pytest.mark.asyncio
async def test_error_falls_back_to_stdout_when_stderr_empty() -> None:
    """RuntimeError message uses stdout when stderr is empty."""
    factory = _make_proc_mock(
        stdout=b"error on stdout",
        stderr=b"",
        returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        with pytest.raises(RuntimeError, match="error on stdout"):
            await vlm.generate("go")


@pytest.mark.asyncio
async def test_error_message_truncated_to_500_chars() -> None:
    """Very long error output is truncated in the RuntimeError."""
    long_err = b"E" * 1000
    factory = _make_proc_mock(
        stdout=b"", stderr=long_err, returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        with pytest.raises(RuntimeError) as exc_info:
            await vlm.generate("go")

    # The error message should contain at most 500 E's
    err_str = str(exc_info.value)
    assert err_str.count("E") <= 500


@pytest.mark.asyncio
async def test_temp_files_cleaned_up_when_exec_raises() -> None:
    """If create_subprocess_exec raises OSError, temp files are still removed."""
    img = Image.new("RGB", (4, 4))
    created_paths: list[str] = []

    original_mkstemp = __import__("tempfile").mkstemp

    def _tracking_mkstemp(**kwargs: Any) -> tuple[int, str]:
        fd, path = original_mkstemp(**kwargs)
        created_paths.append(path)
        return fd, path

    async def _raise_exec(*args: Any, **kwargs: Any) -> Any:
        raise OSError("claude binary not found")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_raise_exec),
        patch("tempfile.mkstemp", side_effect=_tracking_mkstemp),
    ):
        vlm = ClaudeCodeVLM(model="sonnet")
        with pytest.raises(OSError, match="claude binary not found"):
            await vlm.generate("go", images=[img])

    import pathlib

    assert created_paths, "expected at least one temp file"
    for p in created_paths:
        assert not pathlib.Path(p).exists(), f"temp file leaked: {p}"


# ---------------------------------------------------------------------------
# generate – warnings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_warns_on_non_default_temperature() -> None:
    """A warning is logged when temperature or max_tokens differ from defaults."""
    payload = json.dumps({"result": "ok"}).encode()
    factory = _make_proc_mock(stdout=payload)

    with (
        patch("asyncio.create_subprocess_exec", factory),
        patch("paperbanana.providers.vlm.claude_code.logger") as mock_logger,
    ):
        vlm = ClaudeCodeVLM(model="sonnet")
        await vlm.generate("hi", temperature=0.5, max_tokens=1024)

    mock_logger.warning.assert_called_once()
    msg = mock_logger.warning.call_args[0][0]
    assert "does not support temperature/max_tokens" in msg


# ---------------------------------------------------------------------------
# generate – error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_cli_error_raises() -> None:
    """Non-zero exit code raises RuntimeError with stderr context."""
    factory = _make_proc_mock(
        stdout=b"", stderr=b"segfault", returncode=1,
    )

    with patch("asyncio.create_subprocess_exec", factory):
        vlm = ClaudeCodeVLM(model="sonnet")
        with pytest.raises(RuntimeError, match="claude CLI exited"):
            await vlm.generate("crash")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_are_serialised() -> None:
    """Concurrent generate() calls are serialised by the lock."""
    call_order: list[str] = []

    async def _slow_exec(*args: Any, **kwargs: Any) -> Any:
        call_order.append("start")
        await asyncio.sleep(0.05)
        call_order.append("end")
        proc = AsyncMock()
        proc.communicate.return_value = (
            json.dumps({"result": "ok"}).encode(),
            b"",
        )
        proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=_slow_exec):
        vlm = ClaudeCodeVLM(model="sonnet")
        await asyncio.gather(
            vlm.generate("a"),
            vlm.generate("b"),
        )

    # With serialisation the pattern is start/end/start/end,
    # never start/start.
    assert call_order == ["start", "end", "start", "end"]
