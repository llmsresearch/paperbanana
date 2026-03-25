"""Claude Code CLI VLM provider.

Uses the locally installed `claude` CLI as the VLM backend.
No API key needed — uses the user's Claude Code subscription.
Maintains conversation context across calls via --resume.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image

from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class ClaudeCodeVLM(VLMProvider):
    """VLM provider that shells out to the `claude` CLI.

    Maintains a single conversation session across planner, stylist,
    and critic calls so each step has full context of prior steps.
    """

    def __init__(self, model: str = "sonnet"):
        self._model = model
        self._session_id: Optional[str] = None

    @property
    def name(self) -> str:
        return "claude_code"

    @property
    def model_name(self) -> str:
        return f"claude-code ({self._model})"

    def is_available(self) -> bool:
        import shutil

        return shutil.which("claude") is not None

    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        cmd = [
            "claude",
            "-p",
            "--output-format", "json",
            "--model", self._model,
        ]

        if self._session_id:
            cmd += ["--resume", self._session_id]

        # Build the full prompt with optional system context
        full_prompt = ""
        if system_prompt:
            full_prompt += f"[System instructions]\n{system_prompt}\n\n"

        if response_format == "json":
            full_prompt += (
                "[Output format: respond with valid JSON only, no markdown fences]\n\n"
            )

        full_prompt += prompt

        # Handle images by saving to temp files and referencing
        # in prompt.  We build the preamble in order then prepend
        # it so Image 1 always comes first.
        temp_files: list[Path] = []
        if images:
            preamble_parts: list[str] = []
            for i, img in enumerate(images):
                fd, tmp_name = tempfile.mkstemp(
                    suffix=f"_pb_img_{i}.png",
                )
                os.close(fd)
                tmp = Path(tmp_name)
                img.save(tmp, format="PNG")
                temp_files.append(tmp)
                preamble_parts.append(
                    f"[Image {i + 1}: see file {tmp}]\n"
                    f"Please read the image at {tmp}"
                    " before responding.\n\n"
                )
            full_prompt = "".join(preamble_parts) + full_prompt

        cmd.append(full_prompt)

        logger.info(
            "Claude Code CLI call",
            session_id=self._session_id,
            prompt_length=len(full_prompt),
            has_images=bool(images),
            num_images=len(images) if images else 0,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        finally:
            for tmp in temp_files:
                tmp.unlink(missing_ok=True)

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            stdout_msg = stdout.decode().strip()
            # Claude CLI may output errors as JSON on stdout
            combined = error_msg or stdout_msg
            logger.error(
                "Claude Code CLI failed",
                returncode=proc.returncode,
                stderr=error_msg[:500] if error_msg else None,
                stdout=stdout_msg[:500] if stdout_msg else None,
            )
            raise RuntimeError(
                "claude CLI exited with code "
                f"{proc.returncode}: {combined[:500]}"
            )

        raw = stdout.decode()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Claude Code returned non-JSON output, using raw text")
            return raw.strip()

        # Capture session_id for conversation continuity
        if "session_id" in data:
            prev = self._session_id
            self._session_id = data["session_id"]
            if prev is None:
                logger.info("Claude Code session started", session_id=self._session_id)

        result_text = data.get("result", raw.strip())

        logger.debug(
            "Claude Code response",
            model=self._model,
            session_id=self._session_id,
            duration_ms=data.get("duration_ms"),
            cost_usd=data.get("total_cost_usd"),
            result_length=len(result_text),
        )

        return result_text
