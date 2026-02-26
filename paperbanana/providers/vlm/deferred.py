"""Deferred VLM provider — auto-selects Flash or Pro based on input complexity."""

from __future__ import annotations

import re
from typing import Optional

import structlog
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.core.types import DiagramType, GenerationInput
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()

# Technical terms indicating complex architectures
_TECHNICAL_TERMS = {
    "encoder", "decoder", "transformer", "attention", "convolution",
    "pooling", "embedding", "fusion", "backbone", "head", "branch",
    "module", "block", "layer", "stream", "pathway", "normalization",
    "activation", "residual", "skip connection", "feature pyramid",
    "upsampling", "downsampling", "bottleneck", "discriminator",
    "generator", "classifier", "segmentation",
}

# Multi-component architecture indicators
_MULTI_COMPONENT = {
    "multi-modal", "multimodal", "multi-scale", "multiscale",
    "hierarchical", "cascade", "parallel", "cross-scale",
    "encoder-decoder", "u-net", "unet", "feature pyramid",
    "dual-branch", "two-stream", "multi-task", "multitask",
}

# Intent keywords suggesting complex output
_COMPLEX_INTENT = {
    "complex", "detailed", "comprehensive", "intricate",
    "architecture", "framework", "pipeline", "overview",
    "end-to-end", "multi-stage", "complete",
}

# Word boundary pattern for matching whole terms
_WORD_RE = re.compile(r"\b\w[\w-]*\w?\b")


def assess_complexity(input: GenerationInput) -> tuple[int, dict[str, int]]:
    """Score the complexity of a generation input.

    Returns:
        (total_score, breakdown_dict) where breakdown shows per-factor scores.
    """
    text = input.source_context.lower()
    intent = input.communicative_intent.lower()
    words = set(_WORD_RE.findall(text))
    breakdown: dict[str, int] = {}

    # Factor 1: Context length
    ctx_len = len(input.source_context)
    if ctx_len > 1500:
        breakdown["context_length"] = 2
    elif ctx_len > 600:
        breakdown["context_length"] = 1
    else:
        breakdown["context_length"] = 0

    # Factor 2: Diagram type
    breakdown["diagram_type"] = 1 if input.diagram_type == DiagramType.METHODOLOGY else 0

    # Factor 3: Technical term density
    matched_terms = _TECHNICAL_TERMS & words
    term_count = len(matched_terms)
    if term_count >= 8:
        breakdown["tech_density"] = 3
    elif term_count >= 5:
        breakdown["tech_density"] = 2
    elif term_count >= 3:
        breakdown["tech_density"] = 1
    else:
        breakdown["tech_density"] = 0

    # Factor 4: Multi-component architecture hints
    multi_count = sum(1 for term in _MULTI_COMPONENT if term in text)
    if multi_count >= 2:
        breakdown["multi_component"] = 2
    elif multi_count >= 1:
        breakdown["multi_component"] = 1
    else:
        breakdown["multi_component"] = 0

    # Factor 5: Intent keywords
    intent_count = sum(1 for kw in _COMPLEX_INTENT if kw in intent)
    if intent_count >= 2:
        breakdown["intent_complexity"] = 2
    elif intent_count >= 1:
        breakdown["intent_complexity"] = 1
    else:
        breakdown["intent_complexity"] = 0

    total = sum(breakdown.values())
    return total, breakdown


class DeferredVLMProvider(VLMProvider):
    """VLM proxy that auto-selects Flash or Pro based on input complexity.

    Transparent to all agents — implements the same VLMProvider interface.
    Call ``select_model(input)`` before the pipeline runs to resolve the
    actual model.  If ``generate()`` is called without prior selection,
    falls back to Flash for safety.
    """

    # Score threshold: >= this value selects Pro
    COMPLEXITY_THRESHOLD = 5

    def __init__(self, settings: Settings):
        self._settings = settings
        self._actual_vlm: Optional[VLMProvider] = None
        self._selected_model: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────

    def select_model(self, input: GenerationInput) -> str:
        """Analyze input complexity and initialize the chosen GeminiVLM.

        Returns the selected model name.
        """
        score, breakdown = assess_complexity(input)
        if score >= self.COMPLEXITY_THRESHOLD:
            model = self._settings.vlm_model_pro
        else:
            model = self._settings.vlm_model_flash

        logger.info(
            "Auto VLM selection",
            score=score,
            threshold=self.COMPLEXITY_THRESHOLD,
            selected=model,
            breakdown=breakdown,
        )

        self._selected_model = model
        self._actual_vlm = self._create_vlm(model)
        return model

    # ── VLMProvider interface ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._selected_model or "auto (pending)"

    def is_available(self) -> bool:
        return self._settings.google_api_key is not None

    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        response_format: Optional[str] = None,
    ) -> str:
        if self._actual_vlm is None:
            logger.warning("DeferredVLM: generate() called before select_model(), falling back to Flash")
            self._selected_model = self._settings.vlm_model_flash
            self._actual_vlm = self._create_vlm(self._selected_model)

        return await self._actual_vlm.generate(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _create_vlm(self, model: str) -> VLMProvider:
        from paperbanana.providers.vlm.gemini import GeminiVLM

        return GeminiVLM(
            api_key=self._settings.google_api_key,
            model=model,
        )
