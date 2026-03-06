"""External exemplar retrieval client and mapping helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from paperbanana.core.types import DiagramType, ReferenceExample
from paperbanana.reference.store import ReferenceStore

logger = structlog.get_logger()


class ExemplarRetrievalError(RuntimeError):
    """Raised when external exemplar retrieval fails."""


class ExemplarHit(BaseModel):
    """Normalized external exemplar payload."""

    id: str
    caption: str = ""
    source_context: str = ""
    image_path: str = ""
    score: Optional[float] = None
    structure_hints: Optional[dict[str, Any] | list[Any] | str] = None


class ExternalExemplarRetriever:
    """Client for external exemplar retrieval adapters."""

    _RESULT_KEYS = ("exemplars", "results", "items", "hits")

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        base_backoff_seconds: float = 0.5,
    ):
        endpoint = endpoint.strip()
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be >= 0")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds

    async def retrieve(
        self,
        source_context: str,
        caption: str,
        diagram_type: DiagramType,
        top_k: int,
    ) -> list[ExemplarHit]:
        """Retrieve top-k exemplars from an external adapter."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        payload = {
            "source_context": source_context,
            "caption": caption,
            "diagram_type": diagram_type.value,
            "top_k": top_k,
            "query": {
                "source_context": source_context,
                "caption": caption,
                "diagram_type": diagram_type.value,
            },
        }

        body = await self._post_with_retries(payload)

        raw_items = self._extract_items(body)
        hits: list[ExemplarHit] = []
        seen_ids: set[str] = set()

        for item in raw_items:
            try:
                hit = self._parse_hit(item)
            except (ValidationError, ValueError) as e:
                logger.warning("Skipping malformed external exemplar hit", error=str(e), item=item)
                continue

            if hit.id in seen_ids:
                continue
            seen_ids.add(hit.id)
            hits.append(hit)

            if len(hits) >= top_k:
                break

        return hits

    async def _post_with_retries(self, payload: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds, follow_redirects=True
                ) as client:
                    response = await client.post(self.endpoint, json=payload)
                    response.raise_for_status()
                    return response.json()
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                if attempt == self.max_retries:
                    break
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                retryable = status_code >= 500 or status_code == 429
                if not retryable or attempt == self.max_retries:
                    break
            except ValueError as e:
                raise ExemplarRetrievalError(
                    f"Failed to decode external retriever response: {e}"
                ) from e

            sleep_seconds = self.base_backoff_seconds * (2**attempt)
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

        raise ExemplarRetrievalError(
            f"Failed to query external retriever: {last_error}"
        ) from last_error

    @classmethod
    def _extract_items(cls, body: Any) -> list[Any]:
        if isinstance(body, list):
            return body
        if not isinstance(body, dict):
            raise ExemplarRetrievalError(
                "External retriever response must be a JSON object or list"
            )

        for key in cls._RESULT_KEYS:
            value = body.get(key)
            if isinstance(value, list):
                return value

        raise ExemplarRetrievalError(
            "External retriever response must contain one of: exemplars, results, items, hits"
        )

    @staticmethod
    def _parse_hit(item: Any) -> ExemplarHit:
        if isinstance(item, str):
            return ExemplarHit(id=item)
        if not isinstance(item, dict):
            raise ValueError("hit must be an object or id string")

        exemplar_id = (
            item.get("id")
            or item.get("paper_id")
            or item.get("example_id")
            or item.get("ref_id")
            or item.get("uid")
        )
        if not exemplar_id:
            raise ValueError("missing exemplar id")

        raw_score = item.get("score")
        score: Optional[float] = None
        if raw_score is not None:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = None

        return ExemplarHit(
            id=str(exemplar_id),
            caption=str(
                item.get("caption") or item.get("visual_intent") or item.get("title") or ""
            ),
            source_context=str(
                item.get("source_context")
                or item.get("context")
                or item.get("snippet")
                or item.get("text")
                or ""
            ),
            image_path=str(
                item.get("image_path") or item.get("thumbnail_url") or item.get("image_url") or ""
            ),
            score=score,
            structure_hints=(
                item.get("structure_hints") or item.get("node_edge_tags") or item.get("hints")
            ),
        )


def map_external_hits_to_examples(
    hits: list[ExemplarHit], reference_store: ReferenceStore
) -> list[ReferenceExample]:
    """Map external hits to local examples, falling back to external metadata."""
    examples: list[ReferenceExample] = []
    for hit in hits:
        local = reference_store.get_by_id(hit.id)
        if local is not None:
            examples.append(local)
            continue

        examples.append(
            ReferenceExample(
                id=hit.id,
                source_context=hit.source_context,
                caption=hit.caption,
                image_path=hit.image_path,
                structure_hints=hit.structure_hints,
            )
        )
    return examples
