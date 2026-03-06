"""Tests for external exemplar retrieval client."""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
import pytest

from paperbanana.core.types import DiagramType
from paperbanana.reference.exemplar_retrieval import (
    ExemplarHit,
    ExemplarRetrievalError,
    ExternalExemplarRetriever,
    map_external_hits_to_examples,
)
from paperbanana.reference.store import ReferenceStore


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self._request = httpx.Request("POST", "https://retriever.test/query")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self._request,
                response=httpx.Response(self.status_code, request=self._request),
            )

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, endpoint: str, json: dict):
        if callable(self._response):
            return self._response()
        return self._response


@pytest.mark.asyncio
async def test_external_retriever_parses_and_dedupes(monkeypatch):
    response = _FakeResponse(
        200,
        {
            "results": [
                {"id": "ref_1", "caption": "A"},
                {"example_id": "ref_2", "context": "ctx 2"},
                {"id": "ref_1", "caption": "duplicate"},
                {"caption": "missing id"},
            ]
        },
    )

    monkeypatch.setattr(
        "paperbanana.reference.exemplar_retrieval.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(response),
    )

    retriever = ExternalExemplarRetriever("https://retriever.test/query")
    hits = await retriever.retrieve(
        source_context="source",
        caption="caption",
        diagram_type=DiagramType.METHODOLOGY,
        top_k=5,
    )

    assert [hit.id for hit in hits] == ["ref_1", "ref_2"]
    assert hits[1].source_context == "ctx 2"


@pytest.mark.asyncio
async def test_external_retriever_raises_on_http_error(monkeypatch):
    response = _FakeResponse(500, {"error": "server error"})
    monkeypatch.setattr(
        "paperbanana.reference.exemplar_retrieval.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(response),
    )

    retriever = ExternalExemplarRetriever("https://retriever.test/query")
    with pytest.raises(ExemplarRetrievalError):
        await retriever.retrieve(
            source_context="source",
            caption="caption",
            diagram_type=DiagramType.METHODOLOGY,
            top_k=3,
        )


@pytest.mark.asyncio
async def test_external_retriever_retries_transient_http_error(monkeypatch):
    responses = iter(
        [
            _FakeResponse(500, {"error": "server error"}),
            _FakeResponse(200, {"results": [{"id": "ref_1"}]}),
        ]
    )
    call_count = {"count": 0}

    def _next_response():
        call_count["count"] += 1
        return next(responses)

    monkeypatch.setattr(
        "paperbanana.reference.exemplar_retrieval.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(_next_response),
    )

    retriever = ExternalExemplarRetriever(
        "https://retriever.test/query",
        max_retries=2,
        base_backoff_seconds=0.0,
    )
    hits = await retriever.retrieve(
        source_context="source",
        caption="caption",
        diagram_type=DiagramType.METHODOLOGY,
        top_k=3,
    )

    assert call_count["count"] == 2
    assert [h.id for h in hits] == ["ref_1"]


@pytest.mark.asyncio
async def test_external_retriever_does_not_retry_non_retryable_4xx(monkeypatch):
    call_count = {"count": 0}

    def _resp():
        call_count["count"] += 1
        return _FakeResponse(400, {"error": "bad request"})

    monkeypatch.setattr(
        "paperbanana.reference.exemplar_retrieval.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(_resp),
    )

    retriever = ExternalExemplarRetriever(
        "https://retriever.test/query",
        max_retries=3,
        base_backoff_seconds=0.0,
    )
    with pytest.raises(ExemplarRetrievalError):
        await retriever.retrieve(
            source_context="source",
            caption="caption",
            diagram_type=DiagramType.METHODOLOGY,
            top_k=3,
        )
    assert call_count["count"] == 1


def test_map_external_hits_to_examples_preserves_structure_hints():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "index.json").write_text('{"examples":[]}', encoding="utf-8")
        store = ReferenceStore(tmpdir)
        mapped = map_external_hits_to_examples(
            [
                ExemplarHit(
                    id="ext_1",
                    caption="cap",
                    source_context="ctx",
                    image_path="",
                    structure_hints={"nodes": ["A", "B"], "edges": ["A->B"]},
                )
            ],
            store,
        )

        assert len(mapped) == 1
        assert mapped[0].id == "ext_1"
        assert mapped[0].structure_hints == {"nodes": ["A", "B"], "edges": ["A->B"]}
