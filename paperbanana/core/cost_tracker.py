"""Cost tracking and budget guard for PaperBanana pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from paperbanana.core.pricing import lookup_image_price, lookup_vlm_price

logger = structlog.get_logger()


@dataclass
class CostEntry:
    """A single API call's cost record."""

    provider: str
    model: str
    call_type: str  # "vlm" or "image_gen"
    agent: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float = 0.0
    pricing_known: bool = True


@dataclass
class CostTracker:
    """Accumulates API call costs and enforces an optional budget cap.

    Injected into providers via their ``cost_tracker`` attribute.
    Providers call ``record_vlm_call`` / ``record_image_call`` after each API
    invocation; the tracker prices the call and checks the budget.
    """

    budget: float | None = None
    _entries: list[CostEntry] = field(default_factory=list)
    _current_agent: str = ""

    def set_agent(self, agent: str) -> None:
        """Set the current agent name for subsequent API call records."""
        self._current_agent = agent

    def record_vlm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent: str = "",
    ) -> None:
        """Record a VLM API call and check budget."""
        agent = agent or self._current_agent
        pricing = lookup_vlm_price(provider, model)
        if pricing is not None:
            cost = (
                input_tokens * pricing["input_per_1k"] / 1000
                + output_tokens * pricing["output_per_1k"] / 1000
            )
            known = True
        else:
            cost = 0.0
            known = False

        entry = CostEntry(
            provider=provider,
            model=model,
            call_type="vlm",
            agent=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            pricing_known=known,
        )
        self._entries.append(entry)
        logger.debug(
            "Cost tracked (VLM)",
            agent=agent,
            cost=f"${cost:.6f}",
            total=f"${self.total_cost:.6f}",
        )
        if self.is_over_budget:
            logger.warning(
                "Budget exceeded during VLM call",
                agent=agent,
                budget=self.budget,
                spent=f"${self.total_cost:.6f}",
            )

    def record_image_call(
        self,
        provider: str,
        model: str,
        agent: str = "",
        count: int = 1,
    ) -> None:
        """Record an image generation API call and check budget."""
        agent = agent or self._current_agent
        price = lookup_image_price(provider, model)
        if price is not None:
            cost = price * count
            known = True
        else:
            cost = 0.0
            known = False

        entry = CostEntry(
            provider=provider,
            model=model,
            call_type="image_gen",
            agent=agent,
            cost_usd=cost,
            pricing_known=known,
        )
        self._entries.append(entry)
        logger.debug(
            "Cost tracked (image)",
            agent=agent,
            cost=f"${cost:.6f}",
            total=f"${self.total_cost:.6f}",
        )
        if self.is_over_budget:
            logger.warning(
                "Budget exceeded during image call",
                agent=agent,
                budget=self.budget,
                spent=f"${self.total_cost:.6f}",
            )

    @property
    def is_over_budget(self) -> bool:
        """Return True if cumulative cost exceeds the budget cap."""
        return self.budget is not None and self.total_cost > self.budget

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self._entries)

    @property
    def vlm_cost(self) -> float:
        return sum(e.cost_usd for e in self._entries if e.call_type == "vlm")

    @property
    def image_cost(self) -> float:
        return sum(e.cost_usd for e in self._entries if e.call_type == "image_gen")

    @property
    def pricing_complete(self) -> bool:
        return all(e.pricing_known for e in self._entries)

    @property
    def entries(self) -> list[CostEntry]:
        return list(self._entries)

    def summary(self) -> dict:
        """Return a dict suitable for metadata.json and terminal display."""
        by_agent: dict[str, float] = {}
        for e in self._entries:
            by_agent[e.agent] = by_agent.get(e.agent, 0.0) + e.cost_usd

        return {
            "total_usd": round(self.total_cost, 6),
            "vlm_usd": round(self.vlm_cost, 6),
            "image_usd": round(self.image_cost, 6),
            "pricing_complete": self.pricing_complete,
            "num_vlm_calls": sum(1 for e in self._entries if e.call_type == "vlm"),
            "num_image_calls": sum(1 for e in self._entries if e.call_type == "image_gen"),
            "by_agent": {k: round(v, 6) for k, v in by_agent.items()},
        }
