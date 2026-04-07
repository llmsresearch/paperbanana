"""Health-check logic for `paperbanana doctor`."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markup import escape as markup_escape
from rich.table import Table

console = Console()

# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    label: str
    ok: bool
    detail: str
    hint: Optional[str] = field(default=None)


# ── Individual checks ─────────────────────────────────────────────────────────


def check_python() -> CheckResult:
    v = sys.version_info
    return CheckResult("Python", True, f"{v.major}.{v.minor}.{v.micro}")


def check_paperbanana() -> CheckResult:
    try:
        v = pkg_version("paperbanana")
        return CheckResult("paperbanana", True, v)
    except PackageNotFoundError:
        return CheckResult("paperbanana", False, "not found")


def check_optional_package(label: str, package: str, extra: str) -> CheckResult:
    try:
        v = pkg_version(package)
        return CheckResult(label, True, v)
    except PackageNotFoundError:
        return CheckResult(label, False, "not installed", f"pip install 'paperbanana[{extra}]'")


def check_env_key(env_var: str) -> CheckResult:
    ok = bool(os.environ.get(env_var, "").strip())
    return CheckResult(env_var, ok, "set" if ok else "not set")


def check_aws_credentials() -> CheckResult:
    has_env = bool(os.environ.get("AWS_ACCESS_KEY_ID", "").strip())
    has_profile = bool(os.environ.get("AWS_PROFILE", "").strip())
    has_file = Path.home().joinpath(".aws", "credentials").exists()
    ok = has_env or has_profile or has_file
    detail = "configured" if ok else "not configured"
    hint = (
        None
        if ok
        else "see: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html"
    )
    return CheckResult("AWS credentials", ok, detail, hint)


def check_builtin_refs() -> CheckResult:
    from paperbanana.data.manager import resolve_reference_path

    ref_path = Path(resolve_reference_path("data/reference_sets"))
    index = ref_path / "index.json"
    if not index.exists():
        return CheckResult("Built-in set", False, "index missing")
    try:
        import json

        data = json.loads(index.read_text(encoding="utf-8"))
        count = len(data.get("examples", []))
        return CheckResult("Built-in set", True, f"{count} diagrams")
    except Exception:
        return CheckResult("Built-in set", False, "unreadable")


def check_expanded_refs() -> CheckResult:
    from paperbanana.data.manager import DatasetManager

    dm = DatasetManager()
    if not dm.is_downloaded():
        return CheckResult("Expanded set", False, "not downloaded", "paperbanana data download")
    info = dm.get_info() or {}
    count = info.get("example_count") or dm.get_example_count()
    datasets = info.get("datasets", [])
    label = f"{count} diagrams ({', '.join(datasets)})" if datasets else f"{count} diagrams"
    return CheckResult("Expanded set", True, label)


# ── Rendering ─────────────────────────────────────────────────────────────────


def _status(ok: bool) -> str:
    return "[green]✓[/green]" if ok else "[red]✗[/red]"


def _render_section(title: str, results: list[CheckResult]) -> None:
    console.print(f"\n  [bold]{title}[/bold]")
    t = Table.grid(padding=(0, 2))
    t.add_column(width=24)
    t.add_column(width=22)
    t.add_column(width=4)
    t.add_column()
    for r in results:
        hint = f"[dim]{markup_escape(r.hint)}[/dim]" if r.hint and not r.ok else ""
        t.add_row(f"  {r.label}", r.detail, _status(r.ok), hint)
    console.print(t)


# ── Orchestration ─────────────────────────────────────────────────────────────

_OPTIONAL_PACKAGES = [
    ("PDF (pymupdf)", "pymupdf", "pdf"),
    ("Studio (gradio)", "gradio", "studio"),
    ("OpenAI", "openai", "openai"),
    ("Google (google-genai)", "google-genai", "google"),
    ("Anthropic", "anthropic", "anthropic"),
    ("Bedrock (boto3)", "boto3", "bedrock"),
]

_API_KEYS = [
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
]


def run_doctor() -> int:
    """Run all health checks. Returns 0 if healthy, 1 if any check fails."""
    from paperbanana import __version__

    console.print(f"\n[bold]PaperBanana v{__version__}[/bold] — System Check")

    runtime = [check_python(), check_paperbanana()]
    optional = [check_optional_package(*args) for args in _OPTIONAL_PACKAGES]
    api_keys = [check_env_key(k) for k in _API_KEYS] + [check_aws_credentials()]
    refs = [check_builtin_refs(), check_expanded_refs()]

    _render_section("Runtime", runtime)
    _render_section("Optional features", optional)
    _render_section("API keys", api_keys)
    _render_section("Reference data", refs)

    all_results = runtime + optional + api_keys + refs
    failures = [r for r in all_results if not r.ok]

    console.print()
    if not failures:
        console.print("  [green]All checks passed.[/green]\n")
        return 0

    console.print(f"  [red]{len(failures)} issue(s) found.[/red]\n")
    return 1
