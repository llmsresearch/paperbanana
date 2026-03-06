"""Tests for find_prompt_dir() — prompts directory resolution."""

from __future__ import annotations

from pathlib import Path

from paperbanana.core.utils import find_prompt_dir


class TestFindPromptDir:
    def test_finds_prompts_in_cwd(self, tmp_path: Path, monkeypatch):
        """When CWD contains a prompts/ dir with expected subdirs, use it."""
        prompts = tmp_path / "prompts" / "evaluation"
        prompts.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert find_prompt_dir() == "prompts"

    def test_falls_back_to_package_relative(self, tmp_path: Path, monkeypatch):
        """When CWD has no prompts/, resolve relative to the package."""
        # Change to a dir without prompts/
        monkeypatch.chdir(tmp_path)
        result = find_prompt_dir()
        # Should either find the real prompts dir (package-relative)
        # or fall back to "prompts"
        resolved = Path(result)
        if resolved.is_absolute() or resolved.exists():
            # Found the package-relative prompts dir
            assert (resolved / "evaluation").exists() or (resolved / "diagram").exists()
        else:
            # Fell back to default "prompts"
            assert result == "prompts"

    def test_default_when_nothing_found(self, tmp_path: Path, monkeypatch):
        """When no prompts/ dir exists anywhere, return 'prompts'."""
        monkeypatch.chdir(tmp_path)
        # Also patch __file__ resolution to a non-existent path
        import paperbanana.core.utils as utils_mod

        original_file = utils_mod.__file__
        monkeypatch.setattr(utils_mod, "__file__", str(tmp_path / "fake" / "core" / "utils.py"))
        try:
            assert find_prompt_dir() == "prompts"
        finally:
            monkeypatch.setattr(utils_mod, "__file__", original_file)

    def test_prefers_cwd_over_package(self, tmp_path: Path, monkeypatch):
        """CWD prompts/ takes priority over package-relative."""
        prompts = tmp_path / "prompts" / "diagram"
        prompts.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        # Should return the CWD-relative "prompts", not an absolute path
        assert find_prompt_dir() == "prompts"
