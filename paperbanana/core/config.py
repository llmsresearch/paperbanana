"""Configuration management for PaperBanana."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class VLMConfig(BaseSettings):
    """VLM provider configuration."""

    provider: str = "gemini"
    model: str = "auto"


class ImageConfig(BaseSettings):
    """Image generation provider configuration."""

    provider: str = "google_imagen"
    model: str = "gemini-3-pro-image-preview"


class PipelineConfig(BaseSettings):
    """Pipeline execution configuration."""

    num_retrieval_examples: int = 10
    refinement_iterations: int = 3
    output_resolution: str = "4k"
    diagram_type: str = "methodology"


class ReferenceConfig(BaseSettings):
    """Reference set configuration."""

    path: str = "data/reference_sets"
    guidelines_path: str = "data/guidelines"


class OutputConfig(BaseSettings):
    """Output configuration."""

    dir: str = "outputs"
    save_iterations: bool = True
    save_prompts: bool = True
    save_metadata: bool = True


class Settings(BaseSettings):
    """Main PaperBanana settings, loaded from env vars and config files."""

    # Provider settings
    vlm_provider: str = "gemini"
    vlm_model: str = "auto"
    vlm_model_flash: str = "gemini-3-flash-preview"
    vlm_model_pro: str = "gemini-3-pro-preview"
    image_provider: str = "google_imagen"
    image_model: str = "gemini-3-pro-image-preview"
    polish_image_model: str = "gemini-3-pro-image-preview"

    # Pipeline settings
    num_retrieval_examples: int = 10
    refinement_iterations: int = 3
    output_resolution: str = "4k"

    # Reference settings
    reference_set_path: str = "data/reference_sets"
    guidelines_path: str = "data/guidelines"

    # Output settings
    output_dir: str = "outputs"
    save_iterations: bool = True

    # Experiment mode (from official PaperBanana)
    exp_mode: str = Field(default="full", description="Pipeline mode: vanilla, planner, planner_stylist, planner_critic, full, polish")
    retrieval_setting: str = Field(default="auto", description="Retrieval: auto, manual, random, none")
    max_critic_rounds: int = Field(default=3, description="Max critic iteration rounds")
    batch_concurrent: int = Field(default=10, description="Max concurrent batch processing")

    # PaperBananaBench path (optional)
    paperbananabench_path: str = Field(default="", description="Path to PaperBananaBench dataset")

    # API Keys (loaded from environment)
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")

    # SSL
    skip_ssl_verification: bool = Field(default=False, alias="SKIP_SSL_VERIFICATION")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @classmethod
    def from_yaml(cls, config_path: str | Path, **overrides: Any) -> Settings:
        """Load settings from a YAML config file with optional overrides."""
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
        else:
            yaml_config = {}

        flat = _flatten_yaml(yaml_config)
        flat.update(overrides)
        return cls(**flat)


def _flatten_yaml(config: dict, prefix: str = "") -> dict:
    """Flatten nested YAML config into flat settings keys."""
    flat = {}
    key_map = {
        "vlm.provider": "vlm_provider",
        "vlm.model": "vlm_model",
        "image.provider": "image_provider",
        "image.model": "image_model",
        "pipeline.num_retrieval_examples": "num_retrieval_examples",
        "pipeline.refinement_iterations": "refinement_iterations",
        "pipeline.output_resolution": "output_resolution",
        "pipeline.exp_mode": "exp_mode",
        "pipeline.retrieval_setting": "retrieval_setting",
        "pipeline.max_critic_rounds": "max_critic_rounds",
        "pipeline.batch_concurrent": "batch_concurrent",
        "reference.path": "reference_set_path",
        "reference.guidelines_path": "guidelines_path",
        "output.dir": "output_dir",
        "output.save_iterations": "save_iterations",
        # Official PaperBanana model_config.yaml compatibility
        "defaults.model_name": "vlm_model",
        "defaults.image_model_name": "image_model",
        "defaults.polish_image_model_name": "polish_image_model",
        "api_keys.google_api_key": "google_api_key",
    }

    def _recurse(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _recurse(v, full_key)
            else:
                if full_key in key_map:
                    flat[key_map[full_key]] = v

    _recurse(config)
    return flat
