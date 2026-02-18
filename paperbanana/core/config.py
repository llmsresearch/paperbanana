"""Configuration management for PaperBanana."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

OutputFormat = Literal["png", "jpeg", "webp"]


class VLMConfig(BaseSettings):
    """VLM provider configuration."""

    provider: str = "openai"
    model: str = "gpt-5.2"


class ImageConfig(BaseSettings):
    """Image generation provider configuration."""

    provider: str = "openai_imagen"
    model: str = "gpt-image-1.5"


class PipelineConfig(BaseSettings):
    """Pipeline execution configuration."""

    num_retrieval_examples: int = 10
    refinement_iterations: int = 3
    output_resolution: str = "2k"
    diagram_type: str = "methodology"


class ReferenceConfig(BaseSettings):
    """Reference set configuration."""

    path: str = "data/reference_sets"
    guidelines_path: str = "data/guidelines"


class OutputConfig(BaseSettings):
    """Output configuration."""

    dir: str = "outputs"
    format: str = "png"
    save_iterations: bool = True
    save_prompts: bool = True
    save_metadata: bool = True


class Settings(BaseSettings):
    """Main PaperBanana settings, loaded from env vars and config files."""

    # Provider settings
    vlm_provider: str = "openai"
    vlm_model: str = "gpt-5.2"
    image_provider: str = "openai_imagen"
    image_model: str = "gpt-image-1.5"

    # Pipeline settings
    num_retrieval_examples: int = 10
    refinement_iterations: int = 3
    auto_refine: bool = False
    max_iterations: int = 30
    optimize_inputs: bool = False
    output_resolution: str = "2k"

    # Reference settings
    reference_set_path: str = "data/reference_sets"
    guidelines_path: str = "data/guidelines"

    # Output settings
    output_dir: str = "outputs"
    output_format: OutputFormat = "png"
    save_iterations: bool = True

    # API Keys (loaded from environment)
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_vlm_model: Optional[str] = Field(default=None, alias="OPENAI_VLM_MODEL")
    openai_image_model: Optional[str] = Field(default=None, alias="OPENAI_IMAGE_MODEL")

    # SSL
    skip_ssl_verification: bool = Field(default=False, alias="SKIP_SSL_VERIFICATION")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @field_validator("output_format", mode="before")
    @classmethod
    def validate_output_format(cls, v: Any) -> str:
        """Validate output_format is png, jpeg, or webp (case-insensitive)."""
        if v is None:
            return "png"
        v = str(v).lower()
        if v not in ("png", "jpeg", "webp"):
            raise ValueError(f"output_format must be png, jpeg, or webp. Got: {v}")
        return v

    @classmethod
    def from_yaml(cls, config_path: str | Path, **overrides: Any) -> Settings:
        """Load settings from a YAML config file with optional overrides."""
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
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
        "pipeline.auto_refine": "auto_refine",
        "pipeline.max_iterations": "max_iterations",
        "pipeline.optimize_inputs": "optimize_inputs",
        "pipeline.output_resolution": "output_resolution",
        "reference.path": "reference_set_path",
        "reference.guidelines_path": "guidelines_path",
        "output.dir": "output_dir",
        "output.format": "output_format",
        "output.save_iterations": "save_iterations",
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
