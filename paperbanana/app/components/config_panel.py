"""Shared configuration sidebar for all tabs.

Supports all PaperBanana providers with independent VLM / Image Gen selection.
Users must provide their own API keys.
"""

from __future__ import annotations

import gradio as gr

# ── Provider definitions ──────────────────────────────────────────

VLM_PROVIDERS = [
    ("Gemini (Recommended)", "gemini"),
    ("OpenRouter", "openrouter"),
    ("OpenAI", "openai"),
    ("Anthropic", "anthropic"),
    ("AWS Bedrock", "bedrock"),
]

IMAGE_PROVIDERS = [
    ("Google Imagen (Recommended)", "google_imagen"),
    ("OpenRouter Imagen", "openrouter_imagen"),
    ("OpenAI DALL-E", "openai_imagen"),
    ("AWS Bedrock Imagen", "bedrock_imagen"),
]

VLM_PROVIDER_CHOICES = [label for label, _ in VLM_PROVIDERS]
VLM_PROVIDER_MAP = {label: value for label, value in VLM_PROVIDERS}

IMAGE_PROVIDER_CHOICES = [label for label, _ in IMAGE_PROVIDERS]
IMAGE_PROVIDER_MAP = {label: value for label, value in IMAGE_PROVIDERS}

DEFAULT_VLM_MODELS = {
    "gemini": "gemini-3.1-flash-lite-preview",
    "openrouter": "google/gemini-2.0-flash-001",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0",
}

DEFAULT_IMAGE_MODELS = {
    "google_imagen": "gemini-3-pro-image-preview",
    "openrouter_imagen": "google/gemini-3-pro-image-preview",
    "openai_imagen": "dall-e-3",
    "bedrock_imagen": "stability.sd3-large-v1:0",
}

# Which API key each provider needs
VLM_KEY_LABEL = {
    "gemini": "Google API Key",
    "openrouter": "OpenRouter API Key",
    "openai": "OpenAI API Key",
    "anthropic": "Anthropic API Key",
    "bedrock": "AWS credentials (configure via aws configure)",
}

IMAGE_KEY_LABEL = {
    "google_imagen": "Google API Key",
    "openrouter_imagen": "OpenRouter API Key",
    "openai_imagen": "OpenAI API Key",
    "bedrock_imagen": "AWS credentials",
}

# API key help URLs
API_KEY_URLS = {
    "gemini": "https://makersuite.google.com/app/apikey",
    "openrouter": "https://openrouter.ai/keys",
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "bedrock": "https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html",
}

RESOLUTIONS = ["1k", "2k", "4k"]


def create_config_panel() -> dict[str, gr.components.Component]:
    """Create the configuration sidebar and return component references."""

    gr.Markdown("### Settings")
    gr.Markdown(
        "*Bring your own API key. "
        "[Gemini](https://makersuite.google.com/app/apikey) is free and recommended.*"
    )

    # ── VLM Provider ──
    gr.Markdown("#### VLM (Text Analysis)")
    vlm_provider = gr.Dropdown(
        choices=VLM_PROVIDER_CHOICES,
        value="Gemini (Recommended)",
        label="VLM Provider",
        interactive=True,
    )
    vlm_model = gr.Textbox(
        value=DEFAULT_VLM_MODELS["gemini"],
        label="VLM Model",
        interactive=True,
    )

    # ── Image Gen Provider ──
    gr.Markdown("#### Image Generation")
    image_provider = gr.Dropdown(
        choices=IMAGE_PROVIDER_CHOICES,
        value="Google Imagen (Recommended)",
        label="Image Provider",
        interactive=True,
    )
    image_model = gr.Textbox(
        value=DEFAULT_IMAGE_MODELS["google_imagen"],
        label="Image Model",
        interactive=True,
    )

    # ── API Keys ──
    gr.Markdown("#### API Keys")
    api_key_help = gr.Markdown(
        value=f"*Get a free key: [{API_KEY_URLS['gemini']}]({API_KEY_URLS['gemini']})*"
    )
    api_key = gr.Textbox(
        value="",
        label="Google API Key",
        type="password",
        placeholder="Enter your API key here",
        interactive=True,
    )
    # Second key field — shown when VLM and image providers use different keys
    api_key_2_label = gr.Markdown(value="", visible=False)
    api_key_2 = gr.Textbox(
        value="",
        label="Second API Key",
        type="password",
        placeholder="Enter second API key if needed",
        interactive=True,
        visible=False,
    )

    # ── Pipeline Settings ──
    gr.Markdown("#### Pipeline")
    iterations = gr.Slider(
        minimum=1,
        maximum=10,
        value=3,
        step=1,
        label="Refinement Iterations",
        interactive=True,
    )
    resolution = gr.Dropdown(
        choices=RESOLUTIONS,
        value="2k",
        label="Output Resolution",
        interactive=True,
    )
    auto_refine = gr.Checkbox(
        value=False,
        label="Auto-refine (loop until critic satisfied)",
        interactive=True,
    )
    optimize_inputs = gr.Checkbox(
        value=False,
        label="Optimize Inputs (parallel enrichment)",
        interactive=True,
    )

    # ── Auto-update on provider change ──
    def on_vlm_change(vlm_label, img_label):
        vlm_val = VLM_PROVIDER_MAP.get(vlm_label, "gemini")
        img_val = IMAGE_PROVIDER_MAP.get(img_label, "google_imagen")
        vlm_default = DEFAULT_VLM_MODELS.get(vlm_val, "")
        key_label = VLM_KEY_LABEL.get(vlm_val, "API Key")
        url = API_KEY_URLS.get(vlm_val, "")
        help_text = f"*Get a key: [{url}]({url})*" if url else ""

        # Check if VLM and image providers need different keys
        vlm_key_type = _key_type(vlm_val)
        img_key_type = _key_type_image(img_val)
        needs_second = vlm_key_type != img_key_type

        second_label = ""
        if needs_second:
            img_key_label = IMAGE_KEY_LABEL.get(img_val, "Image API Key")
            img_url = _image_key_url(img_val)
            second_label = f"**{img_key_label}** — [Get key]({img_url})"

        return (
            vlm_default,
            gr.update(label=key_label),
            help_text,
            gr.update(value=second_label, visible=needs_second),
            gr.update(visible=needs_second),
        )

    def on_image_change(vlm_label, img_label):
        img_val = IMAGE_PROVIDER_MAP.get(img_label, "google_imagen")
        img_default = DEFAULT_IMAGE_MODELS.get(img_val, "")

        vlm_val = VLM_PROVIDER_MAP.get(vlm_label, "gemini")
        vlm_key_type = _key_type(vlm_val)
        img_key_type = _key_type_image(img_val)
        needs_second = vlm_key_type != img_key_type

        second_label = ""
        if needs_second:
            img_key_label = IMAGE_KEY_LABEL.get(img_val, "Image API Key")
            img_url = _image_key_url(img_val)
            second_label = f"**{img_key_label}** — [Get key]({img_url})"

        return (
            img_default,
            gr.update(value=second_label, visible=needs_second),
            gr.update(visible=needs_second),
        )

    vlm_provider.change(
        fn=on_vlm_change,
        inputs=[vlm_provider, image_provider],
        outputs=[vlm_model, api_key, api_key_help, api_key_2_label, api_key_2],
    )
    image_provider.change(
        fn=on_image_change,
        inputs=[vlm_provider, image_provider],
        outputs=[image_model, api_key_2_label, api_key_2],
    )

    return {
        "vlm_provider": vlm_provider,
        "vlm_model": vlm_model,
        "image_provider": image_provider,
        "image_model": image_model,
        "api_key": api_key,
        "api_key_2": api_key_2,
        "iterations": iterations,
        "resolution": resolution,
        "auto_refine": auto_refine,
        "optimize_inputs": optimize_inputs,
    }


def get_settings_from_config(config: dict[str, gr.components.Component]) -> list:
    """Return the list of config component references for use as Gradio inputs."""
    return [
        config["vlm_provider"],
        config["vlm_model"],
        config["image_provider"],
        config["image_model"],
        config["api_key"],
        config["api_key_2"],
        config["iterations"],
        config["resolution"],
        config["auto_refine"],
        config["optimize_inputs"],
    ]


# ── Helpers ──

def _key_type(vlm_provider: str) -> str:
    """Return the key 'family' for a VLM provider."""
    return {
        "gemini": "google",
        "openrouter": "openrouter",
        "openai": "openai",
        "anthropic": "anthropic",
        "bedrock": "aws",
    }.get(vlm_provider, vlm_provider)


def _key_type_image(image_provider: str) -> str:
    """Return the key 'family' for an image gen provider."""
    return {
        "google_imagen": "google",
        "openrouter_imagen": "openrouter",
        "openai_imagen": "openai",
        "bedrock_imagen": "aws",
    }.get(image_provider, image_provider)


def _image_key_url(image_provider: str) -> str:
    """Return the API key URL for an image gen provider."""
    return {
        "google_imagen": API_KEY_URLS["gemini"],
        "openrouter_imagen": API_KEY_URLS["openrouter"],
        "openai_imagen": API_KEY_URLS["openai"],
        "bedrock_imagen": API_KEY_URLS["bedrock"],
    }.get(image_provider, "")
